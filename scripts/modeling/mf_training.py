import os
import pickle
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CyclicLR

from scripts.utils import root_dir
from scripts.modeling.mf_model import MatrixFactorization, RatingDataset

# --- 1. PARAMETERS ---
MODEL_ARTIFACTS_DIR = os.path.join(root_dir, 'models', 'mf_model_files')
EMBEDDING_DIM = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 20
BATCH_SIZE = 4096
NUM_NEG_SAMPLES = 4 # Number of negative samples per positive interaction
K_NDCG = 10

# Check for GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found. Using GPU.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device found. Using GPU.")
else:
    device = torch.device("cpu")
    print("No GPU backend found. Using CPU.")

# --- 2. DATA SETUP ---
print("Loading and preparing data...")
train_data_path = os.path.join(root_dir, 'data', 'ml-32m', 'processed', 'train_20_core.csv')
validation_data_path = os.path.join(root_dir, 'data', 'ml-32m', 'processed', 'val_10_core.csv')

train_df_orig = pd.read_csv(train_data_path)
val_df = pd.read_csv(validation_data_path)

# Filter for items with at least one 4+ rating
train_items = set(train_df_orig[train_df_orig['rating'] >= 4.0].movieId.unique())
train_users = set(train_df_orig[train_df_orig['rating'] >= 4.0].userId.unique())
train_df = train_df_orig[train_df_orig['movieId'].isin(train_items) & train_df_orig['userId'].isin(train_users)].copy()

# Create mappings from the filtered training data
uid2idx = {user_id: idx for idx, user_id in enumerate(train_df['userId'].unique())}
iid2idx = {item_id: idx for idx, item_id in enumerate(train_df['movieId'].unique())}
num_users = len(uid2idx)
num_items = len(iid2idx)
all_item_indices = set(range(num_items))

# Map IDs to indices
train_df['user_idx'] = train_df['userId'].map(uid2idx)
train_df['item_idx'] = train_df['movieId'].map(iid2idx)

# --- 3. IMPLICIT DATA PREPARATION (KEY CHANGE) ---
print("Creating implicit training data with negative sampling...")
# Get all positive interactions
positive_interactions = train_df[['user_idx', 'item_idx']].copy()
positive_interactions['rating'] = 1.0

# Group interactions to find negatives for each user
user_item_sets = train_df.groupby('user_idx')['item_idx'].apply(set)

negative_samples = []
for user_idx, seen_items in tqdm(user_item_sets.items(), desc="Negative Sampling"):
    possible_negs = all_item_indices - seen_items
    if not possible_negs:
        continue
    
    num_samples = min(len(possible_negs), len(seen_items) * NUM_NEG_SAMPLES)
    sampled_negs = np.random.choice(list(possible_negs), size=num_samples, replace=False)
    
    for neg_item_idx in sampled_negs:
        negative_samples.append({'user_idx': user_idx, 'item_idx': neg_item_idx, 'rating': 0.0})

negative_df = pd.DataFrame(negative_samples)
train_data_implicit = pd.concat([positive_interactions, negative_df], ignore_index=True)

# Create Dataset and DataLoader
train_ds = RatingDataset(
    user_indices=train_data_implicit['user_idx'].values,
    item_indices=train_data_implicit['item_idx'].values,
    ratings=train_data_implicit['rating'].values
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
print(f"Implicit training dataset created with {len(train_ds)} records.")

# --- 4. MODEL TRAINING ---
save_model_path = os.path.join(MODEL_ARTIFACTS_DIR, f'mf_implicit_model_{EMBEDDING_DIM}.pth')
model = MatrixFactorization(num_users, num_items, EMBEDDING_DIM).to(device)

# Use BCEWithLogitsLoss for implicit feedback
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

iterations_per_epoch = len(train_loader)
scheduler = CyclicLR(
    optimizer,
    base_lr=1e-5,
    max_lr=1e-3,
    step_size_up=4 * iterations_per_epoch, # 4 epochs to go from base to max
    mode='triangular',
    cycle_momentum=False
)
print("Added CyclicLR scheduler.")

print("\nStarting model training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for user_indices, item_indices, ratings in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        user_indices, item_indices, ratings = user_indices.to(device), item_indices.to(device), ratings.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(user_indices, item_indices)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()
        
        # Step the scheduler after each batch
        scheduler.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
    
    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")

# --- 5. EVALUATION ---
print("\nRunning MF model evaluation...")
model.eval()

# Prepare validation and training seen data
val_df['user_idx'] = val_df['userId'].map(uid2idx)
val_df['item_idx'] = val_df['movieId'].map(iid2idx)
val_df.dropna(subset=['user_idx', 'item_idx'], inplace=True)
val_df['user_idx'] = val_df['user_idx'].astype(int)
val_df['item_idx'] = val_df['item_idx'].astype(int)

train_seen_dict = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

def ndcg_at_k(model, val_df, train_seen_dict, k=10):
    distinct_users = val_df['user_idx'].unique()
    ndcg_scores = []
    
    all_items_tensor = torch.tensor(list(range(num_items)), device=device)

    for user in tqdm(distinct_users, desc=f"Calculating nDCG@{k}"):
        user_tensor = torch.tensor([user] * num_items, device=device)
        
        with torch.no_grad():
            scores = model(user_tensor, all_items_tensor).cpu().numpy()
        
        item_score_df = pd.DataFrame({'item_idx': list(range(num_items)), 'score': scores})
        
        seen_items = train_seen_dict.get(user, set())
        item_score_df = item_score_df[~item_score_df['item_idx'].isin(seen_items)]
        
        user_recommendations = item_score_df.sort_values('score', ascending=False).head(k)
        
        user_truth = val_df[val_df['user_idx'] == user]
        if user_truth.empty: continue
        
        item_rel_dict = user_truth.set_index('item_idx')['rating'].to_dict()
        
        # Calculate DCG
        user_recommendations['relevance'] = user_recommendations['item_idx'].map(item_rel_dict).fillna(0)
        dcg = (user_recommendations['relevance'] / np.log2(np.arange(2, len(user_recommendations) + 2))).sum()
        
        # Calculate IDCG
        ideal_relevances = user_truth['rating'].sort_values(ascending=False).head(k)
        idcg = (ideal_relevances.values / np.log2(np.arange(2, len(ideal_relevances) + 2))).sum()
        
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)

ndcg_score = ndcg_at_k(model, val_df, train_seen_dict, k=K_NDCG)
print("\n" + "="*40)
print(f"Implicit MF Model nDCG@{K_NDCG}: {ndcg_score:.4f}")
print("="*40)
