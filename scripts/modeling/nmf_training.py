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
from scripts.modeling.nmf_model import NeuralMF, NMFDataset
from scripts.utils import root_dir

# --- 1. PARAMETERS ---
MODEL_ARTIFACTS_DIR = os.path.join(root_dir, 'models', 'nmf_model_files')
EMBEDDING_DIM = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 20
BATCH_SIZE = 4096
NUM_NEG_SAMPLES = 4
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

# --- 3. DATA SETUP ---
print("Loading and preparing data...")
train_data_path = os.path.join(root_dir, 'data', 'ml-32m', 'processed', 'train_20_core.csv')
validation_data_path = os.path.join(root_dir, 'data', 'ml-32m', 'processed', 'val_10_core.csv')

train_df_orig = pd.read_csv(train_data_path)
val_df = pd.read_csv(validation_data_path)

train_items = set(train_df_orig[train_df_orig['rating'] >= 4.0].movieId.unique())
train_users = set(train_df_orig[train_df_orig['rating'] >= 4.0].userId.unique())
train_df = train_df_orig[train_df_orig['movieId'].isin(train_items) & train_df_orig['userId'].isin(train_users)].copy()

uid2idx = {user_id: idx for idx, user_id in enumerate(train_df['userId'].unique())}
iid2idx = {item_id: idx for idx, item_id in enumerate(train_df['movieId'].unique())}
num_users = len(uid2idx)
num_items = len(iid2idx)

train_df['user_idx'] = train_df['userId'].map(uid2idx)
train_df['item_idx'] = train_df['movieId'].map(iid2idx)

# --- 4. DATA PREPARATION FOR NMF (VECTORIZED) ---
print("Creating pairwise training data (vectorized)...")
# Get all unique positive interactions
pos_df = train_df[['user_idx', 'item_idx']].drop_duplicates()
pos_df.rename(columns={'item_idx': 'pos_item_idx'}, inplace=True)

# 1. For each positive interaction, sample NUM_NEG_SAMPLES random items
# Repeat each row of pos_df NUM_NEG_SAMPLES times
num_interactions = len(pos_df)
triplets_df = pos_df.loc[pos_df.index.repeat(NUM_NEG_SAMPLES)].reset_index(drop=True)

# Generate a pool of random negative item candidates
random_negs = np.random.randint(0, num_items, size=len(triplets_df))
triplets_df['neg_item_idx'] = random_negs

# 2. Filter out "false negatives" where the randomly sampled item was actually positive
# We use a merge for an efficient anti-join
# Prepare the positive interactions for the merge check
positive_check_df = train_df[['user_idx', 'item_idx']].rename(columns={'item_idx': 'neg_item_idx'})
positive_check_df.drop_duplicates(inplace=True)

# Merge and find the rows in triplets_df that do NOT have a match in positive_check_df
merged = pd.merge(
    triplets_df,
    positive_check_df,
    on=['user_idx', 'neg_item_idx'],
    how='left',
    indicator=True
)
# Keep only the ones that are true negatives
triplets_df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

# Due to filtering, some positive pairs might have fewer than NUM_NEG_SAMPLES negatives.
# This is usually fine, but to keep it consistent, we can group and take the head.
triplets_df = triplets_df.groupby(['user_idx', 'pos_item_idx']).head(NUM_NEG_SAMPLES).reset_index(drop=True)

print(f"Vectorized training dataset created with {len(triplets_df)} triplets.")

train_ds = NMFDataset(
    user_indices=triplets_df['user_idx'].values,
    pos_item_indices=triplets_df['pos_item_idx'].values,
    neg_item_indices=triplets_df['neg_item_idx'].values
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)


# --- 5. MODEL TRAINING ---
os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)
save_model_path = os.path.join(MODEL_ARTIFACTS_DIR, f'nmf_model_{EMBEDDING_DIM}.pth')
model = NeuralMF(num_users, num_items, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# *** ADDING CYCLIC LR SCHEDULER ***
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

# Loss function for ranking
def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

print("\nStarting NMF model training...")
min_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for users, pos_items, neg_items in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
        
        optimizer.zero_grad()
        pos_scores, neg_scores = model(users, pos_items, neg_items)
        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Average BPR Loss: {avg_loss:.4f}")
    
    if avg_loss < min_loss:
        min_loss = avg_loss
        torch.save(model.state_dict(), save_model_path)
        print(f"New best model saved with loss {min_loss:.4f}")

# --- 6. EVALUATION ---
# load the best model
model.load_state_dict(torch.load(save_model_path))
model.to(device)


print("\nRunning NMF model evaluation...")
model.eval()

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
            scores = model.predict(user_tensor, all_items_tensor).cpu().numpy()
        
        item_score_df = pd.DataFrame({'item_idx': list(range(num_items)), 'score': scores})
        seen_items = train_seen_dict.get(user, set())
        item_score_df = item_score_df[~item_score_df['item_idx'].isin(seen_items)]
        user_recommendations = item_score_df.sort_values('score', ascending=False).head(k)
        
        user_truth = val_df[val_df['user_idx'] == user]
        if user_truth.empty: continue
        
        item_rel_dict = user_truth.set_index('item_idx')['rating'].to_dict()
        
        user_recommendations['relevance'] = user_recommendations['item_idx'].map(item_rel_dict).fillna(0)
        dcg = (user_recommendations['relevance'] / np.log2(np.arange(2, len(user_recommendations) + 2))).sum()
        
        ideal_relevances = user_truth['rating'].sort_values(ascending=False).head(k)
        idcg = (ideal_relevances.values / np.log2(np.arange(2, len(ideal_relevances) + 2))).sum()
        
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)

ndcg_score = ndcg_at_k(model, val_df, train_seen_dict, k=K_NDCG)
print("\n" + "="*40)
print(f"NeuralMF Model nDCG@{K_NDCG}: {ndcg_score:.4f}")
print("="*40)
