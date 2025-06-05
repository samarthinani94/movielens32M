import os
import pickle
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR

import sys
sys.path.append("../../")

from scripts.utils import root_dir
from scripts.modeling.nmf_model import NeuralMF, RatingDataset

# PARAMETERS
train_data_path = os.path.join(root_dir, 'data', 'ml-32m', 'processed', 'train_20_core.csv')
validation_data_path = os.path.join(root_dir, 'data', 'ml-32m', 'processed', 'val_10_core.csv')
save_model_path = os.path.join(root_dir, 'models', 'nmf_model_files')
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found. Using GPU.")
elif torch.cuda.is_available():
    device = torch.device("cuda") # Fallback for NVIDIA GPUs if needed
    print("CUDA device found. Using GPU.")
else:
    device = torch.device("cpu")
    print("No GPU backend found. Using CPU.")
    
# -----------  MODEL DATA SETUP -----------
# Load the training data
train_df = pd.read_csv(train_data_path)
train_df = train_df[['userId', 'movieId', 'rating']]
print(f"Training data loaded with {len(train_df)} records.")

# include items with atleast one 4+ rating
train_items = set(train_df[train_df['rating'] >= 4.0].movieId.unique())
train_users = set(train_df[train_df['rating'] >= 4.0].userId.unique())
train_df = train_df[train_df['movieId'].isin(train_items)]
train_df = train_df[train_df['userId'].isin(train_users)]

train_df.reset_index(drop=True, inplace=True)

# save train items
train_items_path = os.path.join(save_model_path, 'train_items.pkl')
os.makedirs(save_model_path, exist_ok=True)
with open(train_items_path, 'wb') as f:
    pickle.dump(train_items, f)
    
train_users_path = os.path.join(save_model_path, 'train_users.pkl')
with open(train_users_path, 'wb') as f:
    pickle.dump(train_users, f)

# load uid2idx and iid2idx if they exist else create them
if os.path.exists(os.path.join(save_model_path, 'uid2idx.pkl')) and os.path.exists(os.path.join(save_model_path, 'iid2idx.pkl')):
    with open(os.path.join(save_model_path, 'uid2idx.pkl'), 'rb') as f:
        uid2idx = pickle.load(f)
    with open(os.path.join(save_model_path, 'iid2idx.pkl'), 'rb') as f:
        iid2idx = pickle.load(f)
    print("Loaded existing uid2idx and iid2idx mappings.")
else:
    # If not, create them from the training data
    uid2idx = {user_id: idx for idx, user_id in enumerate(train_df['userId'].unique())}
    iid2idx = {item_id: idx for idx, item_id in enumerate(train_df['movieId'].unique())}
    # save the mappings in save_model_path
    os.makedirs(save_model_path, exist_ok=True)

    with open(os.path.join(save_model_path, 'uid2idx.pkl'), 'wb') as f:
        pickle.dump(uid2idx, f)

    with open(os.path.join(save_model_path, 'iid2idx.pkl'), 'wb') as f:
        pickle.dump(iid2idx, f)
    print("Created new uid2idx and iid2idx mappings and saved them.")
    
# Convert userId and movieId to indices
train_df['user_idx'] = train_df['userId'].map(uid2idx)
train_df['item_idx'] = train_df['movieId'].map(iid2idx)   
    
# 4+ ratings to be considered as positive feedback
train_df_pos = train_df[train_df['rating'] >= 4.0].copy()

# existing user ratings; not to be used for negative sampling
uidx2idxs = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
uidx2posidxs = train_df_pos.groupby('user_idx')['item_idx'].apply(set).to_dict()

# save uidx2posidxs
uidx2posidxs_path = os.path.join(save_model_path, 'uidx2posidxs.pkl')
with open(uidx2posidxs_path, 'wb') as f:
    pickle.dump(uidx2posidxs, f)

all_items = set(iid2idx.values())
train_data = []

for user_idx in tqdm(uidx2idxs):
    usr_items = uidx2idxs[user_idx]
    pos_items = np.array(list(uidx2posidxs[user_idx]))
    neg_pool = np.array(list(all_items - usr_items))
    
    if len(neg_pool) == 0 or len(pos_items) == 0:
        continue  # safety check
    
    # For each positive item, sample K negatives (vectorized)
    k = min(10, len(neg_pool))
    # For all positive items, sample k negatives each
    neg_samples = np.random.choice(neg_pool, size=(len(pos_items), k), replace=True)
    
    # Build the training triples in a vectorized way
    user_col = np.full((len(pos_items), k), user_idx, dtype=int)
    pos_col = np.repeat(pos_items[:, np.newaxis], k, axis=1)
    triples = np.stack([user_col, pos_col, neg_samples], axis=-1).reshape(-1, 3)
    train_data.append(triples)

# Concatenate all user triples and create DataFrame
train_data = np.concatenate(train_data, axis=0)
train_data = pd.DataFrame(train_data, columns=["user_idx", "pos_item_idx", "neg_item_idx"])
train_data = train_data.drop_duplicates().reset_index(drop=True)

# save train_data as pickle
train_data_path = os.path.join(save_model_path, 'train_data.pkl')
with open(train_data_path, 'wb') as f:
    pickle.dump(train_data, f)
    
    
# Create the dataset and dataloader
batch_size = 1024
train_ds = RatingDataset(
    user_indices=train_data['user_idx'].values,
    pos_item_indices=train_data['pos_item_idx'].values,
    neg_item_indices=train_data['neg_item_idx'].values
)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

print(f"Training dataset created with {len(train_ds)} records.")
print(f"Train dataloader initialized with batch size {batch_size}.")

# model parameters
embedding_dim = 100
model_file_name = f'nmf_model_{embedding_dim}.pth'
model_save_path = os.path.join(save_model_path, model_file_name)

num_users = len(uid2idx)
num_items = len(iid2idx)

if os.path.exists(model_save_path):
    model = NeuralMF(num_users, num_items, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    print(f"Loaded existing model from {model_save_path}")
else:
    model = NeuralMF(num_users, num_items, embedding_dim=embedding_dim)
    model.to(device)
    print(f"Initialized new model with embedding dimension {embedding_dim}")

# Optimizer and loss function
def bpr_loss(pos_score, neg_score):
    return -torch.log(torch.sigmoid(pos_score - neg_score)).mean()

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

iterations_per_epoch = len(train_loader)
step_size_val = 4 * iterations_per_epoch

scheduler = CyclicLR(
    optimizer, 
    base_lr=5e-4,      # Minimum LR
    max_lr=5e-3,       # Peak LR
    step_size_up=step_size_val,  # Number of training steps to increase LR
    mode='triangular',  
    cycle_momentum=False 
)

#-------------- MODEL TRAINING --------------

def train_model(model, model_save_path, train_loader, optimizer, scheduler, num_epochs=10):
    min_train_loss = float('inf')  # Move this outside the loop

    for epoch in tqdm(range(num_epochs)):
        start_time = time.time()
        model.train()
        train_loss = 0

        for user_indices, pos_item_indices, neg_item_indices in train_loader:
            user_indices = user_indices.to(device)
            pos_item_indices = pos_item_indices.to(device)
            neg_item_indices = neg_item_indices.to(device)

            optimizer.zero_grad()
            pos_score, neg_score = model(user_indices, pos_item_indices, neg_item_indices)
            loss = bpr_loss(pos_score, neg_score)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()     

        current_lr = scheduler.get_last_lr()[0]
        end_time = time.time()

        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Time: {end_time - start_time:.2f}s")

        # Save only if it's the best so far
        if avg_loss < min_train_loss:
            min_train_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path} with loss {min_train_loss:.4f}")
        else:
            print(f"No improvement in loss. Current loss: {avg_loss:.4f}, Min loss: {min_train_loss:.4f}")
        
    return model


print("Starting model training...")
train_model(model, model_save_path, train_loader, optimizer, scheduler, num_epochs=20)

