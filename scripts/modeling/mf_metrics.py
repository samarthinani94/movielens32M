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
from scripts.modeling.mf_model import MatrixFactorization, RatingDataset



# PARAMETERS
model_files_dir = os.path.join(root_dir, 'models', 'mf_model_files')
val_data_dir = os.path.join(root_dir, 'data', 'ml-32m', 'processed')
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found. Using GPU.")
elif torch.cuda.is_available():
    device = torch.device("cuda") # Fallback for NVIDIA GPUs if needed
    print("CUDA device found. Using GPU.")
else:
    device = torch.device("cpu")
    print("No GPU backend found. Using CPU.")

# load validation data
val_filename = 'val_10_core.csv'
val_data_path = os.path.join(val_data_dir, val_filename)
val_df = pd.read_csv(val_data_path)

# load train data
train_filename = 'train_20_core.csv'
train_data_path = os.path.join(val_data_dir, train_filename)
train_df = pd.read_csv(train_data_path)


# load uid2idx and iid2idx
with open(os.path.join(model_files_dir, 'uid2idx.pkl'), 'rb') as f:
    uid2idx = pickle.load(f)
with open(os.path.join(model_files_dir, 'iid2idx.pkl'), 'rb') as f:
    iid2idx = pickle.load(f)

val_df['user_idx'] = val_df['userId'].map(uid2idx)
val_df['item_idx'] = val_df['movieId'].map(iid2idx)

train_df['user_idx'] = train_df['userId'].map(uid2idx)
train_df['item_idx'] = train_df['movieId'].map(iid2idx)

# create a dataset with all user-item pairs
unique_users = val_df['user_idx'].unique()
all_items = np.arange(len(iid2idx))
all_pairs = pd.DataFrame([(user, item) for user in unique_users for item in all_items], columns=['user_idx', 'item_idx'])

dataset = RatingDataset(
    user_indices=all_pairs['user_idx'].values,
    item_indices=all_pairs['item_idx'].values,
    ratings=np.zeros(len(all_pairs))  # Placeholder ratings, we will not use them
)

dl = DataLoader(dataset, batch_size=1024, shuffle=False)

# Load the trained model
embedding_dim = 100
model_file_name = f'mf_model_{embedding_dim}.pth'
model_save_path = os.path.join(model_files_dir, model_file_name)
model = MatrixFactorization(num_users=len(uid2idx), num_items=len(iid2idx), embedding_dim=embedding_dim)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()

# generate predictions and save them in a dataframe

predictions = []
for user_indices, item_indices, _ in tqdm(dl, desc="Generating predictions"):
    user_indices = user_indices.to(device)
    item_indices = item_indices.to(device)
    
    with torch.no_grad():
        preds = model(user_indices, item_indices).cpu().numpy()
    
    predictions.extend(zip(user_indices.cpu().numpy(), item_indices.cpu().numpy(), preds))

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['user_idx', 'item_idx', 'pred_rating'])

def ndcg_at_k(predictions, ground_truth, train_df, k):
    """
    Compute the NDCG at k for the given predictions and ground truth.
    
    :param predictions: DataFrame with columns ['user_idx', 'item_idx', 'pred_rating']
    :param ground_truth: DataFrame with columns ['user_idx', 'item_idx', 'rating']
    :param k: Number of top items to consider for NDCG calculation
    :return: NDCG score
    """
    ndcg_scores = []
    
    for user_idx in predictions['user_idx'].unique():
        user_preds = predictions[predictions['user_idx'] == user_idx]
        
        # remove items from train_df
        train_item_idxs = set(train_df[train_df['user_idx'] == user_idx]['item_idx'].values)
        user_preds = user_preds[~user_preds['item_idx'].isin(train_item_idxs)].reset_index(drop=True)
        user_truth = ground_truth[ground_truth['user_idx'] == user_idx]
        item_rel_dict = user_truth.set_index('item_idx')['rating'].to_dict() 
        
        if user_preds.empty or user_truth.empty:
            continue
        
        # calculate DCG
        user_preds = user_preds.sort_values(by='pred_rating', ascending=False).head(k)
        user_preds['rel'] = user_preds['item_idx'].map(item_rel_dict).fillna(0)
        user_preds['rank'] = np.arange(1, len(user_preds) + 1)
        dcg = (user_preds['rel'] / np.log2(user_preds['rank'] + 1)).sum()
        
        # calculate IDCG
        ideal_relevances = user_truth['rating'].sort_values(ascending=False).head(k)
        idcg = (ideal_relevances / np.log2(np.arange(1, len(ideal_relevances) + 1) + 1)).sum()
        
        ndcg = dcg / idcg if idcg > 0 else 0
    
        ndcg_scores.append(ndcg)
        
    return np.mean(ndcg_scores)

# Calculate NDCG@10
ndcg_score = ndcg_at_k(predictions_df, val_df, train_df, k=10)
