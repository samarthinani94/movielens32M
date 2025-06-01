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

from scripts.utils import root_dir
from scripts.modeling.models import MatrixFactorization, RatingDataset

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


# load uid2idx and iid2idx
with open(os.path.join(model_files_dir, 'uid2idx.pkl'), 'rb') as f:
    uid2idx = pickle.load(f)
with open(os.path.join(model_files_dir, 'iid2idx.pkl'), 'rb') as f:
    iid2idx = pickle.load(f)

df['user_idx'] = df['userId'].map(uid2idx)
df['item_idx'] = df['movieId'].map(iid2idx)

# create a dataset with all user-item pairs
unique_users = df['user_idx'].unique()
all_items = np.arange(len(iid2idx))
all_pairs = pd.DataFrame([(user, item) for user in unique_users for item in all_items], columns=['user_idx', 'item_idx'])

dataset = RatingDataset(
    user_indices=all_pairs['user_idx'].values,
    item_indices=all_pairs['item_idx'].values,
    ratings=np.zeros(len(all_pairs))  # Placeholder ratings, we will not use them
)

dl = DataLoader(dataset, batch_size=1024, shuffle=False)

# Load the trained model
embedding_dim = 10
model_file_name = f'mf_model_{embedding_dim}.pth'
model_save_path = os.path.join(save_model_path, model_file_name)
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

def ndcg_at_k(predictions, ground_truth, k):
    """
    Compute the NDCG at k for the given predictions and ground truth.
    
    :param predictions: DataFrame with columns ['user_idx', 'item_idx', 'pred_rating']
    :param ground_truth: DataFrame with columns ['user_idx', 'item_idx', 'rating']
    :param k: Number of top items to consider for NDCG calculation
    :return: NDCG score
    """
    ndcg_scores = []
    
    for user in predictions['user_idx'].unique():
        user_preds = predictions[predictions['user_idx'] == user].nlargest(k, 'pred_rating')
        user_gt = ground_truth[ground_truth['user_idx'] == user].nlargest(k, 'rating')
        
        if user_gt.empty:
            continue
        
        # treat gt ratings as relevance scores if available else 0
        

    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0

