import os
import pickle
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR

sys.path.append("../../")
from scripts.utils import root_dir
# Import all necessary model classes
from scripts.modeling.two_tower_retriever_model import TwoTowerSystem
from scripts.modeling.reranker_model import ReRankerNN, ReRankerDataset

# CONFIGURATION & SETUP
RETRIEVER_MODEL_DIR = os.path.join(root_dir, 'models', 'two_tower_retriever_files')
RERANKER_ARTIFACTS_DIR = os.path.join(root_dir, 'models', 'reranker_files')
os.makedirs(RERANKER_ARTIFACTS_DIR, exist_ok=True)

# Re-ranker specific hyperparameters
RERANKER_CONFIG = {
    "reranker_embedding_dim": 128,
    "reranker_year_embedding_dim": 32,
    "reranker_hidden_dim_mlp": 256,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "epochs": 20,
    "batch_size": 2048
}
# Evaluation parameters
CANDIDATE_K = 500  # The K used to generate candidates
NDCG_K = 10        # The K for the final nDCG metric

# Setup device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found. Using GPU.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device found. Using GPU.")
else:
    device = torch.device("cpu")
    print("No GPU backend found. Using CPU.")


# LOAD ARTIFACTS
print("--- Loading All Necessary Artifacts ---")
# Load retriever config and artifacts
with open(os.path.join(RETRIEVER_MODEL_DIR, 'model_config.pkl'), 'rb') as f:
    retriever_config = pickle.load(f)
with open(os.path.join(RETRIEVER_MODEL_DIR, 'movies_df.pkl'), 'rb') as f:
    movies_df = pickle.load(f)
with open(os.path.join(RETRIEVER_MODEL_DIR, 'val_data.pkl'), 'rb') as f:
    val_data = pickle.load(f)
# Load the training data generated in the previous step
with open(os.path.join(RERANKER_ARTIFACTS_DIR, 'reranker_training_data.pkl'), 'rb') as f:
    reranker_train_df = pickle.load(f)
    print(f"Loaded {len(reranker_train_df)} training samples for the re-ranker.")

# Combine retriever and re-ranker configs
# This gives the ReRankerNN access to vocab sizes from the retriever
RERANKER_CONFIG.update(retriever_config)

# MODEL INITIALIZATION
print("--- Initializing Models ---")
# Load the trained retriever model to get its frozen movie embedding layer
retriever_model = TwoTowerSystem(retriever_config).to(device)
retriever_model.load_state_dict(torch.load(os.path.join(RETRIEVER_MODEL_DIR, 'two_tower_retriever.pth'), map_location=device))
retriever_model.eval()
print("Loaded trained retriever model.")

# Instantiate the Re-Ranker model
reranker_model = ReRankerNN(RERANKER_CONFIG, retriever_model.shared_movie_embedding).to(device)
print("Instantiated new re-ranker model.")


# RE-RANKER TRAINING
print("\n--- Starting Re-Ranker Training ---")
train_dataset = ReRankerDataset(reranker_train_df)
train_loader = DataLoader(train_dataset, batch_size=RERANKER_CONFIG['batch_size'], shuffle=True)

optimizer = optim.Adam(reranker_model.parameters(), lr=RERANKER_CONFIG['learning_rate'], weight_decay=RERANKER_CONFIG['weight_decay'])
scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=RERANKER_CONFIG['learning_rate'], step_size_up=len(train_loader)*4, mode='triangular')

def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

for epoch in range(RERANKER_CONFIG['epochs']):
    reranker_model.train()
    total_loss = 0
    for user_inputs, pos_item_inputs, neg_item_inputs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{RERANKER_CONFIG['epochs']}"):
        user_inputs = {k: v.to(device) for k, v in user_inputs.items()}
        pos_item_inputs = {k: v.to(device) for k, v in pos_item_inputs.items()}
        neg_item_inputs = {k: v.to(device) for k, v in neg_item_inputs.items()}
        
        optimizer.zero_grad()
        pos_scores, neg_scores = reranker_model(user_inputs, pos_item_inputs, neg_item_inputs)
        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{RERANKER_CONFIG['epochs']}, Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# Save the trained re-ranker model
reranker_model_path = os.path.join(RERANKER_ARTIFACTS_DIR, 'reranker_model.pth')
torch.save(reranker_model.state_dict(), reranker_model_path)
print(f"Re-ranker model saved to {reranker_model_path}")


# FINAL TWO-STAGE EVALUATION
print("\n--- Running Final Two-Stage Evaluation (Retriever + Re-ranker) ---")
reranker_model.eval()

# Prepare data for evaluation
val_users = val_data[['user_idx', 'last_5_items_idx', 'timestamp']].copy()
val_users['last_5_items_idx'] = val_users['last_5_items_idx'].apply(lambda x: tuple(x))
val_users.sort_values(by='timestamp', inplace=True)
val_users = val_users.drop_duplicates(subset=['user_idx'], keep='first').reset_index(drop=True)
val_users['last_5_items_idx'] = val_users['last_5_items_idx'].apply(lambda x: list(x))

ground_truth = val_data[val_data['rating'] >= 4.0].groupby('user_idx')['item_idx'].apply(set).to_dict()
movies_features_df = movies_df.set_index('item_idx')

# Pre-compute all item embeddings for the retriever
with torch.no_grad():
    item_ids = torch.tensor(movies_df['item_idx'].values, dtype=torch.long).to(device)
    genre_indices = torch.tensor(np.stack(movies_df['genres_idx'].values), dtype=torch.long).to(device)
    year_indices = torch.tensor(movies_df['year_idx'].values, dtype=torch.long).to(device)
    tag_indices = torch.tensor(np.stack(movies_df['tags_idx'].values), dtype=torch.long).to(device)
    all_item_embeddings = retriever_model.movie_tower(item_ids, genre_indices, year_indices, tag_indices)

ndcg_scores = []
for i in tqdm(range(0, len(val_users), 512), desc="Evaluating Two-Stage System"):
    batch_users_df = val_users.iloc[i:i+512]
    user_ids = torch.tensor(batch_users_df['user_idx'].values, dtype=torch.long).to(device)
    history_ids = torch.tensor(np.stack(batch_users_df['last_5_items_idx'].values), dtype=torch.long).to(device)

    # Stage 1: Get candidates from the Retriever
    with torch.no_grad():
        user_embeddings = retriever_model.user_tower(user_ids, history_ids)
        scores = torch.matmul(user_embeddings, all_item_embeddings.T)
        _, candidate_indices = torch.topk(scores, k=CANDIDATE_K, dim=1)
    candidate_indices = candidate_indices.cpu().numpy()

    # Stage 2: Re-rank the candidates
    for j, user_idx in enumerate(batch_users_df['user_idx']):
        candidates = candidate_indices[j]
        user_input_reranker = {'user_id': torch.tensor([user_idx] * len(candidates), dtype=torch.long).to(device)}
        
        candidate_features = movies_features_df.loc[candidates]
        item_inputs_reranker = {
            'movie_id': torch.tensor(candidate_features.index.values, dtype=torch.long).to(device),
            'padded_genre_indices': torch.tensor(np.stack(candidate_features['genres_idx'].values), dtype=torch.long).to(device),
            'year_idx': torch.tensor(candidate_features['year_idx'].values, dtype=torch.long).to(device),
            'padded_tag_indices': torch.tensor(np.stack(candidate_features['tags_idx'].values), dtype=torch.long).to(device)
        }
        
        with torch.no_grad():
            reranked_scores = reranker_model.predict(user_input_reranker, item_inputs_reranker).cpu().numpy()

        # Calculate nDCG for this user
        final_ranking = pd.DataFrame({'item_idx': candidates, 'score': reranked_scores}).sort_values('score', ascending=False).head(NDCG_K)
        
        true_items = ground_truth.get(user_idx, set())
        if not true_items: continue
        
        # Calculate DCG
        final_ranking['relevance'] = final_ranking['item_idx'].apply(lambda x: 1 if x in true_items else 0)
        dcg = (final_ranking['relevance'] / np.log2(np.arange(2, len(final_ranking) + 2))).sum()

        # Calculate IDCG
        idcg = (1 / np.log2(np.arange(2, min(len(true_items), NDCG_K) + 2))).sum()
        
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

final_ndcg = np.mean(ndcg_scores)
print("\n" + "="*50)
print("      Final Two-Stage Recommendation Results")
print("="*50)
print(f"nDCG@{NDCG_K}: {final_ndcg:.4f}")
print("="*50)
