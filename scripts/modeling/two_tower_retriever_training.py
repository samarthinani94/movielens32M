import os
import pickle
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR

sys.path.append("../../")
from scripts.utils import root_dir
# Import from the new, clean model file
from scripts.modeling.two_tower_retriever_model import TwoTowerSystem, TwoTowerDataset

# --- 1. CONFIGURATION & SETUP ---
# File Paths
DATA_DIR = os.path.join(root_dir, 'data', 'ml-32m')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_SAVE_DIR = os.path.join(root_dir, 'models', 'two_tower_retriever_files')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Model & Training Hyperparameters
CONFIG = {
    "embedding_dim": 128,
    "year_embedding_dim": 32,
    "hidden_dim_lstm": 64,
    "hidden_dim_mlp_user": 256,
    "hidden_dim_mlp_movie": 256,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "epochs": 12, # Adjusted since it converged
    "batch_size": 4096,
    "num_neg_samples": 4,
    "history_length": 5,
    "genre_list_length": 10,
    "tag_list_length": 3,
    "recall_k_values": [50, 100, 200, 500, 1000] # Added larger K values
}


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


# --- EVALUATION FUNCTION ---
def calculate_recall_at_k(model, val_data, movies_df, config, device):
    """Calculates recall@k for the retriever model on the validation set."""
    model.eval()
    
    # Prepare validation user data and ground truth
    val_users = val_data[['user_idx', 'last_5_items_idx']].copy()
    val_users['last_5_items_idx'] = val_users['last_5_items_idx'].apply(lambda x: tuple(x))
    val_users = val_users.drop_duplicates().reset_index(drop=True)
    val_users['last_5_items_idx'] = val_users['last_5_items_idx'].apply(lambda x: list(x))
    val_ground_truth = val_data.groupby('user_idx')['item_idx'].apply(set).to_dict()
    
    all_recalls = {k: [] for k in config['recall_k_values']}
    max_k = max(config['recall_k_values'])

    with torch.no_grad():
        # Pre-compute all item embeddings once
        item_ids = torch.tensor(movies_df['item_idx'].values, dtype=torch.long).to(device)
        genre_indices = torch.tensor(np.stack(movies_df['genres_idx'].values), dtype=torch.long).to(device)
        year_indices = torch.tensor(movies_df['year_idx'].values, dtype=torch.long).to(device)
        tag_indices = torch.tensor(np.stack(movies_df['tags_idx'].values), dtype=torch.long).to(device)
        all_item_embeddings = model.movie_tower(item_ids, genre_indices, year_indices, tag_indices)

        for i in tqdm(range(0, len(val_users), config['batch_size']), desc="Calculating Recall"):
            batch_df = val_users.iloc[i:i+config['batch_size']]
            user_ids = torch.tensor(batch_df['user_idx'].values, dtype=torch.long).to(device)
            history_ids = torch.tensor(np.stack(batch_df['last_5_items_idx'].values), dtype=torch.long).to(device)

            user_embeddings = model.user_tower(user_ids, history_ids)
            scores = torch.matmul(user_embeddings, all_item_embeddings.T)
            
            _, top_k_indices = torch.topk(scores, k=max_k, dim=1)
            top_k_indices = top_k_indices.cpu().numpy()

            for j, user_idx in enumerate(batch_df['user_idx']):
                ground_truth_items = val_ground_truth.get(user_idx, set())
                if not ground_truth_items:
                    continue

                for k in config['recall_k_values']:
                    top_k_recs = set(top_k_indices[j, :k])
                    hits = len(ground_truth_items.intersection(top_k_recs))
                    recall = hits / len(ground_truth_items)
                    all_recalls[k].append(recall)

    return {f"Recall@{k}": np.mean(recalls) for k, recalls in all_recalls.items()}


# --- 2. DATA LOADING & INITIAL FILTERING ---
print("--- Starting Data Processing ---")
train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_20_core.csv'))
val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'val_10_core.csv'))
movies_df_orig = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
tags_df_orig = pd.read_csv(os.path.join(DATA_DIR, "tags.csv"))

# Filter for items with at least one 4+ rating and users who rated them
print("Filtering users and items based on 4+ ratings...")
train_items_set = set(train_df[train_df['rating'] >= 4.0].movieId.unique())
train_users_set = set(train_df[train_df['rating'] >= 4.0].userId.unique())

train_df = train_df[train_df['movieId'].isin(train_items_set) & train_df['userId'].isin(train_users_set)].copy()

# --- 3. METADATA PREPROCESSING & MAPPINGS ---
print("Processing metadata (genres, years, tags)...")
# Filter metadata to only include items present in the training set
movies_df = movies_df_orig[movies_df_orig['movieId'].isin(train_items_set)].copy()
tags_df = tags_df_orig[tags_df_orig['movieId'].isin(train_items_set)].copy()

# Create main ID to index mappings
uid2idx = {uid: i for i, uid in enumerate(train_df['userId'].unique())}
iid2idx = {iid: i for i, iid in enumerate(train_df['movieId'].unique())}
CONFIG['num_users'] = len(uid2idx)
CONFIG['num_movies'] = len(iid2idx)

# Process Genres
movies_df['genres_list'] = movies_df['genres'].apply(lambda x: x.split('|'))
all_genres = sorted(list(set(g for genres in movies_df['genres_list'] for g in genres)))
genre2idx = {genre: i+1 for i, genre in enumerate(all_genres)} # 0 for padding
genre2idx['<PAD>'] = 0
CONFIG['num_genres'] = len(genre2idx)

# Process Years
def get_year(title):
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else -1
movies_df['year'] = movies_df['title'].apply(get_year)
all_years = sorted(movies_df['year'].unique())
year2idx = {year: i+1 for i, year in enumerate(all_years)}
year2idx['<PAD>'] = 0
CONFIG['num_years'] = len(year2idx)

# Process Tags
tags_grouped = tags_df.groupby('movieId')['tag'].apply(lambda x: x.value_counts().nlargest(CONFIG['tag_list_length']).index.tolist()).reset_index()
movies_df = movies_df.merge(tags_grouped, on='movieId', how='left')
movies_df['tag'].fillna('', inplace=True)
all_tags = sorted(list(set(t for tags in movies_df['tag'] for t in tags if t)))
tag2idx = {tag: i+1 for i, tag in enumerate(all_tags)}
tag2idx['<PAD>'] = 0
CONFIG['num_movie_tags_vocab'] = len(tag2idx)

# Apply mappings and padding to movies_df
def pad_list(lst, length, pad_value):
    return (lst + [pad_value] * length)[:length]

movies_df['genres_idx'] = movies_df['genres_list'].apply(lambda g_list: pad_list([genre2idx.get(g, 0) for g in g_list], CONFIG['genre_list_length'], 0))
movies_df['year_idx'] = movies_df['year'].apply(lambda y: year2idx.get(y, 0))
movies_df['tags_idx'] = movies_df['tag'].apply(lambda t_list: pad_list([tag2idx.get(t, 0) for t in t_list], CONFIG['tag_list_length'], 0))
movies_df['item_idx'] = movies_df['movieId'].map(iid2idx)
movies_df = movies_df[['item_idx', 'genres_idx', 'year_idx', 'tags_idx']].sort_values('item_idx').reset_index(drop=True).copy()


# --- 4. FEATURE ENGINEERING & TRAINING DATA CREATION ---
print("Creating historical features and negative samples...")
# Combine train and val to create user history features correctly
val_df['user_idx'] = val_df['userId'].map(uid2idx)
val_df['item_idx'] = val_df['movieId'].map(iid2idx)
val_df.dropna(subset=['user_idx', 'item_idx'], inplace=True)
train_df['user_idx'] = train_df['userId'].map(uid2idx)
train_df['item_idx'] = train_df['movieId'].map(iid2idx)
combined_df = pd.concat([train_df, val_df]).sort_values(by=['user_idx', 'timestamp'])

# Create "last_n_items"
padding_idx = CONFIG['num_movies']
def rolling_list_exclude_current(s, window=CONFIG['history_length']):
    # Shift to exclude current row, then build rolling lists
    s_shifted = s.shift(1)
    return [list(s_shifted[max(0, i - window + 1):i + 1].dropna()) for i in range(len(s))]

combined_df['last_5_items_idx'] = (
    combined_df
    .groupby('user_idx')['item_idx']
    .transform(lambda x: rolling_list_exclude_current(x, window=CONFIG['history_length']))
)

def pad_last_5_items(items, pad_value, length=5):
    if len(items) < length:
        items = [pad_value] * (length - len(items)) + [int(e) for e in items]
    return items[-length:]  # Ensure we only keep the last 'length' items

combined_df['last_5_items_idx'] = combined_df['last_5_items_idx'].apply(
    lambda x: pad_last_5_items(x, pad_value=padding_idx, length=CONFIG['history_length'])
)

# Separate back into train/val
final_train_df = combined_df[combined_df['timestamp'] <= train_df['timestamp'].max()].copy()
final_val_df = combined_df[combined_df['timestamp'] > train_df['timestamp'].max()].copy()

# Vectorized Negative Sampling
pos_df = final_train_df[final_train_df['rating'] >= 4.0][['user_idx', 'item_idx']].drop_duplicates()
pos_df.rename(columns={'item_idx': 'pos_item_idx'}, inplace=True)

num_pos_interactions = len(pos_df)
triplets_df = pos_df.loc[pos_df.index.repeat(CONFIG['num_neg_samples'])].reset_index(drop=True)
triplets_df['neg_item_idx'] = np.random.randint(0, CONFIG['num_movies'], size=len(triplets_df))

positive_check_df = final_train_df[['user_idx', 'item_idx']].rename(columns={'item_idx': 'neg_item_idx'}).drop_duplicates()
merged = pd.merge(triplets_df, positive_check_df, on=['user_idx', 'neg_item_idx'], how='left', indicator=True)
triplets_df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
triplets_df = triplets_df.groupby(['user_idx', 'pos_item_idx']).head(1).reset_index(drop=True)

# Join all features to create the final training data
history_map_df = final_train_df[['user_idx', 'item_idx', 'last_5_items_idx']].drop_duplicates(subset=['user_idx', 'item_idx'])
train_data = pd.merge(triplets_df, history_map_df, left_on=['user_idx', 'pos_item_idx'], right_on=['user_idx', 'item_idx']).drop('item_idx', axis=1)
train_data = pd.merge(train_data, movies_df, left_on='pos_item_idx', right_on='item_idx').drop('item_idx', axis=1).rename(columns={'genres_idx': 'pos_item_genres_idx', 'year_idx': 'pos_item_year_idx', 'tags_idx': 'pos_item_tags_idx'})
train_data = pd.merge(train_data, movies_df, left_on='neg_item_idx', right_on='item_idx').drop('item_idx', axis=1).rename(columns={'genres_idx': 'neg_item_genres_idx', 'year_idx': 'neg_item_year_idx', 'tags_idx': 'neg_item_tags_idx'})
train_data['user_idx'] = train_data['user_idx'].astype(np.int64)
train_data['pos_item_idx'] = train_data['pos_item_idx'].astype(np.int64)

# convert user_idx, item_idx and last_5_items_idx to int64 for final_val_df
final_val_df['user_idx'] = final_val_df['user_idx'].astype(np.int64)
final_val_df['item_idx'] = final_val_df['item_idx'].astype(np.int64)
final_val_df['last_5_items_idx'] = final_val_df['last_5_items_idx'].apply(lambda x: [int(i) for i in x])

print(f"Final training dataset created with {len(train_data)} triplets.")
train_dataset = TwoTowerDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)

# --- 5. MODEL TRAINING ---
print("\n--- Starting Model Training ---")
model = TwoTowerSystem(CONFIG).to(device)
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=CONFIG['learning_rate'], step_size_up=len(train_loader)*4, mode='triangular')

def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

for epoch in range(CONFIG['epochs']):
    model.train()
    total_loss = 0
    for user_inputs, pos_movie_inputs, neg_movie_inputs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
        user_inputs = {k: v.to(device) for k, v in user_inputs.items()}
        pos_movie_inputs = {k: v.to(device) for k, v in pos_movie_inputs.items()}
        neg_movie_inputs = {k: v.to(device) for k, v in neg_movie_inputs.items()}
        
        optimizer.zero_grad()
        pos_scores, neg_scores = model(user_inputs, pos_movie_inputs, neg_movie_inputs)
        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    
    # --- EVALUATION STEP ---
    recall_scores = calculate_recall_at_k(model, final_val_df, movies_df, CONFIG, device)
    recall_str = " | ".join([f"{key}: {val:.4f}" for key, val in recall_scores.items()])
    
    print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Avg Loss: {avg_loss:.4f}, {recall_str}, LR: {scheduler.get_last_lr()[0]:.6f}")

# --- 6. SAVE ARTIFACTS ---
print("\n--- Saving Artifacts ---")
torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'two_tower_retriever.pth'))
with open(os.path.join(MODEL_SAVE_DIR, 'model_config.pkl'), 'wb') as f: pickle.dump(CONFIG, f)
with open(os.path.join(MODEL_SAVE_DIR, 'movies_df.pkl'), 'wb') as f: pickle.dump(movies_df, f)
with open(os.path.join(MODEL_SAVE_DIR, 'val_data.pkl'), 'wb') as f: pickle.dump(final_val_df, f)
with open(os.path.join(MODEL_SAVE_DIR, 'mappings.pkl'), 'wb') as f:
    pickle.dump({'uid2idx': uid2idx, 'iid2idx': iid2idx, 'genre2idx': genre2idx, 'year2idx': year2idx, 'tag2idx': tag2idx}, f)

print("Training pipeline complete. Artifacts saved.")


print("\n--- Running Final Evaluation on Trained Model ---")
# Load the trained model for evaluation
model_path = os.path.join(MODEL_SAVE_DIR, 'two_tower_retriever.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

recall_scores = calculate_recall_at_k(model, final_val_df, movies_df, CONFIG, device)

print("\n" + "="*50)
print("          Final Retriever Recall Results")
print("="*50)
for key, val in recall_scores.items():
    print(f"{key:<15}: {val:.4f}")
print("="*50)