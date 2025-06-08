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
from scripts.modeling.two_tower_model import TwoTowerSystem, UserTower, MovieTower, TwoTowerTrainingDataset

# PARAMETERS
train_data_path = os.path.join(root_dir, 'data', 'ml-32m', 'processed', 'train_20_core.csv')
validation_data_path = os.path.join(root_dir, 'data', 'ml-32m', 'processed', 'val_10_core.csv')
helper_files_path = os.path.join(root_dir, 'models', 'nmf_model_files')
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
train_df = train_df[['userId', 'movieId', 'rating', 'timestamp']]
print(f"Training data loaded with {len(train_df)} records.")

# include items with atleast one 4+ rating
train_items = set(train_df[train_df['rating'] >= 4.0].movieId.unique())
train_users = set(train_df[train_df['rating'] >= 4.0].userId.unique())
train_df = train_df[train_df['movieId'].isin(train_items)]
train_df = train_df[train_df['userId'].isin(train_users)]
train_df.reset_index(drop=True, inplace=True)
print(f"Filtered training data to {len(train_df)} records with 4+ ratings.")

with open(os.path.join(helper_files_path, 'uid2idx.pkl'), 'rb') as f:
        uid2idx = pickle.load(f)
with open(os.path.join(helper_files_path, 'iid2idx.pkl'), 'rb') as f:
    iid2idx = pickle.load(f)
print("Loaded existing uid2idx and iid2idx mappings.")

# check if train_items and iid2idx have the same items
if set(train_items) != set(iid2idx.keys()):
    print("Warning: train_items and iid2idx do not match. This may lead to issues during training.")

# check if train_users and uid2idx have the same users
if set(train_users) != set(uid2idx.keys()):
    print("Warning: train_users and uid2idx do not match. This may lead to issues during training.")

# Convert userId and movieId to indices  
train_df['user_idx'] = train_df['userId'].map(uid2idx)
train_df['item_idx'] = train_df['movieId'].map(iid2idx)   


# load validation data
val_df = pd.read_csv(validation_data_path)
# Convert userId and movieId to indices
val_df['user_idx'] = val_df['userId'].map(uid2idx)
val_df['item_idx'] = val_df['movieId'].map(iid2idx)
val_df = val_df.loc[~val_df['user_idx'].isna()]
val_df = val_df.loc[~val_df['item_idx'].isna()]
val_df.reset_index(drop=True, inplace=True)
val_df['user_idx'] = val_df['user_idx'].astype(int)
val_df['item_idx'] = val_df['item_idx'].astype(int)

train_df['is_val'] = 0
val_df['is_val'] = 1

# Combine train and validation data
combined_df = pd.concat([train_df, val_df], ignore_index=True)

# for a given user, get the last 5 items just before the current timestamp
combined_df.sort_values(by=['user_idx', 'timestamp'], inplace=True)
def rolling_list_exclude_current(s, window=5):
    # Shift to exclude current row, then build rolling lists
    s_shifted = s.shift(1)
    return [list(s_shifted[max(0, i - window + 1):i + 1].dropna()) for i in range(len(s))]

combined_df['last_5_items'] = (
    combined_df
    .groupby('user_idx')['item_idx']
    .transform(lambda x: rolling_list_exclude_current(x, window=5))
)

#  if fewer than 5 items pre pad with value=len(iid2idx) to achieve a fixed length of 5
# example: [1, 2, 3] -> [len(iid2idx),len(iid2idx),1,2,3]

def pad_last_5_items(items, pad_value, length=5):
    if len(items) < length:
        items = [pad_value] * (length - len(items)) + [int(e) for e in items]
    return items[-length:]  # Ensure we only keep the last 'length' items

combined_df['last_5_items'] = combined_df['last_5_items'].apply(
    lambda x: pad_last_5_items(x, pad_value=len(iid2idx), length=5)
)

# load movies_df (pickle file) from helper_files_path
movies_df_path = os.path.join(helper_files_path, 'movies_df.pkl')
if os.path.exists(movies_df_path):
    movies_df = pd.read_pickle(movies_df_path)

combined_df = combined_df.merge(movies_df, on='item_idx', how='inner')
combined_df.drop(columns=['movieId', 'userId'], inplace=True)

train_df = combined_df[combined_df['is_val'] == 0].drop(columns=['is_val'])
val_df = combined_df[combined_df['is_val'] == 1].drop(columns=['is_val'])

# 4+ ratings to be considered as positive feedback
train_df_pos = train_df[train_df['rating'] >= 4.0].copy()

# existing user ratings; not to be used for negative sampling
uidx2idxs = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
uidx2posidxs = train_df_pos.groupby('user_idx')['item_idx'].apply(set).to_dict()
all_items = set(iid2idx.values())

# build the training triples
train_data = []

for user_idx in tqdm(uidx2idxs):
    usr_items = uidx2idxs[user_idx]
    pos_items = np.array(list(uidx2posidxs[user_idx]))
    neg_pool = np.array(list(all_items - usr_items))
    
    if len(neg_pool) == 0 or len(pos_items) == 0:
        continue  # safety check
    
    # For each positive item, sample K negatives (vectorized)
    k = min(5, len(neg_pool))
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

train_data = train_data.merge(train_df[['user_idx', 'item_idx','last_5_items']],
                              left_on=['pos_item_idx', 'user_idx'], right_on=['item_idx', 'user_idx'], how='inner')
train_data.drop(columns=['item_idx'], inplace=True)
train_data = train_data.merge(movies_df, left_on='pos_item_idx', right_on='item_idx', how='inner')
train_data.drop(columns=['item_idx'], inplace=True)
train_data.rename(columns={'last_5_items': 'last_5_items_idx'}, inplace=True)

train_data.rename(columns={'genres_idx': 'pos_item_genres_idx',
                            'year_idx': 'pos_item_year_idx',
                            'tags_idx': 'pos_item_tags_idx'}, inplace=True)

train_data = train_data.merge(movies_df, left_on='neg_item_idx', right_on='item_idx', how='inner')
train_data.drop(columns=['item_idx'], inplace=True)
train_data.rename(columns={'genres_idx': 'neg_item_genres_idx',
                            'year_idx': 'neg_item_year_idx',
                            'tags_idx': 'neg_item_tags_idx'}, inplace=True)
train_data['last_5_items_idx'] = train_data['last_5_items_idx'].apply(lambda x: [int(i) for i in x])

# train_data.head() to json output print
train_data_json = train_data.head().to_json(orient='records', lines=True)


GENRE_LIST_FIXED_LEN = 10
TAG_LIST_FIXED_LEN = 3
movie_history_max_len = 5
padding_idx_val_for_movies = len(iid2idx)

train_dataset = TwoTowerTrainingDataset(
    dataframe=train_data,
    movie_history_len=movie_history_max_len,
    genre_list_len=GENRE_LIST_FIXED_LEN,
    tag_list_len=TAG_LIST_FIXED_LEN,
    movie_padding_idx=padding_idx_val_for_movies
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=4096,
    shuffle=True
)

genre_dict_path = os.path.join(helper_files_path, 'genre_dict.pkl')
year_dict_path = os.path.join(helper_files_path, 'year_dict.pkl')
tag_dict_path = os.path.join(helper_files_path, 'tag_dict.pkl')

with open(genre_dict_path, 'rb') as f:
    genre_dict = pickle.load(f)
with open(year_dict_path, 'rb') as f:
    year_dict = pickle.load(f)
with open(tag_dict_path, 'rb') as f:
    tag_dict = pickle.load(f)


num_users_val = len(uid2idx)
num_movies_val = len(iid2idx) # Actual count of unique movies (indices 0 to num_movies_val-1)
num_genres_val = len(genre_dict)
num_years_val = len(year_dict)
num_movie_tags_vocab_val = len(tag_dict)

# Model parameters
embedding_dim_val = 128
year_embedding_dim_val = 32
hidden_dim_lstm_val = 64
hidden_dim_mlp_user_val = 256
hidden_dim_mlp_movie_val = 256
# movie_history_max_len is already defined and used for Dataset

print("Instantiating TwoTowerSystem model...")
model = TwoTowerSystem(
    num_users=num_users_val,
    num_movies=num_movies_val,       # Actual count for vocab size; padding index will be num_movies_val
    num_genres=num_genres_val,
    num_years=num_years_val,
    num_movie_tags_vocab=num_movie_tags_vocab_val,
    genre_list_len=GENRE_LIST_FIXED_LEN,
    tag_list_len=TAG_LIST_FIXED_LEN,
    # movie_history_len is used by Dataset, not directly by TwoTowerSystem.__init__
    embedding_dim=embedding_dim_val,
    year_embedding_dim=year_embedding_dim_val,
    hidden_dim_lstm=hidden_dim_lstm_val,
    hidden_dim_mlp_user=hidden_dim_mlp_user_val,
    hidden_dim_mlp_movie=hidden_dim_mlp_movie_val
)
model.to(device)
print("Model instantiated and moved to device.")

# BPR Loss function
def bpr_loss(positive_scores, negative_scores):
    loss = -torch.log(torch.sigmoid(positive_scores - negative_scores)).mean()
    return loss

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # Example LR and L2 reg

# LR Scheduler (optional, but you used it before)
iterations_per_epoch = len(train_dataloader)
scheduler_step_size = 4 * iterations_per_epoch # Example: 4 epochs for one triangle
scheduler = CyclicLR(
    optimizer,
    base_lr=1e-4,    # Example base LR
    max_lr=1e-3,     # Example max LR
    step_size_up=scheduler_step_size,
    mode='triangular',
    cycle_momentum=False
)

# Model saving parameters
model_save_dir = os.path.join(root_dir, 'models', 'two_tower_model_files')
os.makedirs(model_save_dir, exist_ok=True)
model_filename = f'two_tower_model_emb{embedding_dim_val}.pth'
model_save_path = os.path.join(model_save_dir, model_filename)


# ----------- RETRIEVER MODEL TRAINING -----------
num_epochs = 20
best_train_loss = float('inf')

print(f"Starting Two-Tower Model training for {num_epochs} epochs on device: {device}")
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    start_time = time.time()
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

    for batch_idx, (user_inputs_b, pos_movie_inputs_b, neg_movie_inputs_b) in enumerate(progress_bar):
        # Move all tensors within dictionaries to the device
        user_inputs_b = {k: v.to(device) for k, v in user_inputs_b.items()}
        pos_movie_inputs_b = {k: v.to(device) for k, v in pos_movie_inputs_b.items()}
        neg_movie_inputs_b = {k: v.to(device) for k, v in neg_movie_inputs_b.items()}

        optimizer.zero_grad()
        positive_scores, negative_scores = model(user_inputs_b,
                                                 pos_movie_inputs_b,
                                                 neg_movie_inputs_b)
        
        loss = bpr_loss(positive_scores, negative_scores)
        loss.backward()
        optimizer.step()
        
        if scheduler:
             scheduler.step() # CyclicLR steps per batch

        epoch_train_loss += loss.item()
        if (batch_idx + 1) % 200 == 0: # Log progress less frequently for large datasets
             progress_bar.set_postfix({'loss': loss.item(), 
                                       'lr': scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']})

    avg_epoch_train_loss = epoch_train_loss / len(train_dataloader)
    epoch_time = time.time() - start_time
    current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_epoch_train_loss:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")

    if avg_epoch_train_loss < best_train_loss:
        best_train_loss = avg_epoch_train_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path} (Loss: {best_train_loss:.4f})")

print("Training finished.")


# STAGE 2: RE-RANKER NN MODEL TRAINING

print("\n--- Starting Re-Ranker NN Model Preparation & Training ---")

# 1. Load the best trained Two-Tower model (Retriever)
# -------------------------------------------------------------------
print("Loading trained Two-Tower (Retriever) model...")
# Instantiate a new model with the same architecture as the trained one
retriever_model = TwoTowerSystem( # Use the same parameters as when you trained it
    num_users=num_users_val, num_movies=num_movies_val, num_genres=num_genres_val,
    num_years=num_years_val, num_movie_tags_vocab=num_movie_tags_vocab_val,
    genre_list_len=GENRE_LIST_FIXED_LEN, tag_list_len=TAG_LIST_FIXED_LEN,
    embedding_dim=embedding_dim_val, year_embedding_dim=year_embedding_dim_val,
    hidden_dim_lstm=hidden_dim_lstm_val, hidden_dim_mlp_user=hidden_dim_mlp_user_val,
    hidden_dim_mlp_movie=hidden_dim_mlp_movie_val
)
retriever_model.load_state_dict(torch.load(model_save_path, map_location=device))
retriever_model.to(device)
retriever_model.eval()
print(f"Retriever model loaded from {model_save_path}")

frozen_retriever_movie_embedding_layer = retriever_model.shared_movie_embedding_layer
frozen_retriever_movie_embedding_layer.weight.requires_grad = False

K_candidates = 500

unique_user_history_df = train_data.copy()[['user_idx', 'last_5_items_idx']]
unique_user_history_df['last_5_items_idx'] = unique_user_history_df['last_5_items_idx'].apply(lambda x: tuple(x))
unique_user_history_df = unique_user_history_df.drop_duplicates().reset_index(drop=True)
unique_user_history_df['last_5_items_idx'] = unique_user_history_df['last_5_items_idx'].apply(lambda x: list(x))

# Pre-compute all item embeddings for fast scoring
all_items_retriever_embeddings = frozen_retriever_movie_embedding_layer.weight.detach()[:num_movies_val]


# Generate candidates in batches for efficiency
def sample_negatives_from_candidates(candidates, pos_interactions, num_samples=5):
    candidates = set(candidates) - pos_interactions
    if len(candidates) <= num_samples:
        return list(candidates)
    return np.random.choice(np.array(list(candidates)), size=num_samples, replace=False).tolist()

candidate_results = []
cand_gen_batch_size = 512
for i in tqdm(range(0, len(unique_user_history_df), cand_gen_batch_size), desc="Generating candidates"):
    batch_df = unique_user_history_df.iloc[i:i+cand_gen_batch_size]
    user_ids_tensor = torch.tensor(batch_df['user_idx'].values, dtype=torch.long).to(device)
    history_tensor = torch.tensor(np.stack(batch_df['last_5_items_idx'].values), dtype=torch.long).to(device)

    user_input_for_retriever = {
        'user_id': user_ids_tensor,
        'movie_history_ids': history_tensor
    }

    with torch.no_grad():
        user_embeddings = retriever_model.user_tower(**user_input_for_retriever)

    scores = torch.matmul(user_embeddings, all_items_retriever_embeddings.T)
    _, top_k_indices = torch.topk(scores, K_candidates)

    # For each user in the batch, sample negatives from candidates
    for idx, row in batch_df.iterrows():
        user_idx = row['user_idx']
        pos_interactions = set(uidx2posidxs[user_idx])
        candidates = top_k_indices[idx - batch_df.index[0]].cpu().tolist()
        sampled_negatives = sample_negatives_from_candidates(candidates, pos_interactions, num_samples=5)
        candidate_results.append({
            'user_idx': user_idx,
            'last_5_items_idx': row['last_5_items_idx'],
            'sampled_negatives': sampled_negatives
        })

candidates_df = pd.DataFrame(candidate_results).explode('sampled_negatives').reset_index(drop=True)
candidates_df.rename(columns={'sampled_negatives': 'neg_item_idx'}, inplace=True)

# add pos_item_idx using train_data
temp_df = train_data[['user_idx', 'last_5_items_idx', 'pos_item_idx']].copy()
temp_df['last_5_items_idx'] = temp_df['last_5_items_idx'].apply(lambda x: tuple(x))
temp_df = temp_df.drop_duplicates().reset_index(drop=True)

candidates_df['last_5_items_idx'] = candidates_df['last_5_items_idx'].apply(lambda x: tuple(x))
candidates_df = candidates_df.merge(temp_df[['user_idx', 'pos_item_idx', 'last_5_items_idx']],
                                    on=['user_idx', 'last_5_items_idx'], how='inner')

candidates_df = candidates_df.merge(movies_df, left_on='pos_item_idx', right_on='item_idx', how='inner')
candidates_df.drop(columns=['item_idx'], inplace=True)
candidates_df.rename(columns={
    'genres_idx': 'ranker_pos_item_genres_idx',
    'year_idx': 'ranker_pos_item_year_idx',
    'tags_idx': 'ranker_pos_item_tags_idx'
}, inplace=True)

candidates_df = candidates_df.merge(movies_df, left_on='neg_item_idx', right_on='item_idx', how='inner')
candidates_df.drop(columns=['item_idx'], inplace=True)
candidates_df.rename(columns={
    'genres_idx': 'ranker_neg_item_genres_idx',
    'year_idx': 'ranker_neg_item_year_idx',
    'tags_idx': 'ranker_neg_item_tags_idx'
}, inplace=True)

candidates_df.rename(columns={'pos_item_idx': 'ranker_pos_item_idx',
                                 'neg_item_idx': 'ranker_neg_item_idx'}, inplace=True)

candidates_df['last_5_items_idx'] = candidates_df['last_5_items_idx'].apply(lambda x: [int(i) for i in x])

final_columns = [
    'user_idx', 'last_5_items_idx', 'ranker_pos_item_idx', 'ranker_pos_item_genres_idx',
    'ranker_pos_item_year_idx', 'ranker_pos_item_tags_idx', 'ranker_neg_item_idx',
    'ranker_neg_item_genres_idx', 'ranker_neg_item_year_idx', 'ranker_neg_item_tags_idx'
]

# check if all columns are present
for col in final_columns:
    if col not in candidates_df.columns:
        raise ValueError(f"Missing column in candidates_df: {col}")
    
candidates_df = candidates_df[final_columns].reset_index(drop=True)

# save candidates_df as pickle
candidates_df_path = os.path.join(model_save_dir, 'ranker_train_data.pkl')
with open(candidates_df_path, 'wb') as f:
    pickle.dump(candidates_df, f)

# save validation data as pickle
val_df_path = os.path.join(model_save_dir, 'ranker_val_data.pkl')
with open(val_df_path, 'wb') as f:
    pickle.dump(val_df, f)
    
# save two tower train_data as pickle
two_tower_train_data_path = os.path.join(model_save_dir, 'two_tower_train_data.pkl')
with open(two_tower_train_data_path, 'wb') as f:
    pickle.dump(train_data, f)
    
# save twotower uidx2posidxs as pickle
twotower_uidx2posidxs_path = os.path.join(model_save_dir, 'twotower_uidx2posidxs.pkl')
with open(twotower_uidx2posidxs_path, 'wb') as f:
    pickle.dump(uidx2posidxs, f)
    
# save twotower uidx2idxs as pickle
twotower_uidx2idxs_path = os.path.join(model_save_dir, 'twotower_uidx2idxs.pkl')
with open(twotower_uidx2idxs_path, 'wb') as f:
    pickle.dump(uidx2idxs, f)
    
# save iid2idx as pickle
iid2idx_path = os.path.join(model_save_dir, 'iid2idx.pkl')
with open(iid2idx_path, 'wb') as f:
    pickle.dump(iid2idx, f)
    
    
# save two tower config as pickle
two_tower_config = {
    'num_users': num_users_val,
    'num_movies': num_movies_val,
    'num_genres': num_genres_val,
    'num_years': num_years_val,
    'num_movie_tags_vocab': num_movie_tags_vocab_val,
    'genre_list_len': GENRE_LIST_FIXED_LEN,
    'tag_list_len': TAG_LIST_FIXED_LEN,
    'embedding_dim': embedding_dim_val,
    'year_embedding_dim': year_embedding_dim_val,
    'hidden_dim_lstm': hidden_dim_lstm_val,
    'hidden_dim_mlp_user': hidden_dim_mlp_user_val,
    'hidden_dim_mlp_movie': hidden_dim_mlp_movie_val
}
two_tower_config_path = os.path.join(model_save_dir, 'two_tower_config.pkl')
with open(two_tower_config_path, 'wb') as f:
    pickle.dump(two_tower_config, f)
    

# save two tower dataset parameters as pickle
two_tower_dataset_params = {
    'movie_history_len': movie_history_max_len,
    'genre_list_len': GENRE_LIST_FIXED_LEN,
    'tag_list_len': TAG_LIST_FIXED_LEN,
    'movie_padding_idx': padding_idx_val_for_movies
}
two_tower_dataset_params_path = os.path.join(model_save_dir, 'two_tower_dataset_params.pkl')
with open(two_tower_dataset_params_path, 'wb') as f:
    pickle.dump(two_tower_dataset_params, f)
    
# save tag_dict, genre_dict, year_dict as pickle
tag_dict_path = os.path.join(model_save_dir, 'tag_dict.pkl')
with open(tag_dict_path, 'wb') as f:
    pickle.dump(tag_dict, f)
genre_dict_path = os.path.join(model_save_dir, 'genre_dict.pkl')
with open(genre_dict_path, 'wb') as f:
    pickle.dump(genre_dict, f)
year_dict_path = os.path.join(model_save_dir, 'year_dict.pkl')
with open(year_dict_path, 'wb') as f:
    pickle.dump(year_dict, f)
    
# save movies_df as pickle
movies_df_path = os.path.join(model_save_dir, 'movies_df.pkl')
with open(movies_df_path, 'wb') as f:
    pickle.dump(movies_df, f)