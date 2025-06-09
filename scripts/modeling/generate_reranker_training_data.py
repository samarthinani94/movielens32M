import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

import torch

sys.path.append("../../")
from scripts.utils import root_dir
from scripts.modeling.two_tower_retriever_model import TwoTowerSystem

# --- 1. CONFIGURATION ---
RETRIEVER_MODEL_DIR = os.path.join(root_dir, 'models', 'two_tower_retriever_files')
RERANKER_SAVE_DIR = os.path.join(root_dir, 'models', 'reranker_files')
os.makedirs(RERANKER_SAVE_DIR, exist_ok=True)

# The K value we decided on based on recall results
CANDIDATE_K = 500
# How many negative samples to generate for each positive item found in the candidate set
NUM_NEGATIVE_SAMPLES = 4
BATCH_SIZE = 1024

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


# --- 2. LOAD ARTIFACTS FROM RETRIEVER TRAINING ---
print("--- Loading Retriever Artifacts ---")
with open(os.path.join(RETRIEVER_MODEL_DIR, 'model_config.pkl'), 'rb') as f:
    retriever_config = pickle.load(f)
with open(os.path.join(RETRIEVER_MODEL_DIR, 'movies_df.pkl'), 'rb') as f:
    movies_df = pickle.load(f)
with open(os.path.join(RETRIEVER_MODEL_DIR, 'val_data.pkl'), 'rb') as f:
    # We use the validation data to define the set of users we'll generate training data for
    # NOTE: This assumes the users in validation also appear in training, which is typical.
    # For a stricter setup, one might load the full train_df as well.
    val_data = pickle.load(f)

# Load the trained retriever model
print("Loading trained retriever model...")
retriever_model = TwoTowerSystem(retriever_config).to(device)
retriever_model.load_state_dict(torch.load(os.path.join(RETRIEVER_MODEL_DIR, 'two_tower_retriever.pth'), map_location=device))
retriever_model.eval()


# --- 3. PREPARE DATA FOR CANDIDATE GENERATION ---
print("--- Preparing Data ---")
# Get unique user histories for whom we will generate candidates
user_histories = val_data[['user_idx', 'last_5_items_idx']].copy()
user_histories['last_5_items_idx'] = user_histories['last_5_items_idx'].apply(lambda x: tuple(x))
user_histories.drop_duplicates(inplace=True)
user_histories['last_5_items_idx'] = user_histories['last_5_items_idx'].apply(lambda x: list(x))
print(f"Found {len(user_histories)} unique user contexts to process.")

# Ground truth: what items did users actually interact with (positively)?
# Using the validation set for this example, but in a real-world scenario, you'd use the training set.
# This assumes we're generating training data for users who also appear in the validation set.
ground_truth = val_data[val_data['rating'] >= 4.0].groupby('user_idx')['item_idx'].apply(set).to_dict()

# Pre-compute all item embeddings for fast scoring
with torch.no_grad():
    item_ids = torch.tensor(movies_df['item_idx'].values, dtype=torch.long).to(device)
    genre_indices = torch.tensor(np.stack(movies_df['genres_idx'].values), dtype=torch.long).to(device)
    year_indices = torch.tensor(movies_df['year_idx'].values, dtype=torch.long).to(device)
    tag_indices = torch.tensor(np.stack(movies_df['tags_idx'].values), dtype=torch.long).to(device)
    all_item_embeddings = retriever_model.movie_tower(item_ids, genre_indices, year_indices, tag_indices)


# --- 4. GENERATE RE-RANKER TRAINING DATA ---
print(f"--- Generating Training Data (K={CANDIDATE_K}) ---")
reranker_triplets = []

for i in tqdm(range(0, len(user_histories), BATCH_SIZE), desc="Generating Re-ranker Data"):
    batch_df = user_histories.iloc[i:i+BATCH_SIZE]
    user_ids = torch.tensor(batch_df['user_idx'].values, dtype=torch.long).to(device)
    history_ids = torch.tensor(np.stack(batch_df['last_5_items_idx'].values), dtype=torch.long).to(device)

    # Stage 1: Get Top K Candidates from Retriever
    with torch.no_grad():
        user_embeddings = retriever_model.user_tower(user_ids, history_ids)
        scores = torch.matmul(user_embeddings, all_item_embeddings.T)
        _, top_k_indices = torch.topk(scores, k=CANDIDATE_K, dim=1)
    
    top_k_indices = top_k_indices.cpu().numpy()

    # Stage 2: Create Training Triplets from Candidates
    for j, user_idx in enumerate(batch_df['user_idx']):
        candidate_items = set(top_k_indices[j])
        true_positives = ground_truth.get(user_idx, set())
        
        # Find which of the true positives were successfully recalled
        valid_pos_items = candidate_items.intersection(true_positives)

        if not valid_pos_items:
            continue
        
        # For each valid positive, sample negatives from the *other* candidates
        for pos_item_idx in valid_pos_items:
            negative_pool = list(candidate_items - {pos_item_idx})
            if not negative_pool:
                continue
            
            num_to_sample = min(NUM_NEGATIVE_SAMPLES, len(negative_pool))
            sampled_negatives = np.random.choice(negative_pool, size=num_to_sample, replace=False)
            
            for neg_item_idx in sampled_negatives:
                reranker_triplets.append([user_idx, pos_item_idx, neg_item_idx])

# --- 5. ASSEMBLE AND SAVE FINAL DATAFRAME ---
if not reranker_triplets:
    print("Warning: No training samples were generated. The retriever might have 0 recall for the chosen users.")
else:
    reranker_df = pd.DataFrame(reranker_triplets, columns=['user_idx', 'pos_item_idx', 'neg_item_idx'])
    
    # Merge features for all items
    reranker_df = pd.merge(reranker_df, movies_df, left_on='pos_item_idx', right_on='item_idx').drop('item_idx', axis=1).rename(columns={'genres_idx': 'pos_item_genres_idx', 'year_idx': 'pos_item_year_idx', 'tags_idx': 'pos_item_tags_idx'})
    reranker_df = pd.merge(reranker_df, movies_df, left_on='neg_item_idx', right_on='item_idx').drop('item_idx', axis=1).rename(columns={'genres_idx': 'neg_item_genres_idx', 'year_idx': 'neg_item_year_idx', 'tags_idx': 'neg_item_tags_idx'})

    output_path = os.path.join(RERANKER_SAVE_DIR, 'reranker_training_data.pkl')
    reranker_df.to_pickle(output_path)
    print(f"\nSuccessfully generated {len(reranker_df)} training samples.")
    print(f"Re-ranker training data saved to: {output_path}")

