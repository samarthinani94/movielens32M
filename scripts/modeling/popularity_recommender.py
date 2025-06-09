import pandas as pd
import numpy as np
import os
import sys
import pickle
sys.path.append("../../")

from scripts.utils import root_dir
from tqdm import tqdm

MODEL_ARTIFACTS_DIR = os.path.join(root_dir, 'models', 'two_tower_model_files')

print("Loading data...")
with open(os.path.join(MODEL_ARTIFACTS_DIR, 'ranker_val_data.pkl'), 'rb') as f:
    val_df = pickle.load(f)
with open(os.path.join(MODEL_ARTIFACTS_DIR, 'two_tower_train_data.pkl'), 'rb') as f:
    # We need the full training data to know what each user has seen
    full_train_df = pickle.load(f)
print("Data loaded.")


# Create a set of interactions for quick lookup
train_interactions = full_train_df.groupby('user_idx')['pos_item_idx'].apply(set).to_dict()

# Calculate popularity from the training interactions
item_popularity = full_train_df['pos_item_idx'].value_counts().reset_index()
item_popularity.columns = ['item_idx', 'popularity']
# We need more than 10 to have a pool to select from after filtering
top_popular_movies_pool = item_popularity['item_idx'].tolist()

# Isolate ground truth for validation
val_df_truth = val_df[['user_idx', 'item_idx', 'rating']].copy().drop_duplicates()


def ndcg_at_k(popular_pool, val_df, train_seen_dict, k=10):
    
    distinct_users = val_df['user_idx'].unique()
    ndcg_scores = []
    
    print(f"Calculating nDCG@{k} for {len(distinct_users)} users...")
    for user in tqdm(distinct_users):
        
        # Get items this user has seen in training
        seen_items = train_seen_dict.get(user, set())
        
        # Filter the popular pool to get top K *new* items for this user
        user_recommendations = [item for item in popular_pool if item not in seen_items][:k]
        
        if not user_recommendations:
            continue

        user_truth = val_df[val_df['user_idx'] == user]
        item_rel_dict = user_truth.set_index('item_idx')['rating'].to_dict() 
        
        # Calculate DCG for this user's recommendations
        preds_df = pd.DataFrame(user_recommendations, columns=['item_idx'])
        preds_df['relevance'] = preds_df['item_idx'].map(item_rel_dict).fillna(0)
        
        dcg = (preds_df['relevance'] / np.log2(np.arange(2, len(preds_df) + 2))).sum()
        
        # Calculate IDCG
        ideal_relevances = user_truth['rating'].sort_values(ascending=False).head(k)
        idcg = (ideal_relevances.values / np.log2(np.arange(2, len(ideal_relevances) + 2))).sum()
        
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)


print("\nRunning popularity baseline evaluation...")
ndcg_score = ndcg_at_k(top_popular_movies_pool, val_df_truth, train_interactions, k=10)
        
print("\n" + "="*30)
print(f"Popularity Baseline nDCG@10: {ndcg_score:.4f}")
print("="*30)
