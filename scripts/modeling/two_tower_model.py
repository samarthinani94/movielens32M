    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class MovieTower(nn.Module):
    def __init__(self,
                 num_genres: int,
                 num_years: int,
                 num_movie_tags: int,    # Vocabulary size for movie tags
                 genre_list_length: int, # Fixed length of genre list per movie (e.g., 10)
                 tag_list_length: int,   # Fixed length of tag list per movie (e.g., 3)
                 embedding_dim: int = 64,
                 year_embedding_dim: int = 16,
                 hidden_dim_mlp: int = 128,
                 movie_embedding_layer_for_target: nn.Embedding = None,
                 num_movies: int = None): # Only needed if creating its own embedding layer
        super().__init__()

        if movie_embedding_layer_for_target is not None:
            self.movie_embedding = movie_embedding_layer_for_target
        elif num_movies is not None:
            print("MovieTower: Creating its own movie embedding layer.")
            # Assuming 0 is not a valid movie_idx if used as padding for movie_id itself.
            # If movie_id is never padded, padding_idx=0 might be okay if 0 is an <UNK> movie.
            # For consistency with history, if movie_id could be padded, use the history's padding_idx.
            self.movie_embedding = nn.Embedding(num_movies, embedding_dim, padding_idx=None) # Or specific padding_idx
        else:
            raise ValueError("MovieTower requires either 'movie_embedding_layer_for_target' or 'num_movies'.")

        self.embedding_dim = self.movie_embedding.embedding_dim
        self.genre_list_length = genre_list_length
        self.tag_list_length = tag_list_length
        
        # Standard Embeddings for fixed-length, padded multi-hot features
        # padding_idx=0 assumes 0 is your designated padding value for genre/tag indices
        self.genre_embedding = nn.Embedding(num_genres, self.embedding_dim, padding_idx=0)
        self.movie_tag_embedding = nn.Embedding(num_movie_tags, self.embedding_dim, padding_idx=0)
        
        self.year_embedding = nn.Embedding(num_years, year_embedding_dim, padding_idx=0) # Assuming 0 is padding/unknown year_idx

        combined_feature_dim = (
            self.embedding_dim +      # from movie_embedding
            self.embedding_dim +      # from aggregated genre_embedding
            year_embedding_dim +      # from year_embedding
            self.embedding_dim        # from aggregated movie_tag_embedding
        )

        self.fc1 = nn.Linear(combined_feature_dim, hidden_dim_mlp)
        self.fc2 = nn.Linear(hidden_dim_mlp, hidden_dim_mlp // 2)
        self.fc3 = nn.Linear(hidden_dim_mlp // 2, self.embedding_dim)

    def forward(self,
                movie_id: torch.Tensor,         # Shape: (batch_size)
                padded_genre_indices: torch.Tensor, # Shape: (batch_size, genre_list_length)
                year_idx: torch.Tensor,         # Shape: (batch_size)
                padded_tag_indices: torch.Tensor    # Shape: (batch_size, tag_list_length)
                ) -> torch.Tensor:

        m_embed = self.movie_embedding(movie_id)    # (batch_size, embedding_dim)
        y_embed = self.year_embedding(year_idx)    # (batch_size, year_embedding_dim)

        # Process fixed-length padded genres
        genre_embeds_all = self.genre_embedding(padded_genre_indices) # (batch_size, genre_list_length, embedding_dim)
        # Create mask for non-padding genre indices (assuming padding_idx is 0)
        genre_mask = (padded_genre_indices != 0).float().unsqueeze(-1) # (batch_size, genre_list_length, 1)
        genre_embeds_masked = genre_embeds_all * genre_mask
        # Sum valid embeddings and divide by count of valid embeddings (masked average)
        g_embed_aggregated = torch.sum(genre_embeds_masked, dim=1) / (torch.sum(genre_mask, dim=1).clamp(min=1e-9)) # (batch_size, embedding_dim)

        # Process fixed-length padded tags
        tag_embeds_all = self.movie_tag_embedding(padded_tag_indices) # (batch_size, tag_list_length, embedding_dim)
        tag_mask = (padded_tag_indices != 0).float().unsqueeze(-1) # (batch_size, tag_list_length, 1)
        tag_embeds_masked = tag_embeds_all * tag_mask
        mt_embed_aggregated = torch.sum(tag_embeds_masked, dim=1) / (torch.sum(tag_mask, dim=1).clamp(min=1e-9)) # (batch_size, embedding_dim)

        combined_movie_features = torch.cat([m_embed, g_embed_aggregated, y_embed, mt_embed_aggregated], dim=1)
        
        x = F.relu(self.fc1(combined_movie_features))
        x = F.relu(self.fc2(x))
        movie_final_embedding = self.fc3(x)
        return movie_final_embedding


class UserTower(nn.Module):
    def __init__(self,
                 num_users: int,
                 embedding_dim: int = 64,
                 hidden_dim_lstm: int = 32,
                 hidden_dim_mlp: int = 128,
                 movie_embedding_layer_for_history: nn.Embedding = None):
        super().__init__()
        if movie_embedding_layer_for_history is None:
            raise ValueError("A shared 'movie_embedding_layer_for_history' must be provided.")

        # Ensure embedding_dim matches the shared layer if it's for concatenation
        # Or, use movie_embedding_layer_for_history.embedding_dim for LSTM input size
        self.user_native_embedding_dim = embedding_dim # For user_id
        self.history_processed_embedding_dim = hidden_dim_lstm # LSTM output dim
        self.final_mlp_output_embedding_dim = movie_embedding_layer_for_history.embedding_dim # Match output dim

        self.user_embedding = nn.Embedding(num_users, self.user_native_embedding_dim, padding_idx=0) # Assuming 0 is unused user_idx or for padding
        self.movie_history_embedding_layer = movie_embedding_layer_for_history
        self.movie_history_lstm = nn.LSTM(movie_embedding_layer_for_history.embedding_dim, 
                                          hidden_dim_lstm, batch_first=True)

        combined_feature_dim = self.user_native_embedding_dim + hidden_dim_lstm
        
        self.fc1 = nn.Linear(combined_feature_dim, hidden_dim_mlp)
        self.fc2 = nn.Linear(hidden_dim_mlp, hidden_dim_mlp // 2)
        # Output embedding dim should match the movie embeddings for dot product
        self.fc3 = nn.Linear(hidden_dim_mlp // 2, self.final_mlp_output_embedding_dim)

    def forward(self,
                user_id: torch.Tensor,
                movie_history_ids: torch.Tensor,
                ) -> torch.Tensor:
        u_embed = self.user_embedding(user_id)
        mh_embeds = self.movie_history_embedding_layer(movie_history_ids)
        _, (mh_lstm_final_hidden, _) = self.movie_history_lstm(mh_embeds)
        mh_representation = mh_lstm_final_hidden.squeeze(0)
        combined_user_features = torch.cat([u_embed, mh_representation], dim=1)
        x = F.relu(self.fc1(combined_user_features))
        x = F.relu(self.fc2(x))
        user_final_embedding = self.fc3(x)
        return user_final_embedding


class TwoTowerSystem(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_movies: int, # Actual count of unique movies, for padding_idx logic
                 num_genres: int,
                 num_years: int,
                 num_movie_tags_vocab: int,
                 genre_list_len: int, # Fixed N_g for MovieTower
                 tag_list_len: int,   # Fixed N_t for MovieTower
                 # movie_history_len is not directly used by model __init__ if data is pre-padded
                 embedding_dim: int = 64,
                 year_embedding_dim: int = 16,
                 hidden_dim_lstm: int = 32,
                 hidden_dim_mlp_user: int = 128,
                 hidden_dim_mlp_movie: int = 128):
        super().__init__()

        self.padding_idx_for_movies = num_movies # Value of the padding index for movie history
        self.shared_movie_embedding_layer = nn.Embedding(
            num_embeddings=num_movies + 1, # Vocab size including padding index
            embedding_dim=embedding_dim,
            padding_idx=self.padding_idx_for_movies
        )

        self.user_tower = UserTower(
            num_users=num_users,
            embedding_dim=embedding_dim, # This is for user_id embedding itself
            hidden_dim_lstm=hidden_dim_lstm,
            hidden_dim_mlp=hidden_dim_mlp_user,
            movie_embedding_layer_for_history=self.shared_movie_embedding_layer
        )

        self.movie_tower = MovieTower(
            num_movies=num_movies +1, # Total size for its own fallback if needed, consistent with shared.
            num_genres=num_genres,
            num_years=num_years,
            num_movie_tags=num_movie_tags_vocab,
            genre_list_length=genre_list_len,
            tag_list_length=tag_list_len,
            embedding_dim=embedding_dim, # This should match shared_movie_embedding_layer's dim
            year_embedding_dim=year_embedding_dim,
            hidden_dim_mlp=hidden_dim_mlp_movie,
            movie_embedding_layer_for_target=self.shared_movie_embedding_layer
        )

    def forward(self,
                user_inputs: dict,          # Keys: 'user_id', 'movie_history_ids'
                positive_movie_inputs: dict, # Keys: 'movie_id', 'padded_genre_indices', 'year_idx', 'padded_tag_indices'
                negative_movie_inputs: dict = None):

        user_embedding = self.user_tower(
            user_id=user_inputs['user_id'],
            movie_history_ids=user_inputs['movie_history_ids']
        )

        positive_movie_embedding = self.movie_tower(
            movie_id=positive_movie_inputs['movie_id'],
            padded_genre_indices=positive_movie_inputs['padded_genre_indices'],
            year_idx=positive_movie_inputs['year_idx'],
            padded_tag_indices=positive_movie_inputs['padded_tag_indices']
            # Removed offsets, as EmbeddingBag is no longer used here
        )
        positive_scores = torch.sum(user_embedding * positive_movie_embedding, dim=1)

        if negative_movie_inputs is not None:
            negative_movie_embedding = self.movie_tower(
                movie_id=negative_movie_inputs['movie_id'],
                padded_genre_indices=negative_movie_inputs['padded_genre_indices'],
                year_idx=negative_movie_inputs['year_idx'],
                padded_tag_indices=negative_movie_inputs['padded_tag_indices']
            )
            negative_scores = torch.sum(user_embedding * negative_movie_embedding, dim=1)
            return positive_scores, negative_scores
        else:
            return positive_scores


class TwoTowerTrainingDataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 movie_history_len: int, # Max length you padded to
                 genre_list_len: int,    # Max genres you padded to
                 tag_list_len: int,      # Max tags you padded to
                 movie_padding_idx: int  # Value used for padding movie history
                 # genre_padding_idx is assumed to be 0 from your data prep
                 # tag_padding_idx is assumed to be 0 from your data prep
                ):
        self.dataframe = dataframe
        # Store lengths for potential use in __getitem__ if dynamic padding/truncation is needed
        # However, your data sample suggests these are already fixed in the DataFrame.
        self.movie_history_len = movie_history_len
        self.genre_list_len = genre_list_len
        self.tag_list_len = tag_list_len
        self.movie_padding_idx = movie_padding_idx # For movie history, usually len(iid2idx)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple:
        row = self.dataframe.iloc[idx]

        user_inputs = {
            'user_id': torch.tensor(int(row['user_idx']), dtype=torch.long),
            'movie_history_ids': torch.tensor(row['last_5_items_idx'], dtype=torch.long) # Assumes it's already a list of N ints
        }

        positive_movie_inputs = {
            'movie_id': torch.tensor(int(row['pos_item_idx']), dtype=torch.long),
            'padded_genre_indices': torch.tensor(row['pos_item_genres_idx'], dtype=torch.long), # List of N_g ints
            'year_idx': torch.tensor(int(row['pos_item_year_idx']), dtype=torch.long),
            'padded_tag_indices': torch.tensor(row['pos_item_tags_idx'], dtype=torch.long)    # List of N_t ints
        }

        negative_movie_inputs = {
            'movie_id': torch.tensor(int(row['neg_item_idx']), dtype=torch.long),
            'padded_genre_indices': torch.tensor(row['neg_item_genres_idx'], dtype=torch.long),
            'year_idx': torch.tensor(int(row['neg_item_year_idx']), dtype=torch.long),
            'padded_tag_indices': torch.tensor(row['neg_item_tags_idx'], dtype=torch.long)
        }
        return user_inputs, positive_movie_inputs, negative_movie_inputs
    

class ReRankerNN(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_genres: int,
                 num_years: int,
                 num_movie_tags_vocab: int,
                 embedding_dim: int,
                 year_embedding_dim: int,
                 hidden_dim_mlp: int, # A single MLP hidden dim for simplicity
                 retriever_movie_embedding_layer: nn.Embedding):
        super().__init__()

        self.retriever_movie_embedding_layer = retriever_movie_embedding_layer
        self.retriever_movie_embedding_layer.weight.requires_grad = False # Freeze retriever embeddings

        retriever_movie_emb_dim = self.retriever_movie_embedding_layer.embedding_dim

        # --- Re-ranker's Own Learnable Embedding Layers ---
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # We can still learn ranker-specific content embeddings to add more signal
        self.item_genre_embedding = nn.Embedding(num_genres, embedding_dim, padding_idx=0)
        self.item_year_embedding = nn.Embedding(num_years, year_embedding_dim, padding_idx=0)
        self.item_movie_tag_embedding = nn.Embedding(num_movie_tags_vocab, embedding_dim, padding_idx=0)
        
        # --- Final Interaction MLP ---
        # This MLP will learn all interactions from the combined feature vector.
        # Input: Concat(user_embed, retriever_movie_embed, genre_embed, year_embed, tag_embed)
        combined_feature_dim = (
            embedding_dim +             # from this ranker's user_embedding
            retriever_movie_emb_dim +   # from retriever's movie_embedding
            embedding_dim +             # from this ranker's genre aggregation
            year_embedding_dim +        # from this ranker's year_embedding
            embedding_dim               # from this ranker's tag aggregation
        )

        self.interaction_mlp = nn.Sequential(
            nn.Linear(combined_feature_dim, hidden_dim_mlp),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim_mlp, hidden_dim_mlp // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim_mlp // 2, 1) # Final single score output
        )

    def _get_item_features(self, movie_id, padded_genre_indices, year_idx, padded_tag_indices):
        # 1. Get main item embedding from the (frozen) retriever
        retrieved_item_embed = self.retriever_movie_embedding_layer(movie_id)

        # 2. Process content features with ranker's own embedding layers
        genre_embeds_all = self.item_genre_embedding(padded_genre_indices)
        genre_mask = (padded_genre_indices != 0).float().unsqueeze(-1)
        g_embed_aggregated = torch.sum(genre_embeds_all * genre_mask, dim=1) / (torch.sum(genre_mask, dim=1).clamp(min=1e-9))
        
        y_embed = self.item_year_embedding(year_idx)

        tag_embeds_all = self.item_movie_tag_embedding(padded_tag_indices)
        tag_mask = (padded_tag_indices != 0).float().unsqueeze(-1)
        mt_embed_aggregated = torch.sum(tag_embeds_all * tag_mask, dim=1) / (torch.sum(tag_mask, dim=1).clamp(min=1e-9))
        
        return retrieved_item_embed, g_embed_aggregated, y_embed, mt_embed_aggregated

    def forward(self, 
                user_inputs: dict,
                positive_item_inputs: dict,
                negative_item_inputs: dict
               ):
        
        # --- User Representation ---
        # For this simpler model, we only need the user ID embedding.
        # History processing could be added back, but let's keep it clean for now.
        user_representation = self.user_embedding(user_inputs['user_id'])

        # --- Positive Item Representation & Score ---
        (pos_retrieved_embed, 
         pos_genre_embed, 
         pos_year_embed, 
         pos_tag_embed) = self._get_item_features(
            movie_id=positive_item_inputs['movie_id'],
            padded_genre_indices=positive_item_inputs['padded_genre_indices'],
            year_idx=positive_item_inputs['year_idx'],
            padded_tag_indices=positive_item_inputs['padded_tag_indices']
        )
        
        positive_interaction_input = torch.cat([
            user_representation, 
            pos_retrieved_embed, 
            pos_genre_embed, 
            pos_year_embed, 
            pos_tag_embed
        ], dim=1)
        positive_score = self.interaction_mlp(positive_interaction_input).squeeze(-1)

        # --- Negative Item Representation & Score ---
        (neg_retrieved_embed, 
         neg_genre_embed, 
         neg_year_embed, 
         neg_tag_embed) = self._get_item_features(
            movie_id=negative_item_inputs['movie_id'],
            padded_genre_indices=negative_item_inputs['padded_genre_indices'],
            year_idx=negative_item_inputs['year_idx'],
            padded_tag_indices=negative_item_inputs['padded_tag_indices']
        )

        negative_interaction_input = torch.cat([
            user_representation,
            neg_retrieved_embed,
            neg_genre_embed,
            neg_year_embed,
            neg_tag_embed
        ], dim=1)
        negative_score = self.interaction_mlp(negative_interaction_input).squeeze(-1)
        
        return positive_score, negative_score
    
    
class ReRankerTrainingDataset(Dataset):
    def __init__(self,
                 reranker_triples_dataframe: pd.DataFrame,
                 # Params for fixed lengths, used to ensure data consistency if needed by a helper
                 # Though the dataframe should already have fixed-length lists
                 movie_history_len: int,
                 genre_list_len: int,
                 tag_list_len: int
                ):
        self.dataframe = reranker_triples_dataframe
        self.movie_history_len = movie_history_len # from retriever's data prep
        self.genre_list_len = genre_list_len     # from retriever's data prep
        self.tag_list_len = tag_list_len         # from retriever's data prep

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple:
        row = self.dataframe.iloc[idx]

        # User features (should be consistent with how retriever gets them)
        user_inputs = {
            'user_id': torch.tensor(int(row['user_idx']), dtype=torch.long),
            'movie_history_ids': torch.tensor(row['last_5_items_idx'], dtype=torch.long)
        }

        # Positive item features
        positive_item_inputs = {
            'movie_id': torch.tensor(int(row['ranker_pos_item_idx']), dtype=torch.long),
            'padded_genre_indices': torch.tensor(row['ranker_pos_item_genres_idx'], dtype=torch.long),
            'year_idx': torch.tensor(int(row['ranker_pos_item_year_idx']), dtype=torch.long),
            'padded_tag_indices': torch.tensor(row['ranker_pos_item_tags_idx'], dtype=torch.long)
        }

        # Negative item features
        negative_item_inputs = {
            'movie_id': torch.tensor(int(row['ranker_neg_item_idx']), dtype=torch.long),
            'padded_genre_indices': torch.tensor(row['ranker_neg_item_genres_idx'], dtype=torch.long),
            'year_idx': torch.tensor(int(row['ranker_neg_item_year_idx']), dtype=torch.long),
            'padded_tag_indices': torch.tensor(row['ranker_neg_item_tags_idx'], dtype=torch.long)
        }
        
        return user_inputs, positive_item_inputs, negative_item_inputs