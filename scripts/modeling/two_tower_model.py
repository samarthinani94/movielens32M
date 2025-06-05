# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np


# class MovieTower(nn.Module):
#     def __init__(self,
#                  num_genres: int,
#                  num_years: int,
#                  num_movie_tags: int,
#                  embedding_dim: int = 64,
#                  year_embedding_dim: int = 16,
#                  hidden_dim_mlp: int = 128,
#                  # Accept the shared movie embedding layer as an argument
#                  movie_embedding_layer_for_target: nn.Embedding = None,
#                  # num_movies is only needed if creating its own embedding layer
#                  num_movies: int = None):
#         super().__init__()

#         if movie_embedding_layer_for_target is not None:
#             self.movie_embedding = movie_embedding_layer_for_target
#             # Infer embedding_dim from the shared layer if not explicitly passed for consistency
#             # Or ensure embedding_dim passed matches the shared layer's dim.
#             # For simplicity, we assume embedding_dim matches.
#         elif num_movies is not None:
#             # Fallback if no shared layer is provided (for potential standalone use)
#             print("MovieTower: Creating its own movie embedding layer.")
#             self.movie_embedding = nn.Embedding(num_movies, embedding_dim, padding_idx=0)
#         else:
#             raise ValueError("MovieTower requires either 'movie_embedding_layer_for_target' or 'num_movies'.")

#         self.embedding_dim = self.movie_embedding.embedding_dim # Get actual embedding_dim from layer

#         self.genre_embedding_bag = nn.EmbeddingBag(num_genres, self.embedding_dim, mode="mean", padding_idx=0)
#         self.year_embedding = nn.Embedding(num_years, year_embedding_dim, padding_idx=0)
#         self.movie_tag_embedding_bag = nn.EmbeddingBag(num_movie_tags, self.embedding_dim, mode="mean", padding_idx=0)

#         combined_feature_dim = (
#             self.embedding_dim +      # from movie_embedding (now potentially shared)
#             self.embedding_dim +      # from genre_embedding_bag
#             year_embedding_dim +      # from year_embedding
#             self.embedding_dim        # from movie_tag_embedding_bag
#         )

#         self.fc1 = nn.Linear(combined_feature_dim, hidden_dim_mlp)
#         self.fc2 = nn.Linear(hidden_dim_mlp, hidden_dim_mlp // 2)
#         self.fc3 = nn.Linear(hidden_dim_mlp // 2, self.embedding_dim)

#     def forward(self,
#                 movie_id: torch.Tensor,
#                 genre_indices: torch.Tensor,
#                 genre_offsets: torch.Tensor,
#                 year_idx: torch.Tensor,
#                 movie_tag_indices: torch.Tensor,
#                 movie_tag_offsets: torch.Tensor) -> torch.Tensor:
#         m_embed = self.movie_embedding(movie_id)
#         g_embed = self.genre_embedding_bag(genre_indices, offsets=genre_offsets)
#         y_embed = self.year_embedding(year_idx)
#         mt_embed = self.movie_tag_embedding_bag(movie_tag_indices, offsets=movie_tag_offsets)
#         combined_movie_features = torch.cat([m_embed, g_embed, y_embed, mt_embed], dim=1)
#         x = F.relu(self.fc1(combined_movie_features))
#         x = F.relu(self.fc2(x))
#         movie_final_embedding = self.fc3(x)
#         return movie_final_embedding
    
    
# class UserTower(nn.Module):
#     def __init__(self,
#                  num_users: int,
#                  embedding_dim: int = 64,
#                  hidden_dim_lstm: int = 32,
#                  hidden_dim_mlp: int = 128,
#                  movie_embedding_layer_for_history: nn.Embedding = None): # Shared movie embedding layer
#         """
#         Initializes the UserTower.

#         Args:
#             num_users (int): Total number of unique users.
#             embedding_dim (int): Dimension for the main embeddings (user ID, shared movie, tags).
#             hidden_dim_lstm (int): Hidden dimension for the LSTM layers.
#             hidden_dim_mlp (int): Hidden dimension for the MLP layers.
#             movie_embedding_layer_for_history (nn.Embedding): The shared nn.Embedding layer for movies.
#                                                               Used to embed movie IDs in user's history.
#         """
#         super().__init__()
#         if movie_embedding_layer_for_history is None:
#             raise ValueError("A shared 'movie_embedding_layer_for_history' must be provided.")

#         self.embedding_dim = embedding_dim

#         # 1. Embedding for User ID
#         self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)

#         # 2. Movie History Processing (uses the shared movie embedding layer)
#         self.movie_history_embedding_layer = movie_embedding_layer_for_history
#         self.movie_history_lstm = nn.LSTM(embedding_dim, hidden_dim_lstm, batch_first=True)

#         # MLP to combine all features
#         # Calculate the total dimension of concatenated features
#         combined_feature_dim = (
#             embedding_dim +      # from user_embedding
#             hidden_dim_lstm     # from movie_history_lstm (final hidden state)
#             # hidden_dim_lstm      # from tag_history_lstm (final hidden state)
#         )
#         self.fc1 = nn.Linear(combined_feature_dim, hidden_dim_mlp)
#         self.fc2 = nn.Linear(hidden_dim_mlp, hidden_dim_mlp // 2)
#         self.fc3 = nn.Linear(hidden_dim_mlp // 2, embedding_dim) # Output final user embedding

#     def forward(self,
#                 user_id: torch.Tensor,
#                 movie_history_ids: torch.Tensor, # Assumes 0-padded if sequence is shorter
#                 ) -> torch.Tensor:
#         """
#         Forward pass for the UserTower.

#         Args:
#             user_id (torch.Tensor): Tensor of user IDs. Shape: (batch_size).
#             movie_history_ids (torch.Tensor): Tensor of movie ID sequences from user history.
#                                              Shape: (batch_size, movie_history_length).
#                                              Assumes 0 is used for padding shorter sequences.
#             tag_history_ids (torch.Tensor): Tensor of tag ID sequences from user's tag application history.
#                                            Shape: (batch_size, tag_history_length).
#                                            Assumes 0 is used for padding shorter sequences.
#         Returns:
#             torch.Tensor: The final user embedding. Shape: (batch_size, embedding_dim).
#         """

#         # 1. Get User ID embedding
#         u_embed = self.user_embedding(user_id)  # Shape: (batch_size, embedding_dim)

#         # 2. Process Movie History
#         # Embed movie IDs from history using the shared movie embedding layer
#         mh_embeds = self.movie_history_embedding_layer(movie_history_ids)  # Shape: (batch_size, movie_history_length, embedding_dim)
#         _, (mh_lstm_final_hidden, _) = self.movie_history_lstm(mh_embeds)
#         mh_representation = mh_lstm_final_hidden.squeeze(0)  # Shape: (batch_size, hidden_dim_lstm)

#         # Concatenate all feature embeddings
#         combined_user_features = torch.cat([u_embed, mh_representation], dim=1)

#         # Pass through MLP
#         x = F.relu(self.fc1(combined_user_features))
#         x = F.relu(self.fc2(x))
#         user_final_embedding = self.fc3(x)  # Shape: (batch_size, embedding_dim)

#         return user_final_embedding
    

# class TwoTowerSystem(nn.Module):
#     def __init__(self,
#                  num_users: int,
#                  num_movies: int,
#                  num_genres: int,
#                  num_years: int,
#                  num_movie_tags_vocab: int,
#                  movie_history_len: int, # Argument for UserTower (data prep)
#                  embedding_dim: int = 64,
#                  year_embedding_dim: int = 16,
#                  hidden_dim_lstm: int = 32,
#                  hidden_dim_mlp_user: int = 128,
#                  hidden_dim_mlp_movie: int = 128):
#         super().__init__()

#         # 1. Define the Shared Movie Embedding Layer
#         self.shared_movie_embedding_layer = nn.Embedding(num_movies+1, embedding_dim, padding_idx=num_movies)

#         # 2. Instantiate UserTower
#         self.user_tower = UserTower(
#             num_users=num_users,
#             embedding_dim=embedding_dim,
#             hidden_dim_lstm=hidden_dim_lstm,
#             hidden_dim_mlp=hidden_dim_mlp_user,
#             movie_embedding_layer_for_history=self.shared_movie_embedding_layer # Correctly passed
#         )

#         # 3. Instantiate MovieTower, NOW PASSING THE SHARED LAYER
#         self.movie_tower = MovieTower(
#             # num_movies argument is now optional if shared layer is passed, but good for clarity
#             num_movies=num_movies,
#             num_genres=num_genres,
#             num_years=num_years,
#             num_movie_tags=num_movie_tags_vocab,
#             embedding_dim=embedding_dim, # Should match shared_movie_embedding_layer's dim
#             year_embedding_dim=year_embedding_dim,
#             hidden_dim_mlp=hidden_dim_mlp_movie,
#             movie_embedding_layer_for_target=self.shared_movie_embedding_layer # Key change here
#         )

#     def forward(self,
#                 user_inputs: dict,
#                 positive_movie_inputs: dict,
#                 negative_movie_inputs: dict = None):
#         # Forward pass remains the same as previously defined
#         user_embedding = self.user_tower(
#             user_id=user_inputs['user_id'],
#             movie_history_ids=user_inputs['movie_history_ids']
#         )

#         positive_movie_embedding = self.movie_tower(
#             movie_id=positive_movie_inputs['movie_id'],
#             genre_indices=positive_movie_inputs['genre_indices'],
#             genre_offsets=positive_movie_inputs['genre_offsets'],
#             year_idx=positive_movie_inputs['year_idx'],
#             movie_tag_indices=positive_movie_inputs['movie_tag_indices'],
#             movie_tag_offsets=positive_movie_inputs['movie_tag_offsets']
#         )
#         positive_scores = torch.sum(user_embedding * positive_movie_embedding, dim=1)

#         if negative_movie_inputs is not None:
#             negative_movie_embedding = self.movie_tower(
#                 movie_id=negative_movie_inputs['movie_id'],
#                 genre_indices=negative_movie_inputs['genre_indices'],
#                 genre_offsets=negative_movie_inputs['genre_offsets'],
#                 year_idx=negative_movie_inputs['year_idx'],
#                 movie_tag_indices=negative_movie_inputs['movie_tag_indices'],
#                 movie_tag_offsets=negative_movie_inputs['movie_tag_offsets']
#             )
#             negative_scores = torch.sum(user_embedding * negative_movie_embedding, dim=1)
#             return positive_scores, negative_scores
#         else:
#             return positive_scores
        

# class TwoTowerTrainingDataset(Dataset):
#     def __init__(self,
#                  dataframe: pd.DataFrame,
#                  movie_history_len: int,
#                  # No need for separate feature lookups if dataframe has all merged features
#                  padding_idx_for_movies: int # The value you use for padding movie history
#                 ):
#         self.dataframe = dataframe
#         self.movie_history_len = movie_history_len
#         self.padding_idx_for_movies = padding_idx_for_movies

#         # Pre-convert columns to appropriate types if not already done,
#         # or ensure they are correct before passing the dataframe
#         # Example: self.dataframe['user_idx'] = self.dataframe['user_idx'].astype(np.int64)
#         # It's generally better to do this conversion once on the DataFrame itself.

#     def __len__(self):
#         return len(self.dataframe)

#     def _pad_sequence(self, sequence, max_len, pad_value):
#         # Ensure elements are integers before padding
#         sequence = [int(x) for x in sequence if pd.notna(x)] # Handle potential NaNs from original lists if any
#         if len(sequence) > max_len:
#             return torch.tensor(sequence[:max_len], dtype=torch.long)
#         else:
#             # Pad with the specified pad_value
#             return torch.tensor(sequence + [pad_value] * (max_len - len(sequence)), dtype=torch.long)

#     def __getitem__(self, idx):
#         row = self.dataframe.iloc[idx]

#         # --- User Inputs ---
#         # Ensure last_5_items_idx is a list of numbers; handle potential string format from JSON/CSV if needed
#         # Your preprocessing already creates padded lists of ints, using len(iid2idx) as pad_value
#         movie_hist_ids_list = row['last_5_items_idx'] # This should be a list of ints
        
#         user_inputs = {
#             'user_id': torch.tensor(int(row['user_idx']), dtype=torch.long),
#             'movie_history_ids': torch.tensor(movie_hist_ids_list, dtype=torch.long) # Already padded in data prep
#         }
#         # Note: If 'last_5_items_idx' is not already padded to movie_history_len,
#         # you'd use self._pad_sequence(movie_hist_ids_list, self.movie_history_len, self.padding_idx_for_movies)

#         # --- Positive Movie Inputs ---
#         # The DataFrame has lists of indices for genres/tags
#         positive_movie_inputs = {
#             'movie_id': torch.tensor(int(row['pos_item_idx']), dtype=torch.long),
#             'genre_indices_list': [int(g) for g in row['pos_item_genres_idx'] if pd.notna(g)], # List of genre indices
#             'year_idx': torch.tensor(int(row['pos_item_year_idx']), dtype=torch.long),
#             'movie_tag_indices_list': [int(t) for t in row['pos_item_tags_idx'] if pd.notna(t)] # List of tag indices
#         }

#         # --- Negative Movie Inputs ---
#         negative_movie_inputs = {
#             'movie_id': torch.tensor(int(row['neg_item_idx']), dtype=torch.long),
#             'genre_indices_list': [int(g) for g in row['neg_item_genres_idx'] if pd.notna(g)],
#             'year_idx': torch.tensor(int(row['neg_item_year_idx']), dtype=torch.long),
#             'movie_tag_indices_list': [int(t) for t in row['neg_item_tags_idx'] if pd.notna(t)]
#         }

#         return user_inputs, positive_movie_inputs, negative_movie_inputs
    

# def collate_towers(batch):
#     # batch is a list of tuples: [(user_inputs_dict_0, pos_movie_dict_0, neg_movie_dict_0), ...]

#     batched_user_inputs = {}
#     batched_positive_movie_inputs = {}
#     batched_negative_movie_inputs = {}

#     # --- Collate User Inputs ---
#     user_ids = []
#     movie_history_ids_list = []
#     for item in batch: # item is (user_dict, pos_dict, neg_dict)
#         user_inputs = item[0]
#         user_ids.append(user_inputs['user_id'])
#         movie_history_ids_list.append(user_inputs['movie_history_ids'])

#     batched_user_inputs['user_id'] = torch.stack(user_ids)
#     batched_user_inputs['movie_history_ids'] = torch.stack(movie_history_ids_list)


#     # --- Collate Movie Inputs (Positive and Negative) ---
#     # Helper function to collate movie features
#     def _collate_movie_features(list_of_movie_input_dicts):
#         collated = {}
#         movie_ids = []
#         genre_indices_lists = [] # List of lists (for EmbeddingBag)
#         year_idxs = []
#         movie_tag_indices_lists = [] # List of lists (for EmbeddingBag)

#         for movie_dict in list_of_movie_input_dicts:
#             movie_ids.append(movie_dict['movie_id'])
#             genre_indices_lists.append(torch.tensor(movie_dict['genre_indices_list'], dtype=torch.long)) # Keep as list of tensors for now
#             year_idxs.append(movie_dict['year_idx'])
#             movie_tag_indices_lists.append(torch.tensor(movie_dict['movie_tag_indices_list'], dtype=torch.long))

#         collated['movie_id'] = torch.stack(movie_ids)
#         collated['year_idx'] = torch.stack(year_idxs)

#         # Process for EmbeddingBag: concatenate all indices and create offsets
#         genre_offsets = [0]
#         all_genre_indices = []
#         for g_list in genre_indices_lists:
#             all_genre_indices.extend(g_list.tolist()) # Use .tolist() if they are tensors
#             genre_offsets.append(genre_offsets[-1] + len(g_list))
#         collated['genre_indices'] = torch.tensor(all_genre_indices, dtype=torch.long)
#         collated['genre_offsets'] = torch.tensor(genre_offsets[:-1], dtype=torch.long) # Remove last offset end-point

#         tag_offsets = [0]
#         all_tag_indices = []
#         for t_list in movie_tag_indices_lists:
#             all_tag_indices.extend(t_list.tolist())
#             tag_offsets.append(tag_offsets[-1] + len(t_list))
#         collated['movie_tag_indices'] = torch.tensor(all_tag_indices, dtype=torch.long)
#         collated['movie_tag_offsets'] = torch.tensor(tag_offsets[:-1], dtype=torch.long)
#         return collated

#     batched_positive_movie_inputs = _collate_movie_features([item[1] for item in batch])
#     batched_negative_movie_inputs = _collate_movie_features([item[2] for item in batch])

#     return batched_user_inputs, batched_positive_movie_inputs, batched_negative_movie_inputs
    
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

