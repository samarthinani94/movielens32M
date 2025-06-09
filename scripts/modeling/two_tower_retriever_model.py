import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd

class MovieTower(nn.Module):
    """Encodes movie features into a dense embedding."""
    def __init__(self,
                 num_genres: int,
                 num_years: int,
                 num_movie_tags: int,
                 embedding_dim: int,
                 year_embedding_dim: int,
                 hidden_dim_mlp: int,
                 movie_embedding_layer: nn.Embedding):
        super().__init__()
        self.movie_embedding = movie_embedding_layer
        self.embedding_dim = self.movie_embedding.embedding_dim

        self.genre_embedding = nn.Embedding(num_genres, self.embedding_dim, padding_idx=0)
        self.movie_tag_embedding = nn.Embedding(num_movie_tags, self.embedding_dim, padding_idx=0)
        self.year_embedding = nn.Embedding(num_years, year_embedding_dim, padding_idx=0)

        combined_feature_dim = self.embedding_dim + self.embedding_dim + year_embedding_dim + self.embedding_dim
        self.fc1 = nn.Linear(combined_feature_dim, hidden_dim_mlp)
        self.fc2 = nn.Linear(hidden_dim_mlp, hidden_dim_mlp // 2)
        self.fc3 = nn.Linear(hidden_dim_mlp // 2, self.embedding_dim)

    def forward(self, movie_id, padded_genre_indices, year_idx, padded_tag_indices):
        m_embed = self.movie_embedding(movie_id)
        y_embed = self.year_embedding(year_idx)

        # Masked average for genres (works on MPS)
        genre_embeds = self.genre_embedding(padded_genre_indices)
        genre_mask = (padded_genre_indices != 0).float().unsqueeze(-1)
        g_embed_aggregated = (genre_embeds * genre_mask).sum(dim=1) / genre_mask.sum(dim=1).clamp(min=1e-9)

        # Masked average for tags (works on MPS)
        tag_embeds = self.movie_tag_embedding(padded_tag_indices)
        tag_mask = (padded_tag_indices != 0).float().unsqueeze(-1)
        mt_embed_aggregated = (tag_embeds * tag_mask).sum(dim=1) / tag_mask.sum(dim=1).clamp(min=1e-9)

        combined = torch.cat([m_embed, g_embed_aggregated, y_embed, mt_embed_aggregated], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class UserTower(nn.Module):
    """Encodes user features into a dense embedding."""
    def __init__(self,
                 num_users: int,
                 embedding_dim: int,
                 hidden_dim_lstm: int,
                 hidden_dim_mlp: int,
                 movie_embedding_layer: nn.Embedding):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.movie_history_embedding = movie_embedding_layer
        self.movie_history_lstm = nn.LSTM(embedding_dim, hidden_dim_lstm, batch_first=True)

        combined_dim = embedding_dim + hidden_dim_lstm
        self.fc1 = nn.Linear(combined_dim, hidden_dim_mlp)
        self.fc2 = nn.Linear(hidden_dim_mlp, hidden_dim_mlp // 2)
        self.fc3 = nn.Linear(hidden_dim_mlp // 2, embedding_dim)

    def forward(self, user_id, movie_history_ids):
        u_embed = self.user_embedding(user_id)
        mh_embeds = self.movie_history_embedding(movie_history_ids)
        _, (mh_lstm_final_hidden, _) = self.movie_history_lstm(mh_embeds)
        mh_representation = mh_lstm_final_hidden.squeeze(0)
        
        combined = torch.cat([u_embed, mh_representation], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class TwoTowerSystem(nn.Module):
    """The complete two-tower model."""
    def __init__(self, config: dict):
        super().__init__()
        padding_idx = config['num_movies']
        self.shared_movie_embedding = nn.Embedding(
            num_embeddings=config['num_movies'] + 1,
            embedding_dim=config['embedding_dim'],
            padding_idx=padding_idx
        )

        self.user_tower = UserTower(
            num_users=config['num_users'],
            embedding_dim=config['embedding_dim'],
            hidden_dim_lstm=config['hidden_dim_lstm'],
            hidden_dim_mlp=config['hidden_dim_mlp_user'],
            movie_embedding_layer=self.shared_movie_embedding
        )
        self.movie_tower = MovieTower(
            num_genres=config['num_genres'],
            num_years=config['num_years'],
            num_movie_tags=config['num_movie_tags_vocab'],
            embedding_dim=config['embedding_dim'],
            year_embedding_dim=config['year_embedding_dim'],
            hidden_dim_mlp=config['hidden_dim_mlp_movie'],
            movie_embedding_layer=self.shared_movie_embedding
        )

    def forward(self, user_inputs, pos_movie_inputs, neg_movie_inputs):
        user_embedding = self.user_tower(**user_inputs)
        pos_movie_embedding = self.movie_tower(**pos_movie_inputs)
        neg_movie_embedding = self.movie_tower(**neg_movie_inputs)
        
        pos_scores = (user_embedding * pos_movie_embedding).sum(dim=1)
        neg_scores = (user_embedding * neg_movie_embedding).sum(dim=1)
        
        return pos_scores, neg_scores

class TwoTowerDataset(Dataset):
    """Dataset for training the two-tower model."""
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_inputs = {
            'user_id': torch.tensor(row['user_idx'], dtype=torch.long),
            'movie_history_ids': torch.tensor(row['last_5_items_idx'], dtype=torch.long)
        }
        pos_movie_inputs = {
            'movie_id': torch.tensor(row['pos_item_idx'], dtype=torch.long),
            'padded_genre_indices': torch.tensor(row['pos_item_genres_idx'], dtype=torch.long),
            'year_idx': torch.tensor(row['pos_item_year_idx'], dtype=torch.long),
            'padded_tag_indices': torch.tensor(row['pos_item_tags_idx'], dtype=torch.long)
        }
        neg_movie_inputs = {
            'movie_id': torch.tensor(row['neg_item_idx'], dtype=torch.long),
            'padded_genre_indices': torch.tensor(row['neg_item_genres_idx'], dtype=torch.long),
            'year_idx': torch.tensor(row['neg_item_year_idx'], dtype=torch.long),
            'padded_tag_indices': torch.tensor(row['neg_item_tags_idx'], dtype=torch.long)
        }
        return user_inputs, pos_movie_inputs, neg_movie_inputs
