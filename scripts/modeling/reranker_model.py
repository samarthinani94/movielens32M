import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ReRankerNN(nn.Module):
    """
    A re-ranker model that scores user-item pairs based on a combination of
    frozen retriever embeddings and its own learnable embeddings.
    """
    def __init__(self, config: dict, retriever_movie_embedding_layer: nn.Embedding):
        super().__init__()

        # --- Frozen Retriever Embeddings ---
        # Use the pre-trained embeddings from the retriever as a feature
        self.retriever_movie_embedding_layer = retriever_movie_embedding_layer
        self.retriever_movie_embedding_layer.weight.requires_grad = False
        retriever_movie_emb_dim = self.retriever_movie_embedding_layer.embedding_dim

        # --- Re-ranker's Own Learnable Embedding Layers ---
        self.user_embedding = nn.Embedding(config['num_users'], config['reranker_embedding_dim'])
        self.item_genre_embedding = nn.Embedding(config['num_genres'], config['reranker_embedding_dim'], padding_idx=0)
        self.item_year_embedding = nn.Embedding(config['num_years'], config['reranker_year_embedding_dim'], padding_idx=0)
        self.item_tag_embedding = nn.Embedding(config['num_movie_tags_vocab'], config['reranker_embedding_dim'], padding_idx=0)

        # --- Final Interaction MLP ---
        # This MLP learns interactions from a rich feature vector.
        combined_feature_dim = (
            config['reranker_embedding_dim'] +       # Ranker's user embed
            retriever_movie_emb_dim +                # Retriever's movie embed
            config['reranker_embedding_dim'] +       # Ranker's genre embed
            config['reranker_year_embedding_dim'] +  # Ranker's year embed
            config['reranker_embedding_dim']         # Ranker's tag embed
        )

        self.interaction_mlp = nn.Sequential(
            nn.Linear(combined_feature_dim, config['reranker_hidden_dim_mlp']),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config['reranker_hidden_dim_mlp'], config['reranker_hidden_dim_mlp'] // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config['reranker_hidden_dim_mlp'] // 2, 1) # Final single score output
        )

    def _get_item_features(self, movie_id, padded_genre_indices, year_idx, padded_tag_indices):
        """Helper function to compute all features for a given item."""
        retrieved_embed = self.retriever_movie_embedding_layer(movie_id)

        genre_embeds = self.item_genre_embedding(padded_genre_indices)
        genre_mask = (padded_genre_indices != 0).float().unsqueeze(-1)
        genre_agg_embed = (genre_embeds * genre_mask).sum(dim=1) / genre_mask.sum(dim=1).clamp(min=1e-9)
        
        year_embed = self.item_year_embedding(year_idx)

        tag_embeds = self.item_tag_embedding(padded_tag_indices)
        tag_mask = (padded_tag_indices != 0).float().unsqueeze(-1)
        tag_agg_embed = (tag_embeds * tag_mask).sum(dim=1) / tag_mask.sum(dim=1).clamp(min=1e-9)
        
        return retrieved_embed, genre_agg_embed, year_embed, tag_agg_embed

    def forward(self, user_inputs, positive_item_inputs, negative_item_inputs):
        """Forward pass for training with positive and negative items."""
        user_representation = self.user_embedding(user_inputs['user_id'])

        # --- Positive Item Path ---
        pos_features = self._get_item_features(**positive_item_inputs)
        positive_interaction_input = torch.cat([user_representation] + list(pos_features), dim=1)
        positive_score = self.interaction_mlp(positive_interaction_input).squeeze(-1)

        # --- Negative Item Path ---
        neg_features = self._get_item_features(**negative_item_inputs)
        negative_interaction_input = torch.cat([user_representation] + list(neg_features), dim=1)
        negative_score = self.interaction_mlp(negative_interaction_input).squeeze(-1)
        
        return positive_score, negative_score
    
    def predict(self, user_inputs, item_inputs):
        """Forward pass for inference, scoring a single user-item pair."""
        user_representation = self.user_embedding(user_inputs['user_id'])
        item_features = self._get_item_features(**item_inputs)
        interaction_input = torch.cat([user_representation] + list(item_features), dim=1)
        score = self.interaction_mlp(interaction_input).squeeze(-1)
        return score


class ReRankerDataset(Dataset):
    """Dataset for training the ReRankerNN."""
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_inputs = {
            'user_id': torch.tensor(row['user_idx'], dtype=torch.long),
        }
        positive_item_inputs = {
            'movie_id': torch.tensor(row['pos_item_idx'], dtype=torch.long),
            'padded_genre_indices': torch.tensor(row['pos_item_genres_idx'], dtype=torch.long),
            'year_idx': torch.tensor(row['pos_item_year_idx'], dtype=torch.long),
            'padded_tag_indices': torch.tensor(row['pos_item_tags_idx'], dtype=torch.long)
        }
        negative_item_inputs = {
            'movie_id': torch.tensor(row['neg_item_idx'], dtype=torch.long),
            'padded_genre_indices': torch.tensor(row['neg_item_genres_idx'], dtype=torch.long),
            'year_idx': torch.tensor(row['neg_item_year_idx'], dtype=torch.long),
            'padded_tag_indices': torch.tensor(row['neg_item_tags_idx'], dtype=torch.long)
        }
        return user_inputs, positive_item_inputs, negative_item_inputs
