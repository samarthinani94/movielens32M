import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Define the Matrix Factorization model
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=100):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor([0.0]))

        # Initialize the embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_indices, item_indices):
        user_embeds = self.user_embedding(user_indices)
        item_embeds = self.item_embedding(item_indices)
        dot = (user_embeds * item_embeds).sum(1)
        bias = self.user_bias(user_indices).squeeze() + self.item_bias(item_indices).squeeze() + self.global_bias
        return dot + bias
    

class NeuralMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=100, hidden_layers=[64, 32, 16], dropout_p=0.3):
        super(NeuralMF, self).__init__()
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)

        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        mlp_input_size = embedding_dim * 2
        mlp_layers = []
        for hidden in hidden_layers:
            mlp_layers.append(nn.Linear(mlp_input_size, hidden))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=dropout_p))  
            mlp_input_size = hidden
        self.mlp = nn.Sequential(*mlp_layers)

        self.output_layer = nn.Linear(hidden_layers[-1] + embedding_dim, 1)

    def forward(self, user_indices, item_indices):
        # GMF part
        user_embed_gmf = self.user_embedding_gmf(user_indices)
        item_embed_gmf = self.item_embedding_gmf(item_indices)
        gmf = user_embed_gmf * item_embed_gmf

        # MLP part
        user_embed_mlp = self.user_embedding_mlp(user_indices)
        item_embed_mlp = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat([user_embed_mlp, item_embed_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Combine GMF + MLP parts
        concat = torch.cat([gmf, mlp_output], dim=-1)
        prediction = self.output_layer(concat)

        return prediction.squeeze()
    
# Dataset shared between MF and NeuralMF models
class RatingDataset(Dataset):
    def __init__(self, user_indices, item_indices, ratings):
        self.user_indices = user_indices
        self.item_indices = item_indices
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_indices[idx], self.item_indices[idx], self.ratings[idx]


