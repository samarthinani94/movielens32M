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


