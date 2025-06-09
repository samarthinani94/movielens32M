import torch
import torch.nn as nn
from torch.utils.data import Dataset


# --- 2. MODEL AND DATASET DEFINITION ---
class NeuralMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=100, hidden_layers=[64, 32, 16], dropout_p=0.3):
        super(NeuralMF, self).__init__()
        # GMF embeddings
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)

        # MLP embeddings
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        mlp_input_size = embedding_dim * 2
        mlp_layers = []
        for hidden in hidden_layers:
            mlp_layers.append(nn.Linear(mlp_input_size, hidden))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=dropout_p))
            mlp_input_size = hidden
        self.mlp = nn.Sequential(*mlp_layers)

        # Final output projection
        self.output_layer = nn.Linear(hidden_layers[-1] + embedding_dim, 1)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

    def forward(self, user_indices, pos_item_indices, neg_item_indices):
        # GMF part
        user_embed_gmf = self.user_embedding_gmf(user_indices)
        pos_item_embed_gmf = self.item_embedding_gmf(pos_item_indices)
        neg_item_embed_gmf = self.item_embedding_gmf(neg_item_indices)
        gmf_pos = user_embed_gmf * pos_item_embed_gmf
        gmf_neg = user_embed_gmf * neg_item_embed_gmf

        # MLP part
        user_embed_mlp = self.user_embedding_mlp(user_indices)
        pos_item_embed_mlp = self.item_embedding_mlp(pos_item_indices)
        neg_item_embed_mlp = self.item_embedding_mlp(neg_item_indices)
        mlp_input_pos = torch.cat([user_embed_mlp, pos_item_embed_mlp], dim=-1)
        mlp_input_neg = torch.cat([user_embed_mlp, neg_item_embed_mlp], dim=-1)
        mlp_output_pos = self.mlp(mlp_input_pos)
        mlp_output_neg = self.mlp(mlp_input_neg)

        # Combine GMF and MLP outputs
        final_pos = torch.cat([gmf_pos, mlp_output_pos], dim=-1)
        final_neg = torch.cat([gmf_neg, mlp_output_neg], dim=-1)

        pos_score = self.output_layer(final_pos).squeeze()
        neg_score = self.output_layer(final_neg).squeeze()

        return pos_score, neg_score

    def predict(self, user_indices, item_indices):
        # A separate method for inference to score a user against many items
        user_embed_gmf = self.user_embedding_gmf(user_indices)
        item_embed_gmf = self.item_embedding_gmf(item_indices)
        gmf = user_embed_gmf * item_embed_gmf

        user_embed_mlp = self.user_embedding_mlp(user_indices)
        item_embed_mlp = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat([user_embed_mlp, item_embed_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)

        final_input = torch.cat([gmf, mlp_output], dim=-1)
        score = self.output_layer(final_input).squeeze()
        return score

class NMFDataset(Dataset):
    def __init__(self, user_indices, pos_item_indices, neg_item_indices):
        self.user_indices = torch.tensor(user_indices, dtype=torch.long)
        self.pos_item_indices = torch.tensor(pos_item_indices, dtype=torch.long)
        self.neg_item_indices = torch.tensor(neg_item_indices, dtype=torch.long)

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        return self.user_indices[idx], self.pos_item_indices[idx], self.neg_item_indices[idx]
