import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import joblib

class HybridNeuralNet(nn.Module):
    """
    Neural network that takes user embedding, item embedding, and fuzzy membership as input
    and predicts rating.
    """
    def __init__(self, user_embed_dim, item_embed_dim, fuzzy_dim, hidden_units=[128, 64]):
        super().__init__()
        input_dim = user_embed_dim + item_embed_dim + fuzzy_dim
        layers = []
        prev_dim = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # output rating
        self.net = nn.Sequential(*layers)

    def forward(self, user_emb, item_emb, fuzzy_feat):
        x = torch.cat([user_emb, item_emb, fuzzy_feat], dim=1)
        return self.net(x).squeeze()

class NeuralHybridRecommender:
    """
    Recommender that uses ALS embeddings + Fuzzy C-Means membership + neural network.
    """
    def __init__(self, user_factors, item_factors, fuzzy_u, user_map, item_map,
                 hidden_units=[128, 64], lr=0.001, epochs=10, batch_size=64, device='cpu'):
        """
        user_factors: np.array (n_users, embedding_dim)
        item_factors: np.array (n_items, embedding_dim)
        fuzzy_u: np.array (n_items, n_clusters)   # fuzzy membership for items
        user_map: dict user_id -> index in user_factors
        item_map: dict item_id -> index in item_factors
        """
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.fuzzy_u = fuzzy_u
        self.user_map = user_map
        self.item_map = item_map
        self.user_reverse = {v: k for k, v in user_map.items()}
        self.item_reverse = {v: k for k, v in item_map.items()}

        self.model = None
        self.hidden_units = hidden_units
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def _build_dataset(self, df_ratings):
        """Create Torch dataset from ratings dataframe using embeddings and fuzzy features."""
        user_ids = df_ratings['user_id'].map(self.user_map).values
        item_ids = df_ratings['movie_id'].map(self.item_map).values
        ratings = df_ratings['rating'].values.astype(np.float32)

        user_emb = self.user_factors[user_ids]
        item_emb = self.item_factors[item_ids]
        fuzzy_feat = self.fuzzy_u[item_ids]   # item fuzzy membership

        X_user = torch.tensor(user_emb, dtype=torch.float32)
        X_item = torch.tensor(item_emb, dtype=torch.float32)
        X_fuzzy = torch.tensor(fuzzy_feat, dtype=torch.float32)
        y = torch.tensor(ratings, dtype=torch.float32)

        return TensorDataset(X_user, X_item, X_fuzzy, y)

    def fit(self, df_ratings, epochs=None, batch_size=None, lr=None):
        if epochs is None: epochs = self.epochs
        if batch_size is None: batch_size = self.batch_size
        if lr is None: lr = self.lr

        dataset = self._build_dataset(df_ratings)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        embed_dim_user = self.user_factors.shape[1]
        embed_dim_item = self.item_factors.shape[1]
        fuzzy_dim = self.fuzzy_u.shape[1]
        self.model = HybridNeuralNet(embed_dim_user, embed_dim_item, fuzzy_dim,
                                     hidden_units=self.hidden_units).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                u, i, f, r = [x.to(self.device) for x in batch]
                optimizer.zero_grad()
                pred = self.model(u, i, f)
                loss = criterion(pred, r)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * u.size(0)
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        return self

    def predict(self, user_id, movie_id):
        """Predict rating for a single user-movie pair."""
        self.model.eval()
        if user_id not in self.user_map or movie_id not in self.item_map:
            return None
        u_idx = self.user_map[user_id]
        i_idx = self.item_map[movie_id]
        user_emb = torch.tensor(self.user_factors[u_idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        item_emb = torch.tensor(self.item_factors[i_idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        fuzzy = torch.tensor(self.fuzzy_u[i_idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(user_emb, item_emb, fuzzy).item()
        return pred

    def recommend_for_user(self, user_id, df_ratings, df_movies, n=5):
        """Generate top-n recommendations for a user (excluding already rated movies)."""
        if user_id not in self.user_map:
            return []  # cold start not handled here; external handling recommended
        rated_items = df_ratings[df_ratings['user_id'] == user_id]['movie_id'].tolist()
        all_items = list(self.item_map.keys())
        candidates = [mid for mid in all_items if mid not in rated_items]

        preds = []
        for mid in candidates:
            r = self.predict(user_id, mid)
            if r is not None:
                preds.append((mid, r))

        preds.sort(key=lambda x: x[1], reverse=True)
        top_n = preds[:n]
        recommendations = []
        for mid, score in top_n:
            title = df_movies[df_movies['movie_id'] == mid]['title'].values[0]
            recommendations.append({
                'title': title,
                'predicted_rating': round(score, 2)
            })
        return recommendations

    def save(self, path='models/neural_hybrid.pkl'):
        joblib.dump(self, path)

    @staticmethod
    def load(path='models/neural_hybrid.pkl'):
        return joblib.load(path)
