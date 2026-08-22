import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import joblib


class HybridNeuralNet(nn.Module):
    def __init__(self, user_embed_dim, item_embed_dim, fuzzy_dim, hidden_units=[128, 64]):
        super().__init__()
        self.user_norm = nn.LayerNorm(user_embed_dim)
        self.item_norm = nn.LayerNorm(item_embed_dim)
        if fuzzy_dim > 0:
            self.fuzzy_norm = nn.LayerNorm(fuzzy_dim)
        else:
            self.fuzzy_norm = None

        input_dim = user_embed_dim + item_embed_dim + (user_embed_dim) + fuzzy_dim
        layers = []
        prev_dim = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, user_emb, item_emb, fuzzy_feat):
        u_norm = self.user_norm(user_emb)
        i_norm = self.item_norm(item_emb)
        interaction = u_norm * i_norm

        if self.fuzzy_norm is not None and fuzzy_feat.size(1) > 0:
            f_norm = self.fuzzy_norm(fuzzy_feat)
            x = torch.cat([u_norm, i_norm, interaction, f_norm], dim=1)
        else:
            x = torch.cat([u_norm, i_norm, interaction], dim=1)

        return self.net(x).squeeze(-1)


class NeuralHybridRecommender:
    def __init__(self, user_factors, item_factors, fuzzy_u, user_map, item_map,
                 hidden_units=[128, 64], lr=0.0005, epochs=10, batch_size=128, device='cpu'):
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

        self._popular_cache = None
        self.train_user_items = {}
        self.user_fuzzy_profiles = {}

    def _cache_popular_items(self, df_ratings, min_ratings=10):
        movie_stats = df_ratings.groupby("movie_id")["rating"].agg(["mean", "count"])
        global_average = df_ratings["rating"].mean()
        movie_stats["bayesian_avg"] = (
            (movie_stats["mean"] * movie_stats["count"] + global_average * min_ratings)
            / (movie_stats["count"] + min_ratings)
        )
        self._popular_cache = movie_stats["bayesian_avg"].sort_values(ascending=False)

    def _build_user_fuzzy_profiles(self, df_ratings):
        if self.fuzzy_u is None or not np.any(self.fuzzy_u):
            return
        for user_id, group in df_ratings.groupby('user_id'):
            if user_id not in self.user_map:
                continue
            liked = group[group['rating'] >= 3.5]
            if liked.empty:
                liked = group
            idxs = [self.item_map[m] for m in liked['movie_id'] if m in self.item_map]
            if idxs:
                prof = np.mean(self.fuzzy_u[idxs], axis=0)
                prof = prof / (np.linalg.norm(prof) + 1e-8)
                self.user_fuzzy_profiles[user_id] = prof

    def _build_dataset(self, df_ratings):
        user_ids = df_ratings['user_id'].map(self.user_map).values
        item_ids = df_ratings['movie_id'].map(self.item_map).values
        ratings = df_ratings['rating'].values.astype(np.float32)

        user_emb = self.user_factors[user_ids]
        item_emb = self.item_factors[item_ids]
        fuzzy_feat = self.fuzzy_u[item_ids]

        X_user = torch.tensor(user_emb, dtype=torch.float32)
        X_item = torch.tensor(item_emb, dtype=torch.float32)
        X_fuzzy = torch.tensor(fuzzy_feat, dtype=torch.float32)
        y = torch.tensor(ratings, dtype=torch.float32)

        return TensorDataset(X_user, X_item, X_fuzzy, y)

    def _compute_sample_weights(self, df_ratings):
        n = len(df_ratings)
        weights = np.ones(n, dtype=np.float32)
        for pos, (_, row) in enumerate(df_ratings.iterrows()):
            user_id = row['user_id']
            movie_id = row['movie_id']
            if user_id in self.user_fuzzy_profiles and movie_id in self.item_map:
                item_idx = self.item_map[movie_id]
                item_fuzzy = self.fuzzy_u[item_idx]
                item_norm = item_fuzzy / (np.linalg.norm(item_fuzzy) + 1e-8)
                sim = float(np.dot(self.user_fuzzy_profiles[user_id], item_norm))
                weights[pos] = 1.0 + 1.5 * sim  # slightly lower weight to avoid instability
            else:
                weights[pos] = 1.0
        return torch.tensor(weights, dtype=torch.float32)

    def fit(self, df_ratings, epochs=None, batch_size=None, lr=None):
        if epochs is None: epochs = self.epochs
        if batch_size is None: batch_size = self.batch_size
        if lr is None: lr = self.lr

        self._cache_popular_items(df_ratings)
        self.train_user_items = df_ratings.groupby('user_id')['movie_id'].apply(set).to_dict()
        self._build_user_fuzzy_profiles(df_ratings)

        dataset = self._build_dataset(df_ratings)
        sample_weights = self._compute_sample_weights(df_ratings)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        embed_dim_user = self.user_factors.shape[1]
        embed_dim_item = self.item_factors.shape[1]
        fuzzy_dim = self.fuzzy_u.shape[1]
        self.model = HybridNeuralNet(embed_dim_user, embed_dim_item, fuzzy_dim,
                                     hidden_units=self.hidden_units).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss(reduction='none')

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, (u, i, f, r) in enumerate(loader):
                u, i, f, r = u.to(self.device), i.to(self.device), f.to(self.device), r.to(self.device)
                start = batch_idx * batch_size
                end = start + u.size(0)
                w = sample_weights[start:end].to(self.device)

                optimizer.zero_grad()
                pred = self.model(u, i, f)
                loss = (criterion(pred, r) * w).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * u.size(0)
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        return self

    def predict(self, user_id, movie_id):
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
        movie_lookup = df_movies.set_index('movie_id')['title'].to_dict() if df_movies is not None else {}

        if user_id not in self.user_map or self.model is None:
            recommendations = []
            if self._popular_cache is not None:
                for mid in self._popular_cache.head(n).index:
                    recommendations.append({
                        "movie_id": int(mid),
                        "title": movie_lookup.get(mid, "Unknown"),
                        "predicted_rating": round(float(self._popular_cache[mid]), 3),
                    })
            return recommendations

        u_idx = self.user_map[user_id]
        train_rated = self.train_user_items.get(user_id, set())

        all_item_mids = list(self.item_map.keys())
        candidates_mids = [m for m in all_item_mids if m not in train_rated]

        if not candidates_mids:
            return []

        cand_idxs = [self.item_map[m] for m in candidates_mids]

        self.model.eval()
        with torch.no_grad():
            u_emb = torch.tensor(self.user_factors[u_idx], dtype=torch.float32).unsqueeze(0).repeat(len(cand_idxs), 1).to(self.device)
            i_emb = torch.tensor(self.item_factors[cand_idxs], dtype=torch.float32).to(self.device)
            f_feat = torch.tensor(self.fuzzy_u[cand_idxs], dtype=torch.float32).to(self.device)

            scores = self.model(u_emb, i_emb, f_feat).cpu().numpy()

        top_indices = np.argsort(scores)[::-1][:n]

        recommendations = []
        for idx in top_indices:
            mid = candidates_mids[idx]
            score = scores[idx]
            recommendations.append({
                "movie_id": int(mid),
                "title": movie_lookup.get(mid, "Unknown"),
                "predicted_rating": round(float(score), 3)
            })

        return recommendations

    def save(self, path='models/neural_hybrid.pkl'):
        joblib.dump(self, path)

    @staticmethod
    def load(path='models/neural_hybrid.pkl'):
        return joblib.load(path)
