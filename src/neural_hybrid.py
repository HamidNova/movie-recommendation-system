import os
import random

import joblib
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset


class HybridNeuralNet(nn.Module):

    def __init__(
        self,
        user_embed_dim,
        item_embed_dim,
        fuzzy_dim,
        hidden_units=None
    ):
        super().__init__()

        if hidden_units is None:
            hidden_units = [128, 64]

        if user_embed_dim != item_embed_dim:
            raise ValueError(
                "user_embed_dim and item_embed_dim must be equal "
                "for element-wise interaction."
            )

        if fuzzy_dim < 0:
            raise ValueError(
                "fuzzy_dim cannot be negative."
            )

        self.user_norm = nn.LayerNorm(user_embed_dim)
        self.item_norm = nn.LayerNorm(item_embed_dim)

        self.fuzzy_norm = (
            nn.LayerNorm(fuzzy_dim)
            if fuzzy_dim > 0
            else None
        )

        input_dim = (
            user_embed_dim
            + item_embed_dim
            + user_embed_dim
            + fuzzy_dim
        )

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_units:

            layers.append(
                nn.Linear(prev_dim, hidden_dim)
            )

            layers.append(
                nn.LayerNorm(hidden_dim)
            )

            layers.append(
                nn.ReLU()
            )

            layers.append(
                nn.Dropout(0.1)
            )

            prev_dim = hidden_dim

        layers.append(
            nn.Linear(prev_dim, 1)
        )

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        user_emb,
        item_emb,
        fuzzy_feat
    ):

        user_norm = self.user_norm(user_emb)
        item_norm = self.item_norm(item_emb)

        interaction = (
            user_norm * item_norm
        )

        features = [
            user_norm,
            item_norm,
            interaction
        ]

        if self.fuzzy_norm is not None:

            if fuzzy_feat is None:
                raise ValueError(
                    "fuzzy_feat cannot be None when fuzzy_dim > 0."
                )

            fuzzy_norm = self.fuzzy_norm(
                fuzzy_feat
            )

            features.append(
                fuzzy_norm
            )

        x = torch.cat(
            features,
            dim=1
        )

        return self.net(x).squeeze(-1)


class NeuralHybridRecommender:

    def __init__(
        self,
        user_factors,
        item_factors,
        fuzzy_u,
        user_map,
        item_map,
        hidden_units=None,
        lr=0.0005,
        epochs=10,
        batch_size=128,
        device="cpu",
        random_state=42
    ):

        if hidden_units is None:
            hidden_units = [128, 64]

        self.user_factors = np.asarray(
            user_factors,
            dtype=np.float32
        )

        self.item_factors = np.asarray(
            item_factors,
            dtype=np.float32
        )

        self.fuzzy_u = np.asarray(
            fuzzy_u,
            dtype=np.float32
        )

        self.user_map = user_map
        self.item_map = item_map

        self.user_reverse = {
            value: key
            for key, value in user_map.items()
        }

        self.item_reverse = {
            value: key
            for key, value in item_map.items()
        }

        self.hidden_units = hidden_units
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self._validate_inputs()

        requested_device = torch.device(device)

        if (
            requested_device.type == "cuda"
            and torch.cuda.is_available()
        ):
            self.device = requested_device
        else:
            self.device = torch.device("cpu")

        self.model = None

        self._popular_cache = None
        self.train_user_items = {}

    def _validate_inputs(self):

        if self.user_factors.ndim != 2:
            raise ValueError(
                "user_factors must be a 2D array."
            )

        if self.item_factors.ndim != 2:
            raise ValueError(
                "item_factors must be a 2D array."
            )

        if self.fuzzy_u.ndim != 2:
            raise ValueError(
                "fuzzy_u must be a 2D array with one row per user."
            )

        if self.user_factors.shape[0] != len(self.user_map):
            raise ValueError(
                "The number of user_factors rows must match "
                "the number of users in user_map."
            )

        if self.item_factors.shape[0] != len(self.item_map):
            raise ValueError(
                "The number of item_factors rows must match "
                "the number of items in item_map."
            )

        if self.fuzzy_u.shape[0] != len(self.user_map):
            raise ValueError(
                "The number of fuzzy_u rows must match "
                "the number of users in user_map."
            )

        if self.user_factors.shape[1] != self.item_factors.shape[1]:
            raise ValueError(
                "user_factors and item_factors must have "
                "the same embedding dimension."
            )

        if self.fuzzy_u.shape[1] == 0:
            raise ValueError(
                "fuzzy_u must contain at least one fuzzy feature."
            )

        if not np.isfinite(self.user_factors).all():
            raise ValueError(
                "user_factors contain NaN or infinite values."
            )

        if not np.isfinite(self.item_factors).all():
            raise ValueError(
                "item_factors contain NaN or infinite values."
            )

        if not np.isfinite(self.fuzzy_u).all():
            raise ValueError(
                "fuzzy_u contains NaN or infinite values."
            )

    def _set_seed(self):

        random.seed(self.random_state)

        np.random.seed(
            self.random_state
        )

        torch.manual_seed(
            self.random_state
        )

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(
                self.random_state
            )

    def _cache_popular_items(
        self,
        df_ratings,
        min_ratings=10
    ):

        if df_ratings.empty:

            self._popular_cache = None

            return

        movie_stats = (
            df_ratings
            .groupby("movie_id")["rating"]
            .agg(["mean", "count"])
        )

        global_average = float(
            df_ratings["rating"].mean()
        )

        movie_stats["bayesian_avg"] = (
            (
                movie_stats["mean"]
                * movie_stats["count"]
                + global_average
                * min_ratings
            )
            /
            (
                movie_stats["count"]
                + min_ratings
            )
        )

        self._popular_cache = (
            movie_stats["bayesian_avg"]
            .sort_values(ascending=False)
        )

    def _build_dataset(
        self,
        df_ratings
    ):

        valid_mask = (
            df_ratings["user_id"]
            .isin(self.user_map)
            &
            df_ratings["movie_id"]
            .isin(self.item_map)
        )

        valid_ratings = (
            df_ratings
            .loc[valid_mask]
            .reset_index(drop=True)
        )

        if valid_ratings.empty:

            raise ValueError(
                "No valid ratings found for Neural Hybrid training."
            )

        user_indices = (
            valid_ratings["user_id"]
            .map(self.user_map)
            .astype(int)
            .to_numpy()
        )

        item_indices = (
            valid_ratings["movie_id"]
            .map(self.item_map)
            .astype(int)
            .to_numpy()
        )

        ratings = (
            valid_ratings["rating"]
            .astype(np.float32)
            .to_numpy()
        )

        user_embeddings = (
            self.user_factors[user_indices]
        )

        item_embeddings = (
            self.item_factors[item_indices]
        )

        # fuzzy_u belongs to users, so it must be indexed
        # using user_indices, not item_indices.
        fuzzy_features = (
            self.fuzzy_u[user_indices]
        )

        x_user = torch.tensor(
            user_embeddings,
            dtype=torch.float32
        )

        x_item = torch.tensor(
            item_embeddings,
            dtype=torch.float32
        )

        x_fuzzy = torch.tensor(
            fuzzy_features,
            dtype=torch.float32
        )

        y = torch.tensor(
            ratings,
            dtype=torch.float32
        )

        return TensorDataset(
            x_user,
            x_item,
            x_fuzzy,
            y
        )

    def fit(
        self,
        df_ratings,
        epochs=None,
        batch_size=None,
        lr=None
    ):

        self._set_seed()

        if df_ratings is None or df_ratings.empty:

            raise ValueError(
                "df_ratings cannot be empty."
            )

        required_columns = {
            "user_id",
            "movie_id",
            "rating"
        }

        if not required_columns.issubset(
            df_ratings.columns
        ):
            raise ValueError(
                "df_ratings must contain "
                "'user_id', 'movie_id', and 'rating' columns."
            )

        if epochs is None:
            epochs = self.epochs

        if batch_size is None:
            batch_size = self.batch_size

        if lr is None:
            lr = self.lr

        if epochs <= 0:
            raise ValueError(
                "epochs must be greater than 0."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        if lr <= 0:
            raise ValueError(
                "lr must be greater than 0."
            )

        self._cache_popular_items(
            df_ratings
        )

        self.train_user_items = (
            df_ratings
            .groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )

        dataset = self._build_dataset(
            df_ratings
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        user_embed_dim = (
            self.user_factors.shape[1]
        )

        item_embed_dim = (
            self.item_factors.shape[1]
        )

        fuzzy_dim = (
            self.fuzzy_u.shape[1]
        )

        self.model = HybridNeuralNet(
            user_embed_dim=user_embed_dim,
            item_embed_dim=item_embed_dim,
            fuzzy_dim=fuzzy_dim,
            hidden_units=self.hidden_units
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        criterion = nn.MSELoss(
            reduction="none"
        )

        self.model.train()

        for epoch in range(epochs):

            total_loss = 0.0
            total_samples = 0

            for (
                user_batch,
                item_batch,
                fuzzy_batch,
                rating_batch
            ) in loader:

                user_batch = (
                    user_batch.to(self.device)
                )

                item_batch = (
                    item_batch.to(self.device)
                )

                fuzzy_batch = (
                    fuzzy_batch.to(self.device)
                )

                rating_batch = (
                    rating_batch.to(self.device)
                )

                optimizer.zero_grad()

                predictions = self.model(
                    user_batch,
                    item_batch,
                    fuzzy_batch
                )

                loss = criterion(
                    predictions,
                    rating_batch
                ).mean()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()

                total_loss += (
                    loss.item()
                    * user_batch.size(0)
                )

                total_samples += (
                    user_batch.size(0)
                )

            average_loss = (
                total_loss
                / max(total_samples, 1)
            )

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Loss: {average_loss:.4f}"
            )

        return self

    def predict(
        self,
        user_id,
        movie_id
    ):

        if self.model is None:
            return None

        if (
            user_id not in self.user_map
            or movie_id not in self.item_map
        ):
            return None

        self.model.eval()

        user_index = self.user_map[
            user_id
        ]

        item_index = self.item_map[
            movie_id
        ]

        user_embedding = torch.tensor(
            self.user_factors[user_index],
            dtype=torch.float32
        ).unsqueeze(0).to(
            self.device
        )

        item_embedding = torch.tensor(
            self.item_factors[item_index],
            dtype=torch.float32
        ).unsqueeze(0).to(
            self.device
        )

        fuzzy_feature = torch.tensor(
            self.fuzzy_u[user_index],
            dtype=torch.float32
        ).unsqueeze(0).to(
            self.device
        )

        with torch.no_grad():

            prediction = self.model(
                user_embedding,
                item_embedding,
                fuzzy_feature
            ).item()

        prediction = float(
            np.clip(
                prediction,
                1.0,
                5.0
            )
        )

        return prediction

    def recommend_for_user(
        self,
        user_id,
        df_ratings=None,
        df_movies=None,
        n=5
    ):

        if df_movies is None:
            return []

        required_columns = {
            "movie_id",
            "title"
        }

        if not required_columns.issubset(
            df_movies.columns
        ):
            raise ValueError(
                "df_movies must contain "
                "'movie_id' and 'title' columns."
            )

        movie_lookup = (
            df_movies
            .set_index("movie_id")["title"]
            .to_dict()
        )

        watched_movies = set()

        if user_id in self.train_user_items:

            watched_movies = (
                self.train_user_items[user_id]
            )

        elif df_ratings is not None:

            if not {
                "user_id",
                "movie_id"
            }.issubset(df_ratings.columns):
                raise ValueError(
                    "df_ratings must contain "
                    "'user_id' and 'movie_id' columns."
                )

            watched_movies = set(
                df_ratings.loc[
                    df_ratings["user_id"] == user_id,
                    "movie_id"
                ]
            )

        # Cold-start / unknown user
        if (
            user_id not in self.user_map
            or self.model is None
        ):

            recommendations = []

            if self._popular_cache is None:
                return recommendations

            for movie_id in self._popular_cache.index:

                if movie_id in watched_movies:
                    continue

                recommendations.append({
                    "movie_id": int(movie_id),
                    "title": movie_lookup.get(
                        movie_id,
                        "Unknown"
                    ),
                    "predicted_rating": round(
                        float(
                            self._popular_cache[
                                movie_id
                            ]
                        ),
                        3
                    )
                })

                if len(recommendations) >= n:
                    break

            return recommendations

        user_index = self.user_map[
            user_id
        ]

        candidate_movie_ids = [
            movie_id
            for movie_id in self.item_map.keys()
            if movie_id not in watched_movies
        ]

        if not candidate_movie_ids:
            return []

        candidate_indices = [
            self.item_map[movie_id]
            for movie_id in candidate_movie_ids
        ]

        self.model.eval()

        with torch.no_grad():

            user_embedding = torch.tensor(
                self.user_factors[user_index],
                dtype=torch.float32
            ).unsqueeze(0).repeat(
                len(candidate_indices),
                1
            ).to(self.device)

            item_embeddings = torch.tensor(
                self.item_factors[
                    candidate_indices
                ],
                dtype=torch.float32
            ).to(self.device)

            # The user's fuzzy membership is the same for all
            # candidate movies.
            fuzzy_feature = torch.tensor(
                self.fuzzy_u[user_index],
                dtype=torch.float32
            ).unsqueeze(0).repeat(
                len(candidate_indices),
                1
            ).to(self.device)

            scores = self.model(
                user_embedding,
                item_embeddings,
                fuzzy_feature
            ).cpu().numpy()

        scores = np.clip(
            scores,
            1.0,
            5.0
        )

        top_indices = (
            np.argsort(scores)[::-1][:n]
        )

        recommendations = []

        for index in top_indices:

            movie_id = (
                candidate_movie_ids[index]
            )

            recommendations.append({
                "movie_id": int(movie_id),
                "title": movie_lookup.get(
                    movie_id,
                    "Unknown"
                ),
                "predicted_rating": round(
                    float(scores[index]),
                    3
                )
            })

        return recommendations

    def save(
        self,
        path="models/neural_hybrid.pkl"
    ):

        directory = os.path.dirname(path)

        if directory:

            os.makedirs(
                directory,
                exist_ok=True
            )

        joblib.dump(
            self,
            path
        )

    @staticmethod
    def load(
        path="models/neural_hybrid.pkl"
    ):

        return joblib.load(path)
