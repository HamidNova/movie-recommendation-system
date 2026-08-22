import logging
import joblib
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from config import (
    ALS_PARAMS,
    POPULAR_MIN_RATINGS,
)

logger = logging.getLogger(__name__)


class CollaborativeRecommender:

    def __init__(self, alpha=2.0, use_gpu=False):
        self.alpha = alpha
        self.use_gpu = use_gpu

        self.model = None

        self.user_map = {}
        self.item_map = {}

        self.user_reverse = {}
        self.item_reverse = {}

        self.user_factors = None
        self.item_factors = None

        self.sparse_matrix = None
        self._popular_cache = None
        self.movie_lookup = None

    def _build_confidence_matrix(self, df_ratings):
        rows = df_ratings["user_id"].map(self.user_map).values
        cols = df_ratings["movie_id"].map(self.item_map).values

        data = (
            1.0 + self.alpha * df_ratings["rating"].values
        ).astype(np.float32)

        return csr_matrix((data, (rows, cols)))

    def _cache_popular_items(
        self,
        df_ratings,
        min_ratings=POPULAR_MIN_RATINGS,
    ):
        movie_stats = (
            df_ratings
            .groupby("movie_id")["rating"]
            .agg(["mean", "count"])
        )

        global_average = df_ratings["rating"].mean()

        movie_stats["bayesian_avg"] = (
            (
                movie_stats["mean"] * movie_stats["count"]
                + global_average * min_ratings
            )
            / (movie_stats["count"] + min_ratings)
        )

        self._popular_cache = (
            movie_stats["bayesian_avg"]
            .sort_values(ascending=False)
        )

    def build_model(self, df_ratings, df_movies):

        self.user_map = {
            uid: idx
            for idx, uid in enumerate(
                df_ratings["user_id"].unique()
            )
        }

        self.item_map = {
            mid: idx
            for idx, mid in enumerate(
                df_ratings["movie_id"].unique()
            )
        }

        self.user_reverse = {
            idx: uid
            for uid, idx in self.user_map.items()
        }

        self.item_reverse = {
            idx: mid
            for mid, idx in self.item_map.items()
        }

        self.movie_lookup = (
            df_movies
            .set_index("movie_id")["title"]
            .to_dict()
        )

        self.sparse_matrix = self._build_confidence_matrix(df_ratings)

        self.model = AlternatingLeastSquares(
            factors=ALS_PARAMS["factors"],
            iterations=ALS_PARAMS["iterations"],
            regularization=ALS_PARAMS["regularization"],
            random_state=ALS_PARAMS["random_state"],
            use_gpu=self.use_gpu,
        )

        self.model.fit(self.sparse_matrix)

        self.user_factors = np.asarray(self.model.user_factors).copy()
        self.item_factors = np.asarray(self.model.item_factors).copy()

        self._cache_popular_items(df_ratings)

        logger.info(
            "ALS model trained successfully "
            "(users=%d, items=%d, factors=%d)",
            len(self.user_map),
            len(self.item_map),
            ALS_PARAMS["factors"],
        )

        return self

    def get_embeddings(self):
        if self.model is None:
            raise RuntimeError(
                "Model has not been built."
            )

        return self.user_factors, self.item_factors

    def recommend_for_user(self, user_id, df_ratings, df_movies=None, n=5):

        if self.model is None:
            raise RuntimeError("Model has not been built.")

        if user_id not in self.user_map:
            recommendations = []

            for movie_id in self._popular_cache.head(n).index:
                recommendations.append({
                    "movie_id": int(movie_id),
                    "title": self.movie_lookup.get(movie_id, "Unknown"),
                    "predicted_rating": round(
                        float(self._popular_cache[movie_id]), 3
                    ),
                })

            return recommendations

        user_idx = self.user_map[user_id]

        ids, scores = self.model.recommend(
            userid=user_idx,
            user_items=self.sparse_matrix[user_idx],
            N=n,
            filter_already_liked_items=True,
        )

        recommendations = []

        for idx, score in zip(ids, scores):
            movie_id = self.item_reverse[idx]

            recommendations.append({
                "movie_id": int(movie_id),
                "title": self.movie_lookup.get(movie_id, "Unknown"),
                "predicted_rating": round(float(score), 3),
            })

        return recommendations

    def recommend_similar_movies(self, movie_id, df_movies=None, n=5):

        if self.model is None:
            raise RuntimeError("Model has not been built.")

        if movie_id not in self.item_map:
            logger.warning(
                "Movie %s not found. Returning popular movies.",
                movie_id,
            )

            recommendations = []

            for mid in self._popular_cache.head(n).index:
                recommendations.append({
                    "movie_id": int(mid),
                    "title": self.movie_lookup.get(mid, "Unknown"),
                    "similarity_score": round(
                        float(self._popular_cache[mid]),
                        3,
                    ),
                })

            return recommendations

        item_idx = self.item_map[movie_id]

        similar_ids, scores = self.model.similar_items(
            item_idx,
            N=n + 1,
        )

        recommendations = []

        for idx, score in zip(similar_ids, scores):

            if idx == item_idx:
                continue

            mid = self.item_reverse[idx]

            recommendations.append({
                "movie_id": int(mid),
                "title": self.movie_lookup.get(mid, "Unknown"),
                "similarity_score": round(float(score), 4),
            })

            if len(recommendations) >= n:
                break

        return recommendations

    def save(self, path="models/collaborative_model.pkl"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="models/collaborative_model.pkl"):
        return joblib.load(path)
