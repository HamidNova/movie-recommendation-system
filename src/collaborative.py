import logging
import os

import joblib
import numpy as np

from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from config import ALS_PARAMS, POPULAR_MIN_RATINGS


logger = logging.getLogger(__name__)


class CollaborativeRecommender:

    def __init__(self, alpha=2.0, use_gpu=False):
        if alpha < 0:
            raise ValueError("alpha must be greater than or equal to 0.")

        self.alpha = float(alpha)
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
        self.movie_lookup = {}

    def _build_confidence_matrix(self, df_ratings):
        """
        Build user-item confidence matrix for ALS.

        Explicit ratings are converted into implicit confidence values.
        """

        rows = (
            df_ratings["user_id"]
            .map(self.user_map)
            .to_numpy()
        )

        cols = (
            df_ratings["movie_id"]
            .map(self.item_map)
            .to_numpy()
        )

        ratings = df_ratings["rating"].to_numpy(
            dtype=np.float32
        )

        data = 1.0 + self.alpha * ratings

        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(
                len(self.user_map),
                len(self.item_map)
            ),
            dtype=np.float32
        )

        return matrix

    def _cache_popular_items(
        self,
        df_ratings,
        min_ratings=POPULAR_MIN_RATINGS,
    ):
        """
        Build Bayesian popularity fallback for cold-start users/items.
        """

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
                + global_average * min_ratings
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

    def _get_popular_recommendations(self, n=5):
        """
        Return popularity-based fallback recommendations.
        """

        if n <= 0:
            return []

        if (
            self._popular_cache is None
            or self._popular_cache.empty
        ):
            return []

        recommendations = []

        for movie_id in self._popular_cache.head(n).index:

            movie_id = int(movie_id)

            if movie_id not in self.movie_lookup:
                continue

            recommendations.append({
                "movie_id": movie_id,
                "title": str(
                    self.movie_lookup[movie_id]
                ),
                "predicted_rating": round(
                    float(
                        self._popular_cache[movie_id]
                    ),
                    3
                ),
            })

        return recommendations

    def build_model(self, df_ratings, df_movies):

        if df_ratings is None or df_ratings.empty:
            raise ValueError(
                "df_ratings is empty. "
                "Cannot train collaborative model."
            )

        if df_movies is None or df_movies.empty:
            raise ValueError(
                "df_movies is empty. "
                "Cannot build movie lookup."
            )

        required_rating_columns = {
            "user_id",
            "movie_id",
            "rating",
        }

        if not required_rating_columns.issubset(
            df_ratings.columns
        ):
            raise ValueError(
                "df_ratings must contain "
                "'user_id', 'movie_id', and 'rating' columns."
            )

        required_movie_columns = {
            "movie_id",
            "title",
        }

        if not required_movie_columns.issubset(
            df_movies.columns
        ):
            raise ValueError(
                "df_movies must contain "
                "'movie_id' and 'title' columns."
            )

        training_ratings = (
            df_ratings[
                ["user_id", "movie_id", "rating"]
            ]
            .dropna()
            .copy()
        )

        if training_ratings.empty:
            raise ValueError(
                "No valid ratings remain after removing missing values."
            )

        training_ratings["user_id"] = (
            training_ratings["user_id"].astype(int)
        )

        training_ratings["movie_id"] = (
            training_ratings["movie_id"].astype(int)
        )

        training_ratings["rating"] = (
            training_ratings["rating"].astype(np.float32)
        )

        if (
            not np.isfinite(
                training_ratings["rating"].to_numpy()
            ).all()
        ):
            raise ValueError(
                "df_ratings contains non-finite rating values."
            )

        self.user_map = {
            int(uid): idx
            for idx, uid in enumerate(
                training_ratings["user_id"].unique()
            )
        }

        self.item_map = {
            int(mid): idx
            for idx, mid in enumerate(
                training_ratings["movie_id"].unique()
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

        movie_data = (
            df_movies[
                ["movie_id", "title"]
            ]
            .dropna(subset=["movie_id"])
            .copy()
        )

        movie_data["movie_id"] = (
            movie_data["movie_id"].astype(int)
        )

        self.movie_lookup = (
            movie_data
            .drop_duplicates(
                subset="movie_id",
                keep="first"
            )
            .set_index("movie_id")["title"]
            .astype(str)
            .to_dict()
        )

        self.sparse_matrix = (
            self._build_confidence_matrix(
                training_ratings
            )
        )

        logger.info(
            "Training ALS model "
            "(users=%d, items=%d, interactions=%d)...",
            len(self.user_map),
            len(self.item_map),
            self.sparse_matrix.nnz,
        )

        self.model = AlternatingLeastSquares(
            factors=ALS_PARAMS["factors"],
            iterations=ALS_PARAMS["iterations"],
            regularization=ALS_PARAMS["regularization"],
            random_state=ALS_PARAMS["random_state"],
            use_gpu=self.use_gpu,
        )

        self.model.fit(self.sparse_matrix)

        self.user_factors = np.asarray(
            self.model.user_factors,
            dtype=np.float32
        ).copy()

        self.item_factors = np.asarray(
            self.model.item_factors,
            dtype=np.float32
        ).copy()

        self._cache_popular_items(
            training_ratings
        )

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

        return (
            self.user_factors,
            self.item_factors,
        )

    def recommend_for_user(
        self,
        user_id,
        df_ratings=None,
        df_movies=None,
        n=5
    ):

        if n <= 0:
            return []

        if self.model is None:
            raise RuntimeError(
                "Model has not been built."
            )

        user_id = int(user_id)

        # Cold-start user
        if user_id not in self.user_map:

            logger.info(
                "User %s not found. "
                "Returning popular recommendations.",
                user_id
            )

            return self._get_popular_recommendations(
                n=n
            )

        user_idx = self.user_map[user_id]

        ids, scores = self.model.recommend(
            userid=user_idx,
            user_items=self.sparse_matrix[user_idx],
            N=n,
            filter_already_liked_items=True,
        )

        recommendations = []

        for idx, score in zip(ids, scores):

            movie_id = self.item_reverse.get(
                int(idx)
            )

            if movie_id is None:
                continue

            title = self.movie_lookup.get(movie_id)

            if title is None:
                continue

            recommendations.append({
                "movie_id": int(movie_id),
                "title": str(title),
                "predicted_rating": round(
                    float(score),
                    4
                ),
            })

        return recommendations

    def recommend_similar_movies(
        self,
        movie_id,
        df_movies=None,
        n=5
    ):

        if n <= 0:
            return []

        if self.model is None:
            raise RuntimeError(
                "Model has not been built."
            )

        movie_id = int(movie_id)

        if movie_id not in self.item_map:

            logger.warning(
                "Movie %s not found. "
                "Returning popular movies.",
                movie_id,
            )

            return self._get_popular_recommendations(
                n=n
            )

        item_idx = self.item_map[movie_id]

        similar_ids, scores = self.model.similar_items(
            item_idx,
            N=n + 1,
        )

        recommendations = []

        for idx, score in zip(
            similar_ids,
            scores
        ):

            idx = int(idx)

            # Skip the movie itself
            if idx == item_idx:
                continue

            movie_id_result = self.item_reverse.get(
                idx
            )

            if movie_id_result is None:
                continue

            title = self.movie_lookup.get(
                movie_id_result
            )

            if title is None:
                continue

            recommendations.append({
                "movie_id": int(movie_id_result),
                "title": str(title),
                "similarity_score": round(
                    float(score),
                    4
                ),
            })

            if len(recommendations) >= n:
                break

        return recommendations

    def save(
        self,
        path="models/collaborative_model.pkl"
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

        logger.info(
            "Collaborative model saved to %s",
            path
        )

    @staticmethod
    def load(
        path="models/collaborative_model.pkl"
    ):

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Collaborative model not found: {path}"
            )

        return joblib.load(path)
