import numpy as np
import pandas as pd


class PopularRecommender:

    def __init__(self, min_ratings=10):
        self.min_ratings = min_ratings
        self.popularity_scores = None
        self.user_rated_items = {}

    def fit(self, df_ratings):
        required_columns = {
            "user_id",
            "movie_id",
            "rating",
        }

        missing_columns = (
            required_columns
            - set(df_ratings.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing required rating columns: "
                f"{missing_columns}"
            )

        if df_ratings.empty:
            self.popularity_scores = pd.Series(
                dtype=float
            )
            self.user_rated_items = {}

            return self

        movie_stats = (
            df_ratings
            .groupby("movie_id")["rating"]
            .agg(["mean", "count"])
        )

        global_avg = float(
            df_ratings["rating"].mean()
        )

        movie_stats["bayesian_avg"] = (
            (
                movie_stats["mean"]
                * movie_stats["count"]
                + global_avg * self.min_ratings
            )
            /
            (
                movie_stats["count"]
                + self.min_ratings
            )
        )

        self.popularity_scores = (
            movie_stats["bayesian_avg"]
            .sort_values(ascending=False)
        )

        self.user_rated_items = (
            df_ratings
            .groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )

        return self

    def recommend(self, user_id, df_movies, n=5):
        if (
            self.popularity_scores is None
            or self.popularity_scores.empty
            or n <= 0
        ):
            return []

        required_columns = {
            "movie_id",
            "title",
        }

        missing_columns = (
            required_columns
            - set(df_movies.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing required movie columns: "
                f"{missing_columns}"
            )

        movie_lookup = (
            df_movies
            .set_index("movie_id")["title"]
            .to_dict()
        )

        rated_movie_ids = (
            self.user_rated_items
            .get(user_id, set())
        )

        recommendations = []

        for movie_id, score in (
            self.popularity_scores.items()
        ):
            if movie_id in rated_movie_ids:
                continue

            title = movie_lookup.get(movie_id)

            if title is None:
                continue

            recommendations.append({
                "movie_id": int(movie_id),
                "title": str(title),
                "predicted_rating": float(score),
            })

            if len(recommendations) >= n:
                break

        return recommendations


class RandomRecommender:

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.movie_ids = None

    def fit(self, df_movies):
        required_columns = {
            "movie_id",
            "title",
        }

        missing_columns = (
            required_columns
            - set(df_movies.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing required movie columns: "
                f"{missing_columns}"
            )

        if df_movies.empty:
            self.movie_ids = np.array(
                [],
                dtype=np.int64
            )

            return self

        self.movie_ids = (
            df_movies["movie_id"]
            .dropna()
            .unique()
        )

        return self

    def recommend(self, user_id, df_movies, n=5):
        if (
            self.movie_ids is None
            or len(self.movie_ids) == 0
            or n <= 0
        ):
            return []

        required_columns = {
            "movie_id",
            "title",
        }

        missing_columns = (
            required_columns
            - set(df_movies.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing required movie columns: "
                f"{missing_columns}"
            )

        sample_size = min(
            n,
            len(self.movie_ids),
        )

        chosen_movie_ids = self.rng.choice(
            self.movie_ids,
            size=sample_size,
            replace=False,
        )

        movie_lookup = (
            df_movies
            .set_index("movie_id")["title"]
            .to_dict()
        )

        recommendations = []

        for movie_id in chosen_movie_ids:
            title = movie_lookup.get(movie_id)

            if title is None:
                continue

            recommendations.append({
                "movie_id": int(movie_id),
                "title": str(title),
                "predicted_rating": 0.0,
            })

        return recommendations
