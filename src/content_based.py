import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import GENRE_COLUMNS, DEFAULT_TITLE_WEIGHT

logger = logging.getLogger(__name__)


class ContentBasedRecommender:

    def __init__(self, df_movies):
        self.df_movies = df_movies.reset_index(drop=True)
        self.genre_matrix = None
        self.tfidf_matrix = None
        self.combined_similarity = None
        self.use_tfidf = False
        self.title_weight = 0.0

        self.movie_index = pd.Series(
            self.df_movies.index,
            index=self.df_movies["movie_id"]
        ).to_dict()

    def build_model(self, use_title=True, title_weight=DEFAULT_TITLE_WEIGHT):
        self.genre_matrix = self.df_movies[GENRE_COLUMNS].values.astype(np.float32)
        genre_similarity = cosine_similarity(self.genre_matrix)

        self.use_tfidf = use_title

        if use_title:
            tfidf = TfidfVectorizer(
                stop_words="english",
                token_pattern=r"(?u)\b\w+\b"
            )

            titles = self.df_movies["title"].fillna("")
            self.tfidf_matrix = tfidf.fit_transform(titles)
            title_similarity = cosine_similarity(self.tfidf_matrix)

            self.combined_similarity = (
                (1 - title_weight) * genre_similarity +
                title_weight * title_similarity
            )

            self.title_weight = title_weight

        else:
            self.combined_similarity = genre_similarity

        logger.info(
            "Content-based model built (TF-IDF=%s, title_weight=%.2f)",
            use_title,
            title_weight if use_title else 0.0,
        )

        return self

    def recommend_similar_movies(self, movie_id, n=5, exclude_self=True):
        if self.combined_similarity is None:
            raise RuntimeError("Model has not been built. Call build_model() first.")

        idx = self.movie_index.get(movie_id)

        if idx is None:
            return []

        scores = self.combined_similarity[idx]
        ranked_indices = np.argsort(scores)[::-1]

        recommendations = []

        for i in ranked_indices:
            if exclude_self and i == idx:
                continue

            row = self.df_movies.iloc[i]

            recommendations.append({
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "similarity_score": round(float(scores[i]), 4)
            })

            if len(recommendations) >= n:
                break

        return recommendations

    def recommend_by_movie_id(self, movie_id, n=5):
        return self.recommend_similar_movies(movie_id, n=n)

    def recommend(self, movie_title, n=5):
        mask = self.df_movies["title"].str.lower().str.contains(
            movie_title.lower(),
            na=False
        )

        if not mask.any():
            return []

        movie_id = self.df_movies.loc[mask, "movie_id"].iloc[0]

        return self.recommend_similar_movies(movie_id, n=n)

    def save(self, path="models/content_model.pkl"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="models/content_model.pkl"):
        return joblib.load(path)
