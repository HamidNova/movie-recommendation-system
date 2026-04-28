import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
import joblib
import logging

logger = logging.getLogger(__name__)

class CollaborativeRecommender:
    """
    Collaborative filtering recommender using ALS (Alternating Least Squares)
    with confidence weighting and cold-start handling.
    """

    def __init__(self, alpha=2.0, use_gpu=False):
        """
        Parameters
        ----------
        alpha : float
            Confidence parameter: confidence = 1 + alpha * rating
        use_gpu : bool
            Whether to use GPU for ALS training.
        """
        self.alpha = alpha
        self.use_gpu = use_gpu
        self.model = None
        self.user_map = None
        self.item_map = None
        self.user_reverse = None
        self.item_reverse = None
        self.sparse_matrix = None
        self._popular_cache = None

    def _build_confidence_matrix(self, df_ratings):
        """Convert ratings to confidence weights"""
        rows = df_ratings['user_id'].map(self.user_map).values
        cols = df_ratings['movie_id'].map(self.item_map).values
        data = (1.0 + self.alpha * df_ratings['rating'].values).astype(np.float32)
        return csr_matrix((data, (rows, cols)))

    def _cache_popular_items(self, df_ratings, min_ratings=10):
        """Bayesian average for cold-start"""
        movie_stats = df_ratings.groupby('movie_id')['rating'].agg(['mean', 'count'])
        global_avg = df_ratings['rating'].mean()
        movie_stats['bayesian_avg'] = (
            (movie_stats['mean'] * movie_stats['count'] + global_avg * min_ratings) /
            (movie_stats['count'] + min_ratings)
        )
        self._popular_cache = movie_stats['bayesian_avg'].sort_values(ascending=False)

    def build_model(self, df_ratings, factors=100, iterations=30, regularization=0.05):
        """Build ALS model with confidence weights"""
        self.user_map = {uid: i for i, uid in enumerate(df_ratings['user_id'].unique())}
        self.item_map = {mid: i for i, mid in enumerate(df_ratings['movie_id'].unique())}
        self.user_reverse = {i: uid for uid, i in self.user_map.items()}
        self.item_reverse = {i: mid for mid, i in self.item_map.items()}

        self.sparse_matrix = self._build_confidence_matrix(df_ratings)

        self.model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=42,
            use_gpu=self.use_gpu
        )
        self.model.fit(self.sparse_matrix)

        self._cache_popular_items(df_ratings)

        logger.info(f"ALS model built: factors={factors}, iter={iterations}, reg={regularization}, alpha={self.alpha}")
        return self

    def recommend_for_user(self, user_id, df_ratings, df_movies, n=5):
        """Recommend movies to a specific user (handles cold-start)"""
        if user_id not in self.user_map:
            top_movies = self._popular_cache.head(n).index
            recommendations = []
            for mid in top_movies:
                title = df_movies[df_movies['movie_id'] == mid]['title'].iloc[0]
                recommendations.append({'title': title, 'predicted_rating': float(self._popular_cache[mid])})
            return recommendations

        user_idx = self.user_map[user_id]
        ids, scores = self.model.recommend(
            user_idx,
            self.sparse_matrix[user_idx],
            N=n,
            filter_already_liked_items=True
        )
        recommendations = []
        for idx, score in zip(ids, scores):
            movie_id = self.item_reverse[idx]
            title = df_movies[df_movies['movie_id'] == movie_id]['title'].iloc[0]
            recommendations.append({'title': title, 'predicted_rating': round(score, 2)})
        return recommendations

    def recommend_similar_movies(self, movie_id, df_movies, n=5):
        """
        Find similar movies based on item embeddings (without needing a user).

        Parameters
        ----------
        movie_id : int
            Target movie ID.
        df_movies : pandas.DataFrame
            Movie metadata.
        n : int
            Number of similar movies to return.

        Returns
        -------
        list of dict
            Each dict contains 'movie_id', 'title', and 'similarity_score'.
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model first.")

        if movie_id not in self.item_map:
            logger.warning(f"Movie {movie_id} not in training set. Returning popular movies.")
            top_movies = self._popular_cache.head(n).index
            recommendations = []
            for mid in top_movies:
                title = df_movies[df_movies['movie_id'] == mid]['title'].iloc[0]
                recommendations.append({'title': title, 'similarity_score': float(self._popular_cache[mid])})
            return recommendations

        item_idx = self.item_map[movie_id]
        similar_ids, scores = self.model.similar_items(item_idx, N=n+1)
        recommendations = []
        for idx, score in zip(similar_ids, scores):
            if idx == item_idx:
                continue
            mid = self.item_reverse[idx]
            title = df_movies[df_movies['movie_id'] == mid]['title'].iloc[0]
            recommendations.append({
                'movie_id': mid,
                'title': title,
                'similarity_score': float(score)
            })
            if len(recommendations) >= n:
                break
        return recommendations

    def save(self, path='models/collaborative_model.pkl'):
        """Save model to disk using joblib."""
        joblib.dump(self, path)

    @staticmethod
    def load(path='models/collaborative_model.pkl'):
        """Load model from disk using joblib."""
        return joblib.load(path)
