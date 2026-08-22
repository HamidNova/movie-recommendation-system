import numpy as np
import pandas as pd


class PopularRecommender:

    def __init__(self, min_ratings=10):
        self.min_ratings = min_ratings
        self.popularity_scores = None

    def fit(self, df_ratings):
        if df_ratings.empty:
            self.popularity_scores = pd.Series(dtype=float)
            return self

        movie_stats = df_ratings.groupby('movie_id')['rating'].agg(['mean', 'count'])
        global_avg = df_ratings['rating'].mean()
        movie_stats['bayesian_avg'] = (
            (movie_stats['mean'] * movie_stats['count'] + global_avg * self.min_ratings) /
            (movie_stats['count'] + self.min_ratings)
        )
        self.popularity_scores = movie_stats['bayesian_avg'].sort_values(ascending=False)
        return self

    def recommend(self, user_id, df_movies, n=5):
        if self.popularity_scores is None or self.popularity_scores.empty:
            return []

        top_movie_ids = self.popularity_scores.head(n).index
        recommendations = []
        for mid in top_movie_ids:
            match = df_movies[df_movies['movie_id'] == mid]
            if match.empty:
                continue
            title = match['title'].iloc[0]
            recommendations.append({
                'movie_id': int(mid),
                'title': str(title),
                'predicted_rating': float(self.popularity_scores[mid])
            })
        return recommendations


class RandomRecommender:

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.movie_ids = None

    def fit(self, df_movies):
        if df_movies.empty:
            self.movie_ids = np.array([])
            return self

        self.movie_ids = df_movies['movie_id'].unique()
        return self

    def recommend(self, user_id, df_movies, n=5):
        if self.movie_ids is None or len(self.movie_ids) == 0:
            return []

        sample_size = min(n, len(self.movie_ids))
        chosen = self.rng.choice(self.movie_ids, size=sample_size, replace=False)
        recommendations = []
        for mid in chosen:
            match = df_movies[df_movies['movie_id'] == mid]
            if match.empty:
                continue
            title = match['title'].iloc[0]
            recommendations.append({
                'movie_id': int(mid),
                'title': str(title),
                'predicted_rating': 0.0
            })
        return recommendations
