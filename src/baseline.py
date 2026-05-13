import numpy as np
import pandas as pd

class PopularRecommender:
    """
    Recommend the most popular movies based on Bayesian average rating.
    """
    def __init__(self, min_ratings=10):
        self.min_ratings = min_ratings
        self.popularity_scores = None

    def fit(self, df_ratings):
        """Compute Bayesian average for each movie."""
        movie_stats = df_ratings.groupby('movie_id')['rating'].agg(['mean', 'count'])
        global_avg = df_ratings['rating'].mean()
        movie_stats['bayesian_avg'] = (
            (movie_stats['mean'] * movie_stats['count'] + global_avg * self.min_ratings) /
            (movie_stats['count'] + self.min_ratings)
        )
        self.popularity_scores = movie_stats['bayesian_avg'].sort_values(ascending=False)
        return self

    def recommend(self, user_id, df_movies, n=5):
        """Return top-n popular movies (same for every user)."""
        top_movie_ids = self.popularity_scores.head(n).index
        recommendations = []
        for mid in top_movie_ids:
            title = df_movies[df_movies['movie_id'] == mid]['title'].values[0]
            recommendations.append({
                'movie_id': mid,
                'title': title,
                'predicted_rating': float(self.popularity_scores[mid])
            })
        return recommendations


class RandomRecommender:
    """
    Recommend random movies (uniform random baseline).
    """
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.movie_ids = None

    def fit(self, df_movies):
        """Store all movie ids for random selection."""
        self.movie_ids = df_movies['movie_id'].unique()
        return self

    def recommend(self, user_id, df_movies, n=5):
        """Return n random movies."""
        chosen = self.rng.choice(self.movie_ids, size=n, replace=False)
        recommendations = []
        for mid in chosen:
            title = df_movies[df_movies['movie_id'] == mid]['title'].values[0]
            recommendations.append({
                'movie_id': mid,
                'title': title,
                'predicted_rating': 0.0   # no real score
            })
        return recommendations
