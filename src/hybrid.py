import pandas as pd

class HybridRecommender:
    """
    Hybrid recommender that combines content-based and collaborative filtering.
    """

    def __init__(self, content_model, collab_model, df_ratings, df_movies):
        """
        Parameters
        ----------
        content_model : ContentBasedRecommender
        collab_model : CollaborativeRecommender
        df_ratings : pandas.DataFrame
        df_movies : pandas.DataFrame
        """
        self.content_model = content_model
        self.collab_model = collab_model
        self.df_ratings = df_ratings
        self.df_movies = df_movies

    def recommend_by_movie_id(self, user_id, movie_id, n=5, content_weight=0.4):
        """
        Recommend movies to a user based on a liked movie (by movie_id).

        Parameters
        ----------
        user_id : int
        movie_id : int
        n : int
            Number of recommendations.
        content_weight : float
            Weight for content-based scores (0 = purely collaborative, 1 = purely content).

        Returns
        -------
        list of dict
            Each dict contains 'title', 'confidence', and 'reasons'.
        """
        content_recs = self.content_model.recommend_by_movie_id(movie_id, n=n*2)
        collab_recs = self.collab_model.recommend_for_user(user_id, self.df_ratings, self.df_movies, n=n*2)

        hybrid_scores = {}
        for i, rec in enumerate(content_recs):
            title = rec['title']
            score = (1 - i/(n*2)) * content_weight
            hybrid_scores[title] = hybrid_scores.get(title, 0) + score
        for i, rec in enumerate(collab_recs):
            title = rec['title']
            score = (1 - i/(n*2)) * (1 - content_weight)
            hybrid_scores[title] = hybrid_scores.get(title, 0) + score

        sorted_movies = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        recommendations = []
        for title, score in sorted_movies:
            reasons = []
            if any(rec['title'] == title for rec in content_recs):
                reasons.append("similar to your liked movie")
            if any(rec['title'] == title for rec in collab_recs):
                reasons.append("popular among similar users")
            recommendations.append({
                'title': title,
                'confidence': round(score, 3),
                'reasons': reasons
            })
        return recommendations

    def recommend(self, user_id, liked_movie_title, n=5, content_weight=0.4):
        """
        Recommend movies based on a liked movie title (substring match).
        Falls back to collaborative if title not found.
        """
        mask = self.df_movies['title'].str.lower().str.contains(liked_movie_title.lower(), na=False)
        if mask.any():
            movie_id = self.df_movies[mask].iloc[0]['movie_id']
            return self.recommend_by_movie_id(user_id, movie_id, n, content_weight)
        else:
            collab_recs = self.collab_model.recommend_for_user(user_id, self.df_ratings, self.df_movies, n=n)
            return [{'title': r['title'], 'confidence': r['predicted_rating'], 'reasons': ['popular among similar users']} for r in collab_recs]

    def recommend_similar_movies(self, movie_id, n=5, content_weight=0.6):
        """
        Find movies similar to a given movie (without a user).

        Parameters
        ----------
        movie_id : int
        n : int
        content_weight : float
            Weight for content-based similarity.

        Returns
        -------
        list of dict
            Each dict contains 'title', 'confidence', and 'reasons'.
        """
        content_recs = self.content_model.recommend_by_movie_id(movie_id, n=n*2)
        collab_recs = self.collab_model.recommend_similar_movies(movie_id, self.df_movies, n=n*2)

        hybrid_scores = {}
        for i, rec in enumerate(content_recs):
            title = rec['title']
            score = (1 - i/(n*2)) * content_weight
            hybrid_scores[title] = hybrid_scores.get(title, 0) + score
        for i, rec in enumerate(collab_recs):
            title = rec['title']
            score = (1 - i/(n*2)) * (1 - content_weight)
            hybrid_scores[title] = hybrid_scores.get(title, 0) + score

        sorted_movies = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        recommendations = []
        for title, score in sorted_movies:
            reasons = []
            if any(rec['title'] == title for rec in content_recs):
                reasons.append("similar genres/title")
            if any(rec['title'] == title for rec in collab_recs):
                reasons.append("users who liked this also liked")
            recommendations.append({
                'title': title,
                'confidence': round(score, 3),
                'reasons': reasons
            })
        return recommendations
