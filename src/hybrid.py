# src/hybrid.py
import pandas as pd

class HybridRecommender:
    def __init__(self, content_model, collab_model, df_ratings, df_movies):
        self.content_model = content_model
        self.collab_model = collab_model
        self.df_ratings = df_ratings
        self.df_movies = df_movies

    def recommend_by_movie_id(self, user_id, movie_id, n=5, content_weight=0.4):
        """توصیه به کاربر با استفاده از movie_id فیلم مورد علاقه"""
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

    # متد قبلی برای سازگاری با عنوان (اختیاری)
    def recommend(self, user_id, liked_movie_title, n=5, content_weight=0.4):
        mask = self.df_movies['title'].str.lower().str.contains(liked_movie_title.lower(), na=False)
        if mask.any():
            movie_id = self.df_movies[mask].iloc[0]['movie_id']
            return self.recommend_by_movie_id(user_id, movie_id, n, content_weight)
        else:
            # fallback: فقط collaborative
            collab_recs = self.collab_model.recommend_for_user(user_id, self.df_ratings, self.df_movies, n=n)
            return [{'title': r['title'], 'confidence': r['predicted_rating'], 'reasons': ['popular among similar users']} for r in collab_recs]

    def recommend_similar_movies(self, movie_id, n=5, content_weight=0.6):
        """فیلم‌های مشابه به یک فیلم مشخص (بدون کاربر)"""
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
