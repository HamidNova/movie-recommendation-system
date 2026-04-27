# src/content_based.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class ContentBasedRecommender:
    def __init__(self, df_movies):
        self.df_movies = df_movies
        self.genre_matrix = None
        self.tfidf_matrix = None
        self.combined_similarity = None
        self.use_tfidf = False
        self.title_weight = 0.0

    def build_model(self, use_title=True, title_weight=0.2):
        genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                      'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                      'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        self.genre_matrix = self.df_movies[genre_cols].values.astype(np.float32)
        genre_sim = cosine_similarity(self.genre_matrix)

        self.use_tfidf = use_title
        if use_title:
            tfidf = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b\w+\b')
            titles = self.df_movies['title'].fillna('')
            self.tfidf_matrix = tfidf.fit_transform(titles)
            title_sim = cosine_similarity(self.tfidf_matrix)
            self.combined_similarity = (1 - title_weight) * genre_sim + title_weight * title_sim
            self.title_weight = title_weight
        else:
            self.combined_similarity = genre_sim
        print(f"✅ Content-based model built: use_title={use_title}, title_weight={title_weight if use_title else 0}")
        return self

    def recommend_similar_movies(self, movie_id, n=5, exclude_self=True):
        idx = self.df_movies[self.df_movies['movie_id'] == movie_id].index
        if len(idx) == 0:
            return []
        idx = idx[0]
        sim_scores = list(enumerate(self.combined_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        if exclude_self:
            sim_scores = sim_scores[1:n+1]
        else:
            sim_scores = sim_scores[:n]
        recommendations = []
        for i, score in sim_scores:
            row = self.df_movies.iloc[i]
            recommendations.append({
                'movie_id': int(row['movie_id']),
                'title': row['title'],
                'similarity_score': round(float(score), 4)
            })
        return recommendations

    # متد جدید: مستقیماً با movie_id
    def recommend_by_movie_id(self, movie_id, n=5):
        return self.recommend_similar_movies(movie_id, n=n)

    # متد قدیمی برای سازگاری (توصیه بر اساس عنوان)
    def recommend(self, movie_title, n=5):
        mask = self.df_movies['title'].str.lower().str.contains(movie_title.lower(), na=False)
        if not mask.any():
            return []
        movie_id = self.df_movies[mask].iloc[0]['movie_id']
        return self.recommend_similar_movies(movie_id, n=n)

    def save(self, path='models/content_model.pkl'):
        joblib.dump(self, path)

    @staticmethod
    def load(path='models/content_model.pkl'):
        return joblib.load(path)
