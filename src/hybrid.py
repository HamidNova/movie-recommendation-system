import pandas as pd


class HybridRecommender:

    def __init__(self, content_model, collab_model, df_ratings, df_movies):
        self.content_model = content_model
        self.collab_model = collab_model
        self.df_ratings = df_ratings
        self.df_movies = df_movies

    def _get_movie_id_by_title(self, title):
        row = self.df_movies[self.df_movies['title'] == title]
        if not row.empty:
            return int(row['movie_id'].iloc[0])
        return None

    def recommend_by_movie_id(self, user_id, movie_id, n=5, content_weight=0.4):
        fetch_n = max(1, n * 2)
        content_recs = []
        if hasattr(self.content_model, 'recommend_by_movie_id'):
            content_recs = self.content_model.recommend_by_movie_id(movie_id, n=fetch_n) or []

        collab_recs = []
        if hasattr(self.collab_model, 'recommend_for_user'):
            collab_recs = self.collab_model.recommend_for_user(user_id, self.df_ratings, self.df_movies, n=fetch_n) or []

        hybrid_scores = {}
        for i, rec in enumerate(content_recs):
            title = rec.get('title')
            if not title:
                continue
            score = (1 - (i / fetch_n)) * content_weight
            hybrid_scores[title] = hybrid_scores.get(title, 0.0) + score

        for i, rec in enumerate(collab_recs):
            title = rec.get('title')
            if not title:
                continue
            score = (1 - (i / fetch_n)) * (1 - content_weight)
            hybrid_scores[title] = hybrid_scores.get(title, 0.0) + score

        sorted_movies = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        recommendations = []
        for title, score in sorted_movies:
            reasons = []
            if any(rec.get('title') == title for rec in content_recs):
                reasons.append("similar to your liked movie")
            if any(rec.get('title') == title for rec in collab_recs):
                reasons.append("popular among similar users")

            mid = self._get_movie_id_by_title(title)
            rec_dict = {
                'title': title,
                'confidence': round(float(score), 3),
                'predicted_rating': round(float(score * 5), 2),
                'reasons': reasons
            }
            if mid is not None:
                rec_dict['movie_id'] = mid
            recommendations.append(rec_dict)
        return recommendations

    def recommend(self, user_id, liked_movie_title, n=5, content_weight=0.4):
        mask = self.df_movies['title'].str.lower().str.contains(liked_movie_title.lower(), na=False, regex=False)
        if mask.any():
            movie_id = int(self.df_movies[mask].iloc[0]['movie_id'])
            return self.recommend_by_movie_id(user_id, movie_id, n, content_weight)
        else:
            collab_recs = []
            if hasattr(self.collab_model, 'recommend_for_user'):
                collab_recs = self.collab_model.recommend_for_user(user_id, self.df_ratings, self.df_movies, n=n) or []

            res = []
            for r in collab_recs:
                title = r.get('title', 'Unknown')
                mid = self._get_movie_id_by_title(title) or r.get('movie_id')
                pred_rating = r.get('predicted_rating', 4.0)
                rec_dict = {
                    'title': title,
                    'confidence': round(float(pred_rating / 5.0), 3),
                    'predicted_rating': round(float(pred_rating), 2),
                    'reasons': ['popular among similar users']
                }
                if mid is not None:
                    rec_dict['movie_id'] = int(mid)
                res.append(rec_dict)
            return res

    def recommend_similar_movies(self, movie_id, n=5, content_weight=0.6):
        fetch_n = max(1, n * 2)
        content_recs = []
        if hasattr(self.content_model, 'recommend_by_movie_id'):
            content_recs = self.content_model.recommend_by_movie_id(movie_id, n=fetch_n) or []

        collab_recs = []
        if hasattr(self.collab_model, 'recommend_similar_movies'):
            collab_recs = self.collab_model.recommend_similar_movies(movie_id, self.df_movies, n=fetch_n) or []

        hybrid_scores = {}
        for i, rec in enumerate(content_recs):
            title = rec.get('title')
            if not title:
                continue
            score = (1 - (i / fetch_n)) * content_weight
            hybrid_scores[title] = hybrid_scores.get(title, 0.0) + score

        for i, rec in enumerate(collab_recs):
            title = rec.get('title')
            if not title:
                continue
            score = (1 - (i / fetch_n)) * (1 - content_weight)
            hybrid_scores[title] = hybrid_scores.get(title, 0.0) + score

        sorted_movies = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        recommendations = []
        for title, score in sorted_movies:
            reasons = []
            if any(rec.get('title') == title for rec in content_recs):
                reasons.append("similar genres/title")
            if any(rec.get('title') == title for rec in collab_recs):
                reasons.append("users who liked this also liked")

            mid = self._get_movie_id_by_title(title)
            rec_dict = {
                'title': title,
                'confidence': round(float(score), 3),
                'predicted_rating': round(float(score * 5), 2),
                'reasons': reasons
            }
            if mid is not None:
                rec_dict['movie_id'] = mid
            recommendations.append(rec_dict)
        return recommendations
