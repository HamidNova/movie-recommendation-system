import logging
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any


def setup_logging(log_dir='logs', console_level=logging.INFO, file_level=logging.DEBUG):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'project_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logger = logging.getLogger("RecommenderSystem")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        fh = logging.FileHandler(log_file)
        fh.setLevel(file_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


logger = setup_logging()


def save_model(model, name: str, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f'{name}.pkl')
    joblib.dump(model, path)
    logger.info(f"Model '{name}' saved to {path}")


def load_model(name: str, model_dir='models'):
    path = os.path.join(model_dir, f'{name}.pkl')
    if not os.path.exists(path):
        logger.error(f"Model '{name}' not found at {path}")
        raise FileNotFoundError(f"Model {name} not found. Please run main.py first.")
    model = joblib.load(path)
    logger.info(f"Model '{name}' loaded from {path}")
    return model


def handle_cold_start(user_id: int, df_ratings: pd.DataFrame, df_movies: pd.DataFrame, n: int = 5) -> Optional[List[Dict[str, Any]]]:
    if df_ratings.empty or user_id in df_ratings['user_id'].values:
        return None

    logger.info(f"Cold-start user {user_id} -> using popularity-based recommendation")
    movie_stats = df_ratings.groupby('movie_id')['rating'].agg(['mean', 'count'])
    global_avg = float(df_ratings['rating'].mean())
    min_ratings = 10
    movie_stats['bayesian_avg'] = (
            (movie_stats['mean'] * movie_stats['count'] + global_avg * min_ratings) /
            (movie_stats['count'] + min_ratings)
    )
    top_movies = movie_stats.nlargest(n, 'bayesian_avg')
    result = []
    for mid, row in top_movies.iterrows():
        title_row = df_movies[df_movies['movie_id'] == mid]
        if not title_row.empty:
            result.append({
                'movie_id': int(mid),
                'title': str(title_row.iloc[0]['title']),
                'predicted_rating': round(float(row['bayesian_avg']), 2)
            })
    return result


def get_movie_id_from_title(title: str, df_movies: pd.DataFrame) -> Optional[int]:
    if df_movies.empty or not title:
        return None

    exact_match = df_movies[df_movies['title'].str.lower() == title.lower()]
    if not exact_match.empty:
        return int(exact_match.iloc[0]['movie_id'])

    mask = df_movies['title'].str.lower().str.contains(title.lower(), na=False, regex=False)
    if mask.any():
        return int(df_movies[mask].iloc[0]['movie_id'])

    return None


def get_popularity_dict(df_ratings: pd.DataFrame) -> Dict[int, int]:
    if df_ratings.empty or 'movie_id' not in df_ratings.columns:
        return {}
    counts = df_ratings.groupby('movie_id')['rating'].count().to_dict()
    return {int(k): int(v) for k, v in counts.items()}
