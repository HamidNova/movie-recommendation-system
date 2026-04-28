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

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

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
    """Save model with given name and log the action."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f'{name}.pkl')
    joblib.dump(model, path)
    logger.info(f"Model '{name}' saved to {path}")


def load_model(name: str, model_dir='models'):
    """Load model from file."""
    path = os.path.join(model_dir, f'{name}.pkl')
    if not os.path.exists(path):
        logger.error(f"Model '{name}' not found at {path}")
        raise FileNotFoundError(f"Model {name} not found. Please run main.py first.")
    model = joblib.load(path)
    logger.info(f"Model '{name}' loaded from {path}")
    return model


def handle_cold_start(user_id: int, df_ratings: pd.DataFrame, df_movies: pd.DataFrame, n: int = 5) -> Optional[List[str]]:
    """
    If the user is new, return popular movies based on Bayesian average.
    """
    if user_id in df_ratings['user_id'].values:
        return None
    logger.info(f"Cold-start user {user_id} -> using popularity-based recommendation")
    movie_stats = df_ratings.groupby('movie_id')['rating'].agg(['mean', 'count'])
    global_avg = df_ratings['rating'].mean()
    min_ratings = 10
    movie_stats['bayesian_avg'] = (
            (movie_stats['mean'] * movie_stats['count'] + global_avg * min_ratings) /
            (movie_stats['count'] + min_ratings)
    )
    top_movies = movie_stats.nlargest(n, 'bayesian_avg').index
    result = []
    for mid in top_movies:
        title_row = df_movies[df_movies['movie_id'] == mid]['title']
        if not title_row.empty:
            result.append(title_row.iloc[0])
    return result


def get_movie_id_from_title(title: str, df_movies: pd.DataFrame) -> Optional[int]:
    """Find movie_id by title (substring search)."""
    mask = df_movies['title'].str.contains(title, case=False, na=False)
    if mask.any():
        return df_movies[mask].iloc[0]['movie_id']
    return None


def get_popularity_dict(df_ratings: pd.DataFrame) -> Dict[int, int]:
    """Return dictionary of movie popularity (number of ratings)."""
    return df_ratings.groupby('movie_id')['rating'].count().to_dict()
