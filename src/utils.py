import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from config import LOG_PATH, MODEL_PATH, POPULAR_MIN_RATINGS


def setup_logging(
    log_dir=LOG_PATH,
    console_level=logging.INFO,
    file_level=logging.DEBUG
):
    """
    Create and configure the project logger.

    A file handler and a console handler are added only once to
    prevent duplicate log messages when this module is imported
    multiple times.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("RecommenderSystem")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(
            log_dir,
            f"project_{timestamp}.log"
        )

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8"
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = setup_logging()


def save_model(
    model,
    name: str,
    model_dir=MODEL_PATH
) -> str:
    """
    Save a model using joblib.

    Returns:
        The full path of the saved model.
    """
    os.makedirs(model_dir, exist_ok=True)

    path = os.path.join(
        model_dir,
        f"{name}.pkl"
    )

    joblib.dump(model, path)

    logger.info(
        "Model '%s' saved to %s",
        name,
        path
    )

    return path


def load_model(
    name: str,
    model_dir=MODEL_PATH
):
    """
    Load a previously saved model.
    """
    path = os.path.join(
        model_dir,
        f"{name}.pkl"
    )

    if not os.path.exists(path):
        logger.error(
            "Model '%s' not found at %s",
            name,
            path
        )
        raise FileNotFoundError(
            f"Model '{name}' not found at '{path}'. "
            "Please train and save the model first."
        )

    model = joblib.load(path)

    logger.info(
        "Model '%s' loaded from %s",
        name,
        path
    )

    return model


def handle_cold_start(
    user_id: int,
    df_ratings: pd.DataFrame,
    df_movies: pd.DataFrame,
    n: int = 5
) -> Optional[List[Dict[str, Any]]]:
    """
    Return popularity-based recommendations for a new user.

    Returns None when the user already exists in the ratings data.
    """
    required_rating_columns = {
        "user_id",
        "movie_id",
        "rating"
    }

    required_movie_columns = {
        "movie_id",
        "title"
    }

    if not required_rating_columns.issubset(df_ratings.columns):
        missing = required_rating_columns - set(df_ratings.columns)
        raise ValueError(
            f"df_ratings is missing required columns: {sorted(missing)}"
        )

    if not required_movie_columns.issubset(df_movies.columns):
        missing = required_movie_columns - set(df_movies.columns)
        raise ValueError(
            f"df_movies is missing required columns: {sorted(missing)}"
        )

    if df_ratings.empty:
        return None

    if user_id in df_ratings["user_id"].values:
        return None

    logger.info(
        "Cold-start user %s -> using popularity-based recommendations",
        user_id
    )

    movie_stats = (
        df_ratings
        .groupby("movie_id")["rating"]
        .agg(["mean", "count"])
    )

    global_average = float(
        df_ratings["rating"].mean()
    )

    movie_stats["bayesian_avg"] = (
        (
            movie_stats["mean"] * movie_stats["count"]
            + global_average * POPULAR_MIN_RATINGS
        )
        /
        (
            movie_stats["count"]
            + POPULAR_MIN_RATINGS
        )
    )

    top_movies = movie_stats.nlargest(
        n,
        "bayesian_avg"
    )

    movie_lookup = (
        df_movies
        .drop_duplicates("movie_id")
        .set_index("movie_id")["title"]
        .to_dict()
    )

    recommendations = []

    for movie_id, row in top_movies.iterrows():
        title = movie_lookup.get(movie_id)

        if title is None:
            continue

        recommendations.append({
            "movie_id": int(movie_id),
            "title": str(title),
            "predicted_rating": round(
                float(row["bayesian_avg"]),
                3
            )
        })

    return recommendations


def get_movie_id_from_title(
    title: str,
    df_movies: pd.DataFrame
) -> Optional[int]:
    """
    Find a movie ID using an exact title match first,
    then a case-insensitive partial match.
    """
    if not title or df_movies.empty:
        return None

    required_columns = {
        "movie_id",
        "title"
    }

    if not required_columns.issubset(df_movies.columns):
        missing = required_columns - set(df_movies.columns)
        raise ValueError(
            f"df_movies is missing required columns: {sorted(missing)}"
        )

    normalized_title = title.strip().lower()

    if not normalized_title:
        return None

    titles = (
        df_movies["title"]
        .fillna("")
        .astype(str)
    )

    exact_match = df_movies[
        titles.str.lower() == normalized_title
    ]

    if not exact_match.empty:
        return int(
            exact_match.iloc[0]["movie_id"]
        )

    partial_match = df_movies[
        titles.str.lower().str.contains(
            normalized_title,
            na=False,
            regex=False
        )
    ]

    if not partial_match.empty:
        return int(
            partial_match.iloc[0]["movie_id"]
        )

    return None


def get_popularity_dict(
    df_ratings: pd.DataFrame
) -> Dict[int, int]:
    """
    Return the number of ratings for each movie.
    """
    if (
        df_ratings.empty
        or "movie_id" not in df_ratings.columns
    ):
        return {}

    counts = (
        df_ratings
        .groupby("movie_id")["movie_id"]
        .count()
        .to_dict()
    )

    return {
        int(movie_id): int(count)
        for movie_id, count in counts.items()
    }
