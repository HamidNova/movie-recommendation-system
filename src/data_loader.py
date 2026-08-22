import os
import logging
import numpy as np
import pandas as pd

from config import GENRE_COLUMNS

logger = logging.getLogger(__name__)

GENRE_LIST = GENRE_COLUMNS[1:]


def _convert_movies(df_movies_raw):
    genres = (
        df_movies_raw['genres']
        .fillna('')
        .str.get_dummies(sep='|')
        .reindex(columns=GENRE_LIST, fill_value=0)
        .astype(np.float32)
    )

    unknown = (genres.sum(axis=1) == 0).astype(np.float32)

    df_movies = pd.concat(
        [
            df_movies_raw[['movie_id', 'title']].reset_index(drop=True),
            pd.DataFrame({'unknown': unknown}).reset_index(drop=True),
            genres.reset_index(drop=True)
        ],
        axis=1
    )

    return df_movies


def load_movielens(data_path='data/'):

    os.makedirs(data_path, exist_ok=True)

    ratings_csv = os.path.join(data_path, 'ratings.csv')
    movies_csv = os.path.join(data_path, 'movies_with_genres.csv')

    if os.path.exists(ratings_csv) and os.path.exists(movies_csv):
        header = pd.read_csv(ratings_csv, nrows=0).columns.tolist()

        if 'user_id' in header:
            logger.info("Loading processed dataset...")
            return (
                pd.read_csv(ratings_csv),
                pd.read_csv(movies_csv)
            )

    ratings_dat = os.path.join(data_path, 'ratings.dat')
    movies_dat = os.path.join(data_path, 'movies.dat')

    if not (os.path.exists(ratings_dat) and os.path.exists(movies_dat)):
        raise FileNotFoundError(
            "MovieLens dataset not found.\n"
            "Please place ratings.dat and movies.dat inside the data folder."
        )

    logger.info("Converting MovieLens 1M dataset...")

    df_ratings = pd.read_csv(
        ratings_dat,
        sep='::',
        engine='python',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )

    df_movies_raw = pd.read_csv(
        movies_dat,
        sep='::',
        engine='python',
        encoding='latin-1',
        names=['movie_id', 'title', 'genres']
    )

    df_movies = _convert_movies(df_movies_raw)

    df_ratings.to_csv(ratings_csv, index=False)
    df_movies.to_csv(movies_csv, index=False)

    logger.info("Converted dataset saved successfully.")

    return df_ratings, df_movies


def get_genre_matrix(df_movies):
    return df_movies[GENRE_COLUMNS].to_numpy(dtype=np.float32)
