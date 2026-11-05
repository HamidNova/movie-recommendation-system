import os
import logging

import numpy as np
import pandas as pd

from config import DATA_PATH, GENRE_COLUMNS


logger = logging.getLogger(__name__)


if not GENRE_COLUMNS:
    raise ValueError("GENRE_COLUMNS cannot be empty.")

if GENRE_COLUMNS[0] != "unknown":
    raise ValueError(
        "GENRE_COLUMNS must start with 'unknown' to match the processed "
        "movie genre matrix."
    )


GENRE_LIST = GENRE_COLUMNS[1:]


def _convert_movies(df_movies_raw):
    """
    Convert the raw MovieLens movies dataframe into a dataframe
    containing movie metadata and one-hot encoded genre columns.
    """

    required_columns = {
        "movie_id",
        "title",
        "genres",
    }

    missing_columns = (
        required_columns
        - set(df_movies_raw.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required movie columns: {missing_columns}"
        )

    genres = (
        df_movies_raw["genres"]
        .fillna("")
        .astype(str)
        .str.get_dummies(sep="|")
        .reindex(
            columns=GENRE_LIST,
            fill_value=0,
        )
        .astype(np.float32)
    )

    unknown = (
        genres.sum(axis=1) == 0
    ).astype(np.float32)

    df_movies = pd.concat(
        [
            df_movies_raw[
                ["movie_id", "title"]
            ].reset_index(drop=True),

            pd.DataFrame(
                {"unknown": unknown}
            ).reset_index(drop=True),

            genres.reset_index(drop=True),
        ],
        axis=1,
    )

    df_movies["movie_id"] = (
        df_movies["movie_id"].astype(int)
    )

    df_movies["title"] = (
        df_movies["title"].astype(str)
    )

    return df_movies


def _validate_processed_files(
    ratings_csv,
    movies_csv,
):
    """
    Check whether processed CSV files have the required structure.
    """

    try:
        ratings_header = (
            pd.read_csv(
                ratings_csv,
                nrows=0,
            )
            .columns
            .tolist()
        )

        movies_header = (
            pd.read_csv(
                movies_csv,
                nrows=0,
            )
            .columns
            .tolist()
        )

    except Exception as exc:
        logger.warning(
            "Could not read processed dataset headers: %s",
            exc,
        )

        return False

    required_rating_columns = {
        "user_id",
        "movie_id",
        "rating",
    }

    required_movie_columns = {
        "movie_id",
        "title",
        *GENRE_COLUMNS,
    }

    ratings_valid = (
        required_rating_columns
        .issubset(ratings_header)
    )

    movies_valid = (
        required_movie_columns
        .issubset(movies_header)
    )

    return ratings_valid and movies_valid


def load_movielens(data_path=DATA_PATH):
    """
    Load the processed MovieLens dataset.

    If processed CSV files do not exist or have an incompatible
    structure, rebuild them from MovieLens 1M source files.
    """

    os.makedirs(
        data_path,
        exist_ok=True,
    )

    ratings_csv = os.path.join(
        data_path,
        "ratings.csv",
    )

    movies_csv = os.path.join(
        data_path,
        "movies_with_genres.csv",
    )

    processed_files_exist = (
        os.path.exists(ratings_csv)
        and os.path.exists(movies_csv)
    )

    if processed_files_exist:

        if _validate_processed_files(
            ratings_csv,
            movies_csv,
        ):
            logger.info(
                "Loading processed dataset..."
            )

            df_ratings = pd.read_csv(
                ratings_csv,
            )

            df_movies = pd.read_csv(
                movies_csv,
            )

            return (
                df_ratings,
                df_movies,
            )

        logger.warning(
            "Processed CSV files have an incompatible structure. "
            "Rebuilding dataset from MovieLens source files."
        )

    ratings_dat = os.path.join(
        data_path,
        "ratings.dat",
    )

    movies_dat = os.path.join(
        data_path,
        "movies.dat",
    )

    source_files_exist = (
        os.path.exists(ratings_dat)
        and os.path.exists(movies_dat)
    )

    if not source_files_exist:

        raise FileNotFoundError(
            "MovieLens dataset not found.\n"
            "Expected either:\n"
            "- ratings.csv and movies_with_genres.csv\n"
            "or:\n"
            "- ratings.dat and movies.dat\n"
            f"inside: {data_path}"
        )

    logger.info(
        "Converting MovieLens 1M dataset..."
    )

    df_ratings = pd.read_csv(
        ratings_dat,
        sep="::",
        engine="python",
        names=[
            "user_id",
            "movie_id",
            "rating",
            "timestamp",
        ],
    )

    df_movies_raw = pd.read_csv(
        movies_dat,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=[
            "movie_id",
            "title",
            "genres",
        ],
    )

    df_ratings["user_id"] = (
        df_ratings["user_id"].astype(int)
    )

    df_ratings["movie_id"] = (
        df_ratings["movie_id"].astype(int)
    )

    df_ratings["rating"] = (
        df_ratings["rating"].astype(float)
    )

    df_movies = _convert_movies(
        df_movies_raw
    )

    df_ratings.to_csv(
        ratings_csv,
        index=False,
    )

    df_movies.to_csv(
        movies_csv,
        index=False,
    )

    logger.info(
        "Converted dataset saved successfully."
    )

    return (
        df_ratings,
        df_movies,
    )


def get_genre_matrix(df_movies):
    """
    Return the movie genre feature matrix.
    """

    required_columns = set(
        GENRE_COLUMNS
    )

    missing_columns = (
        required_columns
        - set(df_movies.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing genre columns: {missing_columns}"
        )

    return (
        df_movies[GENRE_COLUMNS]
        .to_numpy(
            dtype=np.float32,
        )
    )
