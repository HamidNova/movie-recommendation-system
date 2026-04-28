import pandas as pd
import os
import numpy as np


def load_movielens(data_path='data/'):
    """Load MovieLens 100k dataset. Tries local CSV, raw files, then downloads."""
    os.makedirs(data_path, exist_ok=True)

    if os.path.exists(f'{data_path}/ratings.csv') and os.path.exists(f'{data_path}/movies_with_genres.csv'):
        df_ratings = pd.read_csv(f'{data_path}/ratings.csv')
        df_movies = pd.read_csv(f'{data_path}/movies_with_genres.csv')
        print("Data loaded from local csv files (including genres)")
        return df_ratings, df_movies

    if os.path.exists(f'{data_path}/u.data') and os.path.exists(f'{data_path}/u.item'):
        print("Reading original u.data and u.item files...")
        df_ratings = pd.read_csv(f'{data_path}/u.data', sep='\t', header=None,
                                 names=['user_id', 'movie_id', 'rating', 'timestamp'])

        genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                      'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                      'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        df_movies_raw = pd.read_csv(f'{data_path}/u.item', sep='|', encoding='latin-1', header=None,
                                    names=['movie_id', 'title', 'release_date', 'video_release', 'imdb_url'] + genre_cols,
                                    usecols=list(range(24)))
        df_movies = df_movies_raw[['movie_id', 'title'] + genre_cols].copy()
        df_ratings.to_csv(f'{data_path}/ratings.csv', index=False)
        df_movies.to_csv(f'{data_path}/movies_with_genres.csv', index=False)
        print("Converted and saved with genres")
        return df_ratings, df_movies

    print("Downloading MovieLens 100k from internet...")
    try:
        df_ratings = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/u.data',
                                 sep='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
        df_movies_url = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/u.item',
                                    sep='|', encoding='latin-1', header=None)
        genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                      'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                      'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        df_movies_url.columns = ['movie_id', 'title', 'release_date', 'video_release', 'imdb_url'] + genre_cols
        df_movies = df_movies_url[['movie_id', 'title'] + genre_cols]
        df_ratings.to_csv(f'{data_path}/ratings.csv', index=False)
        df_movies.to_csv(f'{data_path}/movies_with_genres.csv', index=False)
        print("Downloaded and saved with genres")
        return df_ratings, df_movies
    except Exception as e:
        print(f"Error: {e}")
        print("Please manually place u.data and u.item in data/ folder")
        raise


def get_genre_matrix(df_movies):
    """Return numpy array of genre flags (movies x 19)"""
    genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                  'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    return df_movies[genre_cols].values.astype(np.float32)
