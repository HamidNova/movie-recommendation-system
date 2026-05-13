import pandas as pd
import numpy as np
from src.data_loader import load_movielens, get_genre_matrix

def test_load_movielens():
    df_ratings, df_movies = load_movielens()
    assert df_ratings.shape[0] > 0
    assert df_movies.shape[0] > 0
    assert 'rating' in df_ratings.columns
    assert 'title' in df_movies.columns

def test_get_genre_matrix():
    df_movies = pd.DataFrame({
        'movie_id': [1, 2],
        'title': ['A', 'B'],
        'unknown': [0, 1],
        'Action': [1, 0],
        # ... (simplified for test; you can fill all 19 genres with zeros)
    })
    # For simplicity, just test the function runs
    df_movies = df_movies.assign(**{g: 0 for g in ['Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']})
    mat = get_genre_matrix(df_movies)
    assert mat.shape == (2, 19)
