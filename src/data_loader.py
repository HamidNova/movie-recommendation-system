import pandas as pd
import os
import numpy as np

def load_movielens(data_path='data/'):
    os.makedirs(data_path, exist_ok=True)

    # --- حالت اول: فایل‌های تبدیل‌شده (با ستون user_id و movies_with_genres) ---
    ratings_final_path = f'{data_path}/ratings.csv'
    movies_final_path = f'{data_path}/movies_with_genres.csv'
    if os.path.exists(ratings_final_path) and os.path.exists(movies_final_path):
        header = pd.read_csv(ratings_final_path, nrows=0).columns.tolist()
        if 'user_id' in header:
            df_ratings = pd.read_csv(ratings_final_path)
            df_movies = pd.read_csv(movies_final_path)
            print("Data loaded from local csv files (including genres)")
            return df_ratings, df_movies

    # --- حالت جدید: MovieLens 1M (فایل‌های .dat) ---
    if os.path.exists(f'{data_path}/ratings.dat') and os.path.exists(f'{data_path}/movies.dat'):
        print("Detected MovieLens 1M format (.dat files). Converting...")
        df_ratings = pd.read_csv(f'{data_path}/ratings.dat', sep='::', engine='python',
                                 names=['user_id', 'movie_id', 'rating', 'timestamp'])
        df_movies_raw = pd.read_csv(f'{data_path}/movies.dat', sep='::', engine='python',
                                    encoding='latin-1',
                                    names=['movie_id', 'title', 'genres'])

        # تبدیل ژانرها به ماتریس ۱۹ تایی
        genre_list = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                      'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                      'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        genres_binary = np.zeros((len(df_movies_raw), 19), dtype=np.float32)
        for i, row in df_movies_raw.iterrows():
            movie_genres = row['genres'].split('|') if isinstance(row['genres'], str) else []
            found = False
            for g in movie_genres:
                if g in genre_list:
                    j = genre_list.index(g) + 1  # ستون 0 برای unknown
                    genres_binary[i, j] = 1.0
                    found = True
            if not found:
                genres_binary[i, 0] = 1.0

        genre_cols = ['unknown'] + genre_list
        df_genres = pd.DataFrame(genres_binary, columns=genre_cols, dtype=np.float32)
        df_movies = pd.concat([df_movies_raw[['movie_id', 'title']], df_genres], axis=1)

        # ذخیره‌سازی برای دفعات بعد
        df_ratings.to_csv(f'{data_path}/ratings.csv', index=False)
        df_movies.to_csv(f'{data_path}/movies_with_genres.csv', index=False)
        print("Movies data converted and saved as movies_with_genres.csv")
        return df_ratings, df_movies

    # --- حالت سوم: دیتاست 32M خام (movies.csv و ratings.csv با ستون‌های userId/movieId) ---
    if os.path.exists(f'{data_path}/ratings.csv') and os.path.exists(f'{data_path}/movies.csv'):
        print("Detected MovieLens 32M format. Converting genres and aligning columns...")
        df_ratings = pd.read_csv(f'{data_path}/ratings.csv')
        df_movies_raw = pd.read_csv(f'{data_path}/movies.csv')

        if 'userId' in df_ratings.columns:
            df_ratings.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'}, inplace=True)
        if 'movieId' in df_movies_raw.columns:
            df_movies_raw.rename(columns={'movieId': 'movie_id'}, inplace=True)

        genre_list = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                      'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                      'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        genres_binary = np.zeros((len(df_movies_raw), 19), dtype=np.float32)
        for i, row in df_movies_raw.iterrows():
            movie_genres = row['genres'].split('|') if isinstance(row['genres'], str) else []
            found = False
            for g in movie_genres:
                if g in genre_list:
                    j = genre_list.index(g) + 1
                    genres_binary[i, j] = 1.0
                    found = True
            if not found:
                genres_binary[i, 0] = 1.0

        genre_cols = ['unknown'] + genre_list
        df_genres = pd.DataFrame(genres_binary, columns=genre_cols, dtype=np.float32)
        df_movies = pd.concat([df_movies_raw[['movie_id', 'title']], df_genres], axis=1)

        df_ratings.to_csv(f'{data_path}/ratings.csv', index=False)
        df_movies.to_csv(f'{data_path}/movies_with_genres.csv', index=False)
        print("Movies data converted and saved as movies_with_genres.csv")
        return df_ratings, df_movies

    # --- حالت چهارم: دیتاست قدیمی 100K با u.data و u.item ---
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

    # --- حالت پنجم: دانلود از اینترنت (100K) ---
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
    """Return numpy array of genre flags (movies × 19)"""
    genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                  'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    return df_movies[genre_cols].values.astype(np.float32)
