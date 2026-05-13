# download_posters.py
import os
import requests
import pandas as pd
from src.data_loader import load_movielens
import config

OMDB_API_KEY = config.OMDB_API_KEY
POSTER_DIR = "pictures/posters"

def download_poster(movie_title, year=None):
    """Download poster from OMDb and save to pictures/posters/."""
    if not OMDB_API_KEY or OMDB_API_KEY == "your_omdb_api_key_here":
        print("OMDB API key not set in config.py")
        return False

    base_url = "http://www.omdbapi.com/"
    params = {'t': movie_title, 'apikey': OMDB_API_KEY}
    if year:
        params['y'] = year

    try:
        resp = requests.get(base_url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('Response') == 'True' and data.get('Poster') and data['Poster'] != 'N/A':
                poster_url = data['Poster']
                img_resp = requests.get(poster_url, timeout=10)
                if img_resp.status_code == 200:
                    safe_name = movie_title.replace('/', '_').replace(':', '_').replace('?', '').strip()
                    file_path = os.path.join(POSTER_DIR, f"{safe_name}.jpg")
                    with open(file_path, 'wb') as f:
                        f.write(img_resp.content)
                    print(f"Downloaded: {movie_title}")
                    return True
    except Exception as e:
        print(f"Failed for {movie_title}: {e}")
    return False

if __name__ == "__main__":
    os.makedirs(POSTER_DIR, exist_ok=True)
    _, df_movies = load_movielens()
    titles = df_movies['title'].tolist()
    for idx, title in enumerate(titles, 1):
        print(f"({idx}/{len(titles)}) Processing {title}...")
        download_poster(title)
