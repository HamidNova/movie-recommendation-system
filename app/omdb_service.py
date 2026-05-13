import requests
import logging

logger = logging.getLogger(__name__)

class OMDbService:
    """Fetch movie details from OMDb API."""
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://www.omdbapi.com/"

    def get_movie_info(self, title, year=None):
        """Get movie info by title (optionally year)."""
        params = {'t': title, 'apikey': self.api_key}
        if year:
            params['y'] = year
        try:
            resp = requests.get(self.base_url, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('Response') == 'True':
                    return {
                        'title': data.get('Title'),
                        'year': data.get('Year'),
                        'genre': data.get('Genre'),
                        'poster': data.get('Poster') if data.get('Poster') != 'N/A' else None,
                        'plot': data.get('Plot'),
                        'director': data.get('Director'),
                        'actors': data.get('Actors'),
                        'imdb_rating': data.get('imdbRating')
                    }
            return None
        except Exception as e:
            logger.warning(f"OMDb API call failed: {e}")
            return None
