import os
import sys
import re
import requests
from ddgs import DDGS
from PIL import Image, ImageDraw, ImageFont

for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(var, None)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_movielens

POSTER_DIR = os.path.join(os.path.dirname(__file__), "assets", "posters")
os.makedirs(POSTER_DIR, exist_ok=True)


def parse_clean_title(raw_title):
    match = re.search(r'^(.*?)\s*\((\d{4})\)\s*$', raw_title.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return raw_title.strip(), ""


def create_placeholder_poster(file_path, title):
    try:
        width, height = 300, 450
        img = Image.new('RGB', (width, height), color=(24, 28, 36))
        draw = ImageDraw.Draw(img)

        draw.rectangle([12, 12, width - 12, height - 12], outline=(60, 70, 90), width=2)

        clean_title, year = parse_clean_title(title)

        text = f"{clean_title}\n({year})" if year else clean_title

        draw.multiline_text((width / 2, height / 2), text, fill=(200, 210, 225), align="center", anchor="mm")

        img.save(file_path, "JPEG", quality=85)
        return True
    except Exception as e:
        print(f"Error creating placeholder: {e}")
        return False


def try_download_alternative(movie_id, raw_title, file_path):
    clean_title, year = parse_clean_title(raw_title)

    queries = [
        f"{clean_title} {year} imdb movie poster",
        f"{clean_title} movie poster original"
    ]

    for q in queries:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(q, max_results=1))
                if results and 'image' in results[0]:
                    img_url = results[0]['image']

                    session = requests.Session()
                    session.trust_env = False
                    res = session.get(img_url, timeout=6, headers={'User-Agent': 'Mozilla/5.0'})

                    if res.status_code == 200 and len(res.content) > 3000:
                        with open(file_path, 'wb') as f:
                            f.write(res.content)
                        return True
        except Exception:
            continue

    return False


def main():
    print("===== Starting Missing Posters Fixer =====")
    _, df_movies = load_movielens()

    missing_movies = []

    for _, row in df_movies.iterrows():
        m_id = row['movie_id']
        f_path = os.path.join(POSTER_DIR, f"{m_id}.jpg")
        if not os.path.exists(f_path) or os.path.getsize(f_path) < 3000:
            missing_movies.append(row)

    print(f"Total missing posters: {len(missing_movies)}")

    if len(missing_movies) == 0:
        print("All posters are already downloaded and valid!")
        return

    downloaded_count = 0
    placeholder_count = 0

    for i, row in enumerate(missing_movies, 1):
        m_id = row['movie_id']
        title = row['title']
        f_path = os.path.join(POSTER_DIR, f"{m_id}.jpg")

        print(f"[{i}/{len(missing_movies)}] Processing Missing: ID {m_id} - {title[:30]}...")

        if try_download_alternative(m_id, title, f_path):
            print(f"   -> Successfully found & downloaded poster!")
            downloaded_count += 1
        else:
            print(f"   -> Poster not found. Generated custom placeholder poster.")
            create_placeholder_poster(f_path, title)
            placeholder_count += 1

    print("\n==========================================")
    print("            ALL POSTERS COMPLETED         ")
    print("==========================================")
    print(f"Downloaded real posters for: {downloaded_count}")
    print(f"Generated placeholders for: {placeholder_count}")
    print(f"Total posters ready in folder: {len(df_movies)} / {len(df_movies)}")


if __name__ == "__main__":
    main()
