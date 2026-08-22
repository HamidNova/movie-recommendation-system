import streamlit as st
import sys
import os
import base64
import random
import re
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import load_movielens
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender
from src.fuzzy_clustering import FuzzyClustering
from src.utils import load_model, handle_cold_start, get_movie_id_from_title

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSTER_DIR = os.path.join(BASE_DIR, "assets", "posters")

st.set_page_config(page_title="Movie Recommender Pro", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.movie-card {
    background: #f0f2f6;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    display: flex;
    gap: 16px;
    align-items: center;
}
.movie-poster {
    width: 90px;
    height: 135px;
    object-fit: cover;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}
.movie-info {
    flex: 1;
}
.movie-title {
    font-size: 1.3em;
    font-weight: bold;
    color: #1f3a6b;
}
.movie-score {
    font-size: 0.9em;
    color: #2c3e50;
    margin-top: 4px;
}
.confidence-bar {
    height: 6px;
    background: #e0e0e0;
    border-radius: 3px;
    margin: 6px 0 10px;
}
.confidence-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #00b894, #0984e3);
}
.reason-tag {
    display: inline-block;
    background: #dfe6e9;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8em;
    margin-right: 6px;
    color: #2d3436;
}
.header-icon {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0;
}
.header-icon img {
    width: 28px;
    height: 28px;
}
</style>""", unsafe_allow_html=True)


def img_to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def get_poster_b64(movie_id):
    paths = [
        os.path.join(POSTER_DIR, f"{movie_id}.jpg"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "posters", f"{movie_id}.jpg"),
        os.path.join("assets", "posters", f"{movie_id}.jpg")
    ]
    for path in paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return ""


def icon_header(icon_file, text):
    b64 = img_to_base64(icon_file)
    if b64:
        st.markdown(f'<div class="header-icon"><img src="data:image/png;base64,{b64}" alt="icon"><h2 style="margin:0; padding:0;">{text}</h2></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h2>{text}</h2>', unsafe_allow_html=True)


@st.cache_resource
def load_everything():
    df_ratings, df_movies = load_movielens()
    try:
        content_model = load_model('content_based')
        collab_model = load_model('collaborative')
    except Exception as e:
        st.error(f"Models not found. Please run main.py first. Error: {e}")
        st.stop()

    hybrid = HybridRecommender(content_model, collab_model, df_ratings, df_movies)

    # بردارهای نهفته فیلم‌ها (n_items, n_factors)
    item_factors = np.asarray(collab_model.item_factors)

    # خوشه‌بندی فازی روی همین بردارها (بدون ترانهاده)
    fcm = FuzzyClustering(
        n_clusters=config.FCM_PARAMS['n_clusters'],
        m=config.FCM_PARAMS['m'],
        max_iter=config.FCM_PARAMS['max_iter'],
        error=config.FCM_PARAMS['error'],
        random_state=config.FCM_PARAMS['random_state']
    )
    fcm.fit(item_factors)

    return df_ratings, df_movies, hybrid, content_model, collab_model, fcm


df_ratings, df_movies, hybrid, content_model, collab_model, fuzzy_model = load_everything()

n_users = df_ratings['user_id'].nunique()
n_movies = df_movies['movie_id'].nunique()
max_user_id = int(df_ratings['user_id'].max())
total_ratings = len(df_ratings)


def get_movie_meta(identifier):
    if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
        row = df_movies[df_movies['movie_id'] == int(identifier)]
    else:
        row = df_movies[df_movies['title'] == identifier]

    if not row.empty:
        m_id = int(row.iloc[0]['movie_id'])
        full_title = str(row.iloc[0]['title'])
        genres = str(row.iloc[0].get('genres', '')).replace('|', ', ')
        match = re.search(r'\((\d{4})\)', full_title)
        year = match.group(1) if match else ''
        clean_title = re.sub(r'\s*\(\d{4}\)', '', full_title).strip()
        return m_id, clean_title, year, genres
    return identifier, str(identifier), '', ''


def get_fuzzy_info(movie_id):
    if movie_id not in collab_model.item_map:
        return None
    idx = collab_model.item_map[movie_id]
    membership = fuzzy_model.membership[idx]
    dominant = int(np.argmax(membership))
    memberships = {f"Cluster {i}": float(v) for i, v in enumerate(membership)}
    return {
        "dominant_cluster": dominant,
        "memberships": memberships,
        "top_cluster_pct": float(membership[dominant])
    }


def movie_card_html(movie_dict, fuzzy_info=None):
    title = movie_dict.get('title', '')
    confidence = movie_dict.get('confidence', 0.0)
    reasons = movie_dict.get('reasons', [])
    year = movie_dict.get('year', '')
    genre = movie_dict.get('genre', '')
    poster_b64 = movie_dict.get('poster_b64', '')

    poster_tag = f'<img class="movie-poster" src="data:image/jpeg;base64,{poster_b64}" alt="{title}" />' if poster_b64 else ''
    year_str = f'<small>({year})</small>' if year else ''
    genre_str = f'<div style="font-size:0.8em; color:#636e72;">{genre}</div>' if genre else ''
    reasons_html = ''.join(f'<span class="reason-tag">{r}</span>' for r in reasons)

    fuzzy_html = ''
    if fuzzy_info:
        dom = fuzzy_info['dominant_cluster']
        pct = fuzzy_info['memberships'].get(f'Cluster {dom}', 0)
        top_clusters = sorted(fuzzy_info['memberships'].items(), key=lambda x: x[1], reverse=True)[:3]
        bars = ''
        for cluster_name, val in top_clusters:
            val_pct = int(val * 100)
            bars += (
                f'<div style="font-size:0.8em; margin:2px 0; color:#2c3e50;">{cluster_name}: {val:.1%}</div>'
                f'<div style="background:#e0e0e0; height:5px; border-radius:3px;">'
                f'<div style="width:{val_pct}%; background:#6c5ce7; height:5px; border-radius:3px;"></div>'
                f'</div>'
            )
        fuzzy_html = (
            f'<div style="margin-top:8px; color:#2c3e50;">'
            f'<span style="font-weight:bold;">Fuzzy Cluster:</span> Cluster {dom} ({pct:.1%})<br>'
            f'{bars}'
            f'</div>'
        )

    html = (
        f'<div class="movie-card">'
        f'{poster_tag}'
        f'<div class="movie-info">'
        f'<div class="movie-title">{title} {year_str}</div>'
        f'{genre_str}'
        f'<div class="movie-score">Confidence: {confidence:.3f}</div>'
        f'<div class="confidence-bar">'
        f'<div class="confidence-fill" style="width: {confidence*100:.0f}%;"></div>'
        f'</div>'
        f'{fuzzy_html}'
        f'<div>{reasons_html}</div>'
        f'</div>'
        f'</div>'
    )
    return html


with st.sidebar:
    if os.path.exists("pictures/movie-projector.png"):
        st.image("pictures/movie-projector.png", width=80)
    st.title("Movie Recommender Pro")
    st.markdown("---")
    st.markdown(f"**Dataset:** MovieLens ({total_ratings:,} Ratings)")
    st.markdown(f"**Users:** {n_users:,}")
    st.markdown(f"**Movies:** {n_movies:,}")
    st.markdown("---")
    st.caption("2026 Final Year Project - Computer Engineering")
    if st.button("Clear history"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Session cleared!")

if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_recs' not in st.session_state:
    st.session_state.last_recs = None
if 'last_user' not in st.session_state:
    st.session_state.last_user = None


def add_to_history(user_id, recs):
    st.session_state.history.append({'user_id': user_id, 'recs': recs})


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Similar Movies",
    "Personalized For You",
    "History",
    "Fuzzy Clusters",
    "About"
])

with tab1:
    icon_header("pictures/icons/search_icon.png", "Find Movies Similar to Your Favorite")
    st.markdown("Type a movie name or select from suggestions to get similar films.")

    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("Search movie by title:", placeholder="e.g., Toy Story, Star Wars...", key="sim_search")
        selected_movie_id = None
        selected_title = ""
        if search_term:
            filtered = df_movies[df_movies['title'].str.contains(search_term, case=False, na=False)]
            if not filtered.empty:
                options = filtered[['movie_id', 'title']].values.tolist()
                selected = st.selectbox("Select matching movie:", options=options, format_func=lambda x: x[1], key="sim_select")
                if selected:
                    selected_movie_id = selected[0]
                    selected_title = selected[1]
            else:
                st.warning("No movies matched your search.")
        else:
            st.info("Start typing to search for a movie.")

    with col2:
        n_sim = st.slider("Number of results:", 3, 10, 5, key="n_sim")
        content_weight_sim = st.slider("Content vs Collab weight (0=collab, 1=content):", 0.0, 1.0, 0.6, key="cw_sim")

    if st.button("Find Similar Movies", type="primary", disabled=(selected_movie_id is None)):
        with st.spinner("Finding similar movies..."):
            similar = hybrid.recommend_similar_movies(selected_movie_id, n=n_sim, content_weight=content_weight_sim)

        if similar:
            st.success(f"Top {n_sim} similar to **{selected_title}**:")
            for m in similar:
                m_id = m.get('movie_id')
                if not m_id:
                    m_id, clean_title, year, genre = get_movie_meta(m['title'])
                else:
                    _, clean_title, year, genre = get_movie_meta(m_id)

                fuzzy_info = get_fuzzy_info(m_id)
                card_data = {
                    'title': clean_title,
                    'confidence': m['confidence'],
                    'reasons': m['reasons'],
                    'year': year,
                    'genre': genre,
                    'poster_b64': get_poster_b64(m_id)
                }
                st.markdown(movie_card_html(card_data, fuzzy_info), unsafe_allow_html=True)
        else:
            st.warning("No similar movies found. Try different weight.")

with tab2:
    icon_header("pictures/icons/user_icon.png", "Get Personalized Recommendations")
    st.markdown("Tell us your user ID and a movie you like to get hybrid suggestions.")

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col1:
        user_id = st.number_input(
            f"Your User ID (1-{max_user_id}):",
            min_value=1,
            max_value=max_user_id,
            value=1632 if max_user_id >= 1632 else 1,
            step=1,
            key="user_id"
        )
    with col2:
        search_term_user = st.text_input("Search your favorite movie:", placeholder="Type a title...", key="user_fav_search")
        selected_movie_id_user = None
        selected_movie_title_user = ""
        if search_term_user:
            filtered_user = df_movies[df_movies['title'].str.contains(search_term_user, case=False, na=False)]
            if not filtered_user.empty:
                user_options = filtered_user[['movie_id', 'title']].values.tolist()
                selected_user = st.selectbox("Select movie:", options=user_options, format_func=lambda x: x[1], key="user_select")
                if selected_user:
                    selected_movie_id_user = selected_user[0]
                    selected_movie_title_user = selected_user[1]
            else:
                st.warning("No matches.")

    with col3:
        n_recs = st.slider("Number of recommendations:", 3, 10, 5, key="n_rec")
        content_weight_rec = st.slider("Content weight (0=collab, 1=content):", 0.0, 1.0, 0.4, key="cw_rec")

    def surprise_movie():
        movie_stats = df_ratings.groupby('movie_id')['rating'].agg(['mean', 'count'])
        high_rated = movie_stats[movie_stats['mean'] >= 4.0]
        if high_rated.empty:
            return None
        threshold = movie_stats['count'].quantile(0.25)
        niche = high_rated[high_rated['count'] <= threshold]
        candidates = niche if not niche.empty else high_rated
        mid = random.choice(candidates.index)
        title = df_movies[df_movies['movie_id'] == mid]['title'].values[0]
        return mid, title

    col_surprise, _ = st.columns([1, 3])
    if col_surprise.button("Surprise Me", type="secondary"):
        surprise = surprise_movie()
        if surprise:
            mid, title = surprise
            st.session_state.surprise_movie_id = mid
            st.session_state.surprise_title = title
            st.success(f"Today's surprise: **{title}**")
        else:
            st.warning("No suitable surprise found.")

    if st.button("Get Recommendations", type="primary", disabled=(selected_movie_id_user is None and 'surprise_movie_id' not in st.session_state)):
        cold_recs = handle_cold_start(user_id, df_ratings, df_movies, n=n_recs)
        if cold_recs:
            st.subheader("New user! Enjoy these popular movies:")
            for title_item in cold_recs:
                m_id, clean_title, year, genre = get_movie_meta(title_item)
                fuzzy_info = get_fuzzy_info(m_id)
                card = {
                    'title': clean_title,
                    'confidence': 0,
                    'reasons': ['Cold-start Popularity'],
                    'year': year,
                    'genre': genre,
                    'poster_b64': get_poster_b64(m_id)
                }
                st.markdown(movie_card_html(card, fuzzy_info), unsafe_allow_html=True)
        else:
            movie_to_use = selected_movie_id_user
            if st.session_state.get('surprise_movie_id'):
                movie_to_use = st.session_state.surprise_movie_id

            with st.spinner("Generating your personalized list..."):
                recs = hybrid.recommend_by_movie_id(user_id, movie_to_use, n=n_recs, content_weight=content_weight_rec)
            if recs:
                st.session_state.last_recs = recs
                st.session_state.last_user = user_id
                add_to_history(user_id, recs)

                st.subheader(f"Top picks for User #{user_id}")
                for r in recs:
                    m_id = r.get('movie_id')
                    if not m_id:
                        m_id, clean_title, year, genre = get_movie_meta(r['title'])
                    else:
                        _, clean_title, year, genre = get_movie_meta(m_id)

                    fuzzy_info = get_fuzzy_info(m_id)
                    card_data = {
                        'title': clean_title,
                        'confidence': r['confidence'],
                        'reasons': r['reasons'],
                        'year': year,
                        'genre': genre,
                        'poster_b64': get_poster_b64(m_id)
                    }
                    st.markdown(movie_card_html(card_data, fuzzy_info), unsafe_allow_html=True)

                st.markdown("**How was this recommendation?**")
                col_fb1, col_fb2, _ = st.columns([1, 1, 4])
                if col_fb1.button("Like", key="like"):
                    st.session_state['feedback'] = 'liked'
                    st.success("Thanks! We'll keep that in mind.")
                if col_fb2.button("Dislike", key="dislike"):
                    st.session_state['feedback'] = 'disliked'
                    st.info("Thanks for your feedback.")
                if 'feedback' in st.session_state:
                    st.caption(f"Your feedback: {st.session_state['feedback']}")
            else:
                st.warning("Could not generate recommendations. Try a different movie or user ID.")

with tab3:
    icon_header("pictures/icons/history_icon.png", "Recommendation History")
    if not st.session_state.history:
        st.info("No history yet. Get some personalized recommendations first.")
    else:
        for i, entry in enumerate(st.session_state.history[::-1]):
            st.write(f"**Session {len(st.session_state.history)-i}: User #{entry['user_id']}**")
            for r in entry['recs']:
                st.markdown(f"- {r['title']}  (confidence: {r['confidence']:.3f})")
            st.markdown("---")

with tab4:
    icon_header("pictures/icons/search_icon.png", "Fuzzy Clusters")

    st.markdown("### Fuzzy C-Means Clusters (based on ALS item factors)")
    n_clusters = fuzzy_model.membership.shape[1]
    st.write(f"Number of clusters: **{n_clusters}**")

    top_n = st.slider("Top movies per cluster:", 3, 10, 5, key="top_cluster")

    cluster_tabs = st.tabs([f"Cluster {i}" for i in range(n_clusters)])
    for c in range(n_clusters):
        with cluster_tabs[c]:
            cluster_scores = fuzzy_model.membership[:, c]
            top_indices = np.argsort(cluster_scores)[::-1][:top_n]
            rows = []
            for idx in top_indices:
                mid = collab_model.item_reverse[idx]
                title = df_movies[df_movies['movie_id'] == mid]['title'].values[0] if not df_movies[df_movies['movie_id'] == mid].empty else str(mid)
                score = float(cluster_scores[idx])
                rows.append({"Movie": title, "Membership": round(score, 3)})
            st.dataframe(pd.DataFrame(rows))

    st.markdown("### Borderline / Mixed Movies")
    membership = fuzzy_model.membership
    top2 = np.sort(membership, axis=1)[:, -2:]
    diff = top2[:, 1] - top2[:, 0]
    border_indices = np.argsort(diff)[:10]

    border_rows = []
    for idx in border_indices:
        mid = collab_model.item_reverse[idx]
        title = df_movies[df_movies['movie_id'] == mid]['title'].values[0] if not df_movies[
            df_movies['movie_id'] == mid].empty else str(mid)
        mem = membership[idx]
        top_clusters = np.argsort(mem)[::-1][:2]
        border_rows.append({
            "Movie": title,
            "Top Cluster": f"Cluster {top_clusters[0]}",
            "Top Membership": round(float(mem[top_clusters[0]]), 4),
            "Second Cluster": f"Cluster {top_clusters[1]}",
            "Second Membership": round(float(mem[top_clusters[1]]), 4),
            "Difference": round(float(diff[idx]), 5)
        })

    st.dataframe(pd.DataFrame(border_rows))

with tab5:
    icon_header("pictures/icons/info_icon.png", "About the Project")
    st.markdown(f"""
    **Advanced Hybrid Movie Recommender System**  
    *Final Year B.Sc. Project - Computer Engineering*

    ---
    ### Dataset Overview
    - **Total Ratings:** {total_ratings:,}
    - **Total Users:** {n_users:,}
    - **Total Movies:** {n_movies:,}

    ---
    ### Features
    - **Content-Based:** Uses movie genres and title text (TF-IDF similarity)
    - **Collaborative Filtering:** ALS (Alternating Least Squares) with confidence weighting
    - **Neural Hybrid Recommender:** Deep learning model combining collaborative embeddings and fuzzy memberships
    - **Hybrid Engine:** Adjustable combination of both methods with explainable reasons
    - **Fuzzy C-Means Clustering:** Visualized in a separate tab with top and borderline movies
    - **Cold-Start Handling:** Bayesian average for new users
    - **Evaluation Metrics:** Precision@k, Recall@k, NDCG, Coverage, Novelty, Diversity
    - **Interactive UI:** Streamlit app with search, local poster gallery, history, and surprise mode
    """)
    if os.path.exists("reports/figures/rating_distribution.png"):
        st.image("reports/figures/rating_distribution.png", caption="Rating Distribution", use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit - 2026 Computer Engineering Student (Hamidreza Mirzaei) - Final Project")
