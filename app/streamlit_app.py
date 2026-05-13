import streamlit as st
import sys
import os
import base64
import random
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_movielens
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender
from src.utils import load_model, handle_cold_start, get_movie_id_from_title
from app.omdb_service import OMDbService
import config

st.set_page_config(page_title="Movie Recommender Pro", layout="wide", initial_sidebar_state="expanded")

# ---------- CSS (همان نسخه پایه) ----------
st.markdown("""
<style>
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
    width: 80px;
    border-radius: 8px;
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
</style>
""", unsafe_allow_html=True)


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def icon_header(icon_file, text):
    b64 = img_to_base64(icon_file)
    # استفاده از یک خط برای جلوگیری از تورفتگی
    st.markdown(f'<div class="header-icon"><img src="data:image/png;base64,{b64}" alt="icon"><h2 style="margin:0; padding:0;">{text}</h2></div>', unsafe_allow_html=True)


# ---------- Sidebar ----------
with st.sidebar:
    st.image("pictures/movie-projector.png", width=80)
    st.title("Movie Recommender Pro")
    st.markdown("---")
    st.markdown("**Dataset:** MovieLens 100K")
    st.markdown("**Users:** 943")
    st.markdown("**Movies:** 1682")
    st.markdown("---")
    st.caption("2026 Final Year Project - Computer Engineering")
    if st.button("Clear history"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Session cleared!")


# ---------- Data & Models ----------
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
    return df_ratings, df_movies, hybrid, content_model, collab_model


df_ratings, df_movies, hybrid, content_model, collab_model = load_everything()


# ---------- OMDb Service ----------
omdb = OMDbService(config.OMDB_API_KEY) if config.OMDB_API_KEY else None

@st.cache_data
def get_movie_details(title):
    if omdb:
        return omdb.get_movie_info(title)
    return None


# ---------- Session State ----------
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_recs' not in st.session_state:
    st.session_state.last_recs = None
if 'last_user' not in st.session_state:
    st.session_state.last_user = None

def add_to_history(user_id, recs):
    st.session_state.history.append({'user_id': user_id, 'recs': recs})


# ---------- تابع کمکی برای ساخت کارت فیلم ----------
def movie_card_html(movie_dict):
    title = movie_dict.get('title', '')
    confidence = movie_dict.get('confidence', 0.0)
    reasons = movie_dict.get('reasons', [])
    year = movie_dict.get('year', '')
    genre = movie_dict.get('genre', '')
    poster_url = movie_dict.get('poster', '')

    poster_tag = f'<img class="movie-poster" src="{poster_url}" />' if poster_url else ''
    year_str = f'<small>({year})</small>' if year else ''
    genre_str = f'<div style="font-size:0.8em; color:#636e72;">{genre}</div>' if genre else ''
    reasons_html = ''.join(f'<span class="reason-tag">{r}</span>' for r in reasons)

    # ساخت HTML بدون هیچ تورفتگی در ابتدای خطوط
    html = ''
    html += '<div class="movie-card">'
    html += poster_tag
    html += '<div class="movie-info">'
    html += f'<div class="movie-title">{title} {year_str}</div>'
    html += genre_str
    html += f'<div class="movie-score">Confidence: {confidence:.3f}</div>'
    html += '<div class="confidence-bar">'
    html += f'<div class="confidence-fill" style="width: {confidence*100:.0f}%;"></div>'
    html += '</div>'
    html += f'<div>{reasons_html}</div>'
    html += '</div>'
    html += '</div>'
    return html


# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["Similar Movies", "Personalized For You", "History", "About"])

# ---------- Tab 1: Similar Movies ----------
with tab1:
    icon_header("pictures/icons/search_icon.png", "Find Movies Similar to Your Favorite")
    st.markdown("Type a movie name or select from suggestions to get similar films.")

    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("Search movie by title:", placeholder="e.g., Toy Story, Star Wars...", key="sim_search")
        if search_term:
            filtered = df_movies[df_movies['title'].str.contains(search_term, case=False, na=False)]
            if not filtered.empty:
                options = filtered[['movie_id', 'title']].values.tolist()
                selected = st.selectbox("Select matching movie:", options=options, format_func=lambda x: x[1], key="sim_select")
                if selected:
                    selected_movie_id = selected[0]
                    selected_title = selected[1]
                else:
                    selected_movie_id = None
            else:
                st.warning("No movies matched your search.")
                selected_movie_id = None
        else:
            st.info("Start typing to search for a movie.")
            selected_movie_id = None

    with col2:
        n_sim = st.slider("Number of results:", 3, 10, 5, key="n_sim")
        content_weight_sim = st.slider("Content vs Collab weight (0=collab, 1=content):", 0.0, 1.0, 0.6, key="cw_sim")

    if st.button("Find Similar Movies", type="primary", disabled=(selected_movie_id is None)):
        with st.spinner("Finding similar movies..."):
            similar = hybrid.recommend_similar_movies(selected_movie_id, n=n_sim, content_weight=content_weight_sim)

        if similar:
            st.success(f"Top {n_sim} similar to **{selected_title}**:")
            for m in similar:
                details = get_movie_details(m['title'])
                card_data = {
                    'title': m['title'],
                    'confidence': m['confidence'],
                    'reasons': m['reasons'],
                    'year': details.get('year', '') if details else '',
                    'genre': details.get('genre', '') if details else '',
                    'poster': details['poster'] if (details and details.get('poster')) else ''
                }
                # نمایش کارت با استفاده از HTML خالص و بدون تورفتگی
                st.markdown(movie_card_html(card_data), unsafe_allow_html=True)
        else:
            st.warning("No similar movies found. Try different weight.")


# ---------- Tab 2: Personalized Recommendations ----------
with tab2:
    icon_header("pictures/icons/user_icon.png", "Get Personalized Recommendations")
    st.markdown("Tell us your user ID and a movie you like to get hybrid suggestions.")

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col1:
        user_id = st.number_input("Your User ID (1-943):", min_value=1, max_value=943, value=1, step=1, key="user_id")
    with col2:
        search_term_user = st.text_input("Search your favorite movie:", placeholder="Type a title...", key="user_fav_search")
        if search_term_user:
            filtered_user = df_movies[df_movies['title'].str.contains(search_term_user, case=False, na=False)]
            if not filtered_user.empty:
                user_options = filtered_user[['movie_id', 'title']].values.tolist()
                selected_user = st.selectbox("Select movie:", options=user_options, format_func=lambda x: x[1], key="user_select")
                if selected_user:
                    selected_movie_id_user = selected_user[0]
                    selected_movie_title_user = selected_user[1]
                else:
                    selected_movie_id_user = None
            else:
                st.warning("No matches.")
                selected_movie_id_user = None
        else:
            selected_movie_id_user = None

    with col3:
        n_recs = st.slider("Number of recommendations:", 3, 10, 5, key="n_rec")
        content_weight_rec = st.slider("Content weight (0=collab, 1=content):", 0.0, 1.0, 0.4, key="cw_rec")

    # Surprise Me
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

    if st.button("Get Recommendations", type="primary", disabled=(selected_movie_id_user is None)):
        cold_recs = handle_cold_start(user_id, df_ratings, df_movies, n=n_recs)
        if cold_recs:
            st.subheader("New user! Enjoy these popular movies:")
            for title in cold_recs:
                card = {'title': title, 'confidence': 0, 'reasons': []}
                st.markdown(movie_card_html(card), unsafe_allow_html=True)
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
                    details = get_movie_details(r['title'])
                    card_data = {
                        'title': r['title'],
                        'confidence': r['confidence'],
                        'reasons': r['reasons'],
                        'year': details.get('year', '') if details else '',
                        'genre': details.get('genre', '') if details else '',
                        'poster': details['poster'] if (details and details.get('poster')) else ''
                    }
                    st.markdown(movie_card_html(card_data), unsafe_allow_html=True)

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


# ---------- Tab 3: History ----------
with tab3:
    icon_header("pictures/icons/history_icon.png", "Recommendation History")
    if not st.session_state.history:
        st.info("No history yet. Get some personalized recommendations first.")
    else:
        for i, entry in enumerate(st.session_state.history[::-1]):
            st.write(f"**Session {len(st.session_state.history)-i}: User #{entry['user_id']}**")
            for r in entry['recs']:
                st.markdown(f"- {r['title']}  (confidence: {r['confidence']})")
            st.markdown("---")


# ---------- Tab 4: About ----------
with tab4:
    icon_header("pictures/icons/info_icon.png", "About the Project")
    st.markdown("""
    **Advanced Hybrid Movie Recommender System**  
    *Final Year B.Sc. Project - Computer Engineering*

    ---
    ### Features
    - **Content-Based:** Uses movie genres (19 categories) and title text (TF‑IDF similarity)
    - **Collaborative Filtering:** ALS (Alternating Least Squares) with confidence weighting
    - **Neural Collaborative Filtering:** Deep learning model for user-item interactions
    - **SVD from Scratch:** Manual matrix factorization implementation
    - **Hybrid Engine:** Adjustable combination of both methods with explainable reasons
    - **Cold-Start Handling:** Bayesian average for new users
    - **Evaluation Metrics:** Precision@k, Recall@k, NDCG, Coverage, Novelty, Diversity
    - **Visual Analytics:** 5 exploratory plots and model comparison chart
    - **Interactive UI:** Streamlit app with search, history, surprise mode, and movie details via OMDb
    """)
    st.image("reports/figures/rating_distribution.png", caption="Rating Distribution", width='stretch')

st.markdown("---")
st.caption("Built with Streamlit - 2026 Computer Engineering Final Project")
