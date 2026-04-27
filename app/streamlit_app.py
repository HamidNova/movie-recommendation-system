# app/streamlit_app.py
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_movielens
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender
from src.utils import load_model, handle_cold_start, get_movie_id_from_title
import config
import pandas as pd
import numpy as np

st.set_page_config(page_title="🎬 Movie Recommender Pro", layout="wide", initial_sidebar_state="expanded")

# ---------- Custom CSS for cards and styling ----------
st.markdown("""
<style>
.movie-card {
    background: #f0f2f6;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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
.search-box {
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.image("pictures\movie-projector.png", width=80)
    st.title("Movie Recommender Pro")
    st.markdown("---")
    st.markdown("**📌 Dataset:** MovieLens 100K")
    st.markdown("**👥 Users:** 943")
    st.markdown("**🎞 Movies:** 1682")
    st.markdown("---")
    st.caption("© 2026 Final Year Project · Computer Engineering")
    if st.button("✨ Clear history"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Session cleared!")

# ---------- Data loading (cached) ----------
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

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🔍 Similar Movies", "👤 Personalized For You", "📊 About"])

# ---------- Tab 1: Similar Movies ----------
with tab1:
    st.header("Find Movies Similar to Your Favorite")
    st.markdown("Type a movie name or select from suggestions to get similar films.")

    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("Search movie by title:", placeholder="e.g., Toy Story, Star Wars...",
                                    key="sim_search")
        if search_term:
            filtered = df_movies[df_movies['title'].str.contains(search_term, case=False, na=False)]
            if not filtered.empty:
                options = filtered[['movie_id', 'title']].values.tolist()
                selected = st.selectbox("Select matching movie:",
                                        options=options,
                                        format_func=lambda x: x[1],
                                        key="sim_select")
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

    if st.button("🚀 Find Similar Movies", type="primary", disabled=(selected_movie_id is None)):
        with st.spinner("Finding similar movies..."):
            similar = hybrid.recommend_similar_movies(selected_movie_id, n=n_sim, content_weight=content_weight_sim)

        if similar:
            st.success(f"Top {n_sim} similar to **{selected_title}**:")
            for i, m in enumerate(similar, 1):
                # Card layout
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">🎬 {m['title']}</div>
                    <div class="movie-score">Confidence: {m['confidence']:.3f}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {m['confidence']*100:.0f}%;"></div>
                    </div>
                    <div>
                        {''.join(f'<span class="reason-tag">{reason}</span>' for reason in m['reasons'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No similar movies found. Try different weight.")

# ---------- Tab 2: Personalized Recommendations ----------
with tab2:
    st.header("Get Personalized Recommendations")
    st.markdown("Tell us your user ID and a movie you like to get hybrid suggestions.")

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col1:
        user_id = st.number_input("Your User ID (1-943):", min_value=1, max_value=943, value=1, step=1,
                                  key="user_id")
    with col2:
        search_term_user = st.text_input("Search your favorite movie:", placeholder="Type a title...",
                                         key="user_fav_search")
        if search_term_user:
            filtered_user = df_movies[df_movies['title'].str.contains(search_term_user, case=False, na=False)]
            if not filtered_user.empty:
                user_options = filtered_user[['movie_id', 'title']].values.tolist()
                selected_user = st.selectbox("Select movie:",
                                             options=user_options,
                                             format_func=lambda x: x[1],
                                             key="user_select")
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

    if st.button("🔥 Get Recommendations", type="primary", disabled=(selected_movie_id_user is None)):
        # Cold-start check
        cold_recs = handle_cold_start(user_id, df_ratings, df_movies, n=n_recs)
        if cold_recs:
            st.subheader("📌 New user! Enjoy these popular movies:")
            for i, title in enumerate(cold_recs, 1):
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">🎬 {title}</div>
                    <div class="movie-score">(popular pick)</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.spinner("Generating your personalized list..."):
                recs = hybrid.recommend_by_movie_id(user_id, selected_movie_id_user, n=n_recs,
                                                    content_weight=content_weight_rec)
            if recs:
                st.subheader(f"🎯 Top picks for User #{user_id}")
                # Save to session state for feedback
                st.session_state['last_recs'] = recs
                st.session_state['last_user'] = user_id

                for i, r in enumerate(recs, 1):
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">🎬 {r['title']}</div>
                        <div class="movie-score">Confidence: {r['confidence']:.3f}</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {r['confidence']*100:.0f}%;"></div>
                        </div>
                        <div>
                            {''.join(f'<span class="reason-tag">{reason}</span>' for reason in r['reasons'])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Feedback buttons (cosmetic, logged in session)
                st.markdown("**How was this recommendation?**")
                col_fb1, col_fb2, _ = st.columns([1, 1, 4])
                if col_fb1.button("👍 Like", key="like"):
                    st.session_state['feedback'] = 'liked'
                    st.success("Thanks! We'll keep that in mind.")
                if col_fb2.button("👎 Dislike", key="dislike"):
                    st.session_state['feedback'] = 'disliked'
                    st.info("Thanks for your feedback.")
                if 'feedback' in st.session_state:
                    st.caption(f"Your feedback: {st.session_state['feedback']}")
            else:
                st.warning("Could not generate recommendations. Try a different movie or user ID.")

# ---------- Tab 3: About ----------
with tab3:
    st.header("About the Project")
    st.markdown("""
    **🎓 Advanced Hybrid Movie Recommender System**  
    *Final Year B.Sc. Project – Computer Engineering*

    ---
    ### 🔬 Features
    - **Content-Based:** Uses movie genres (19 categories) and title text (TF‑IDF similarity)
    - **Collaborative Filtering:** ALS (Alternating Least Squares) with confidence weighting
    - **Hybrid Engine:** Adjustable combination of both methods with explainable reasons
    - **Cold-Start Handling:** Bayesian average for new users
    - **Evaluation Metrics:** Precision@k, Recall@k, NDCG, Coverage, Novelty, Diversity
    - **Visual Analytics:** 5 exploratory plots and model comparison chart

    ### 📁 Dataset
    MovieLens 100K – 100,000 ratings (1-5) from 943 users on 1,682 movies.

    ### 🛠 Tech Stack
    Python, Streamlit, scikit-learn, Implicit (ALS), Pandas, NumPy, Matplotlib, Seaborn

    ### 📂 Project Structure
    Modular design with separate modules for data loading, modeling, evaluation, visualization, and UI.
    """)
    st.image("reports/figures/rating_distribution.png", caption="Rating Distribution", use_container_width=True)

st.markdown("---")
st.caption("🚀 Built with Streamlit · © 2026 Computer Engineering Final Project")
