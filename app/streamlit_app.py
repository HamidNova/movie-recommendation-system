import streamlit as st
import sys
import os
import random
import re

import numpy as np
import pandas as pd


# ============================================================
# Project Path
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    BASE_DIR
)


# ============================================================
# Project Imports
# ============================================================

import config

from src.data_loader import load_movielens
from src.hybrid import HybridRecommender

from src.utils import (
    load_model,
    handle_cold_start,
)


# ============================================================
# Paths
# ============================================================

POSTER_DIR = os.path.join(
    BASE_DIR,
    "assets",
    "posters",
)


# ============================================================
# Streamlit Config
# ============================================================

st.set_page_config(
    page_title="Movie Recommender Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Helper Functions
# ============================================================

def get_poster_path(movie_id):

    paths = [

        os.path.join(
            POSTER_DIR,
            f"{movie_id}.jpg",
        ),

        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            "assets",
            "posters",
            f"{movie_id}.jpg",
        ),

        os.path.join(
            "assets",
            "posters",
            f"{movie_id}.jpg",
        ),

    ]

    for path in paths:

        if os.path.exists(path):

            return path

    return None


def icon_header(
    icon_file,
    text,
):

    col1, col2 = st.columns(
        [0.08, 0.92]
    )

    with col1:

        if os.path.exists(icon_file):

            st.image(
                icon_file,
                width=40,
            )

    with col2:

        st.subheader(
            text
        )


# ============================================================
# Load Models
# ============================================================

@st.cache_resource
def load_everything():

    df_ratings, df_movies = (
        load_movielens()
    )

    try:

        content_model = load_model(
            "content_based"
        )

        collab_model = load_model(
            "collaborative"
        )

        neural_model = load_model(
            "neural_hybrid"
        )

    except Exception as error:

        st.error(
            "Models not found. "
            "Please run main.py first.\n\n"
            f"Error: {error}"
        )

        st.stop()

    hybrid = HybridRecommender(
        content_model,
        collab_model,
        df_ratings,
        df_movies,
    )

    return (
        df_ratings,
        df_movies,
        hybrid,
        content_model,
        collab_model,
        neural_model,
    )


(
    df_ratings,
    df_movies,
    hybrid,
    content_model,
    collab_model,
    neural_model,
) = load_everything()


# ============================================================
# Dataset Statistics
# ============================================================

n_users = (
    df_ratings["user_id"]
    .nunique()
)

n_movies = (
    df_movies["movie_id"]
    .nunique()
)

max_user_id = int(
    df_ratings["user_id"]
    .max()
)

total_ratings = len(
    df_ratings
)


# ============================================================
# Movie Metadata
# ============================================================

def get_movie_meta(identifier):

    if (
        isinstance(identifier, int)
        or (
            isinstance(identifier, str)
            and identifier.isdigit()
        )
    ):

        row = df_movies[
            df_movies["movie_id"]
            == int(identifier)
        ]

    else:

        row = df_movies[
            df_movies["title"]
            == identifier
        ]

    if not row.empty:

        movie_id = int(
            row.iloc[0]["movie_id"]
        )

        full_title = str(
            row.iloc[0]["title"]
        )

        genres = str(
            row.iloc[0].get(
                "genres",
                "",
            )
        ).replace(
            "|",
            ", ",
        )

        match = re.search(
            r"\((\d{4})\)",
            full_title,
        )

        year = (
            match.group(1)
            if match
            else ""
        )

        clean_title = re.sub(
            r"\s*\(\d{4}\)",
            "",
            full_title,
        ).strip()

        return (
            movie_id,
            clean_title,
            year,
            genres,
        )

    return (
        identifier,
        str(identifier),
        "",
        "",
    )


# ============================================================
# User Fuzzy Information
# ============================================================

def get_user_fuzzy_info(user_id):

    if user_id not in neural_model.user_map:

        return None

    user_index = (
        neural_model.user_map[
            user_id
        ]
    )

    membership = (
        neural_model.fuzzy_u[
            user_index
        ]
    )

    dominant_cluster = int(
        np.argmax(membership)
    )

    memberships = {

        f"Cluster {index}": float(value)

        for index, value in enumerate(
            membership
        )

    }

    return {

        "dominant_cluster":
            dominant_cluster,

        "memberships":
            memberships,

        "top_cluster_pct":
            float(
                membership[
                    dominant_cluster
                ]
            ),

    }


# ============================================================
# Movie Card
# ============================================================

def show_movie_card(
    movie_dict,
):

    title = movie_dict.get(
        "title",
        "",
    )

    confidence = float(
        movie_dict.get(
            "confidence",
            0.0,
        )
    )

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    reasons = movie_dict.get(
        "reasons",
        [],
    )

    year = movie_dict.get(
        "year",
        "",
    )

    genre = movie_dict.get(
        "genre",
        "",
    )

    poster_path = movie_dict.get(
        "poster_path",
        None,
    )

    with st.container(
        border=True
    ):

        col1, col2 = st.columns(
            [1, 4]
        )

        with col1:

            if (
                poster_path
                and os.path.exists(
                    poster_path
                )
            ):

                st.image(
                    poster_path,
                    use_container_width=True,
                )

        with col2:

            title_text = title

            if year:

                title_text += (
                    f" ({year})"
                )

            st.subheader(
                title_text
            )

            if genre:

                st.caption(
                    genre
                )

            st.write(
                f"Recommendation Score: "
                f"**{confidence:.3f}**"
            )

            st.progress(
                confidence
            )

            if reasons:

                st.caption(
                    " • ".join(
                        reasons
                    )
                )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    projector_path = os.path.join(
        BASE_DIR,
        "pictures",
        "movie-projector.png",
    )

    if os.path.exists(
        projector_path
    ):

        st.image(
            projector_path,
            width=80,
        )

    st.title(
        "Movie Recommender Pro"
    )

    st.markdown("---")

    st.markdown(
        f"""
        **Dataset:** MovieLens
        ({total_ratings:,} Ratings)
        """
    )

    st.markdown(
        f"**Users:** {n_users:,}"
    )

    st.markdown(
        f"**Movies:** {n_movies:,}"
    )

    st.markdown("---")

    st.caption(
        "2026 Final Year Project "
        "- Computer Engineering"
    )

    if st.button(
        "Clear history"
    ):

        for key in list(
            st.session_state.keys()
        ):

            del st.session_state[key]

        st.success(
            "Session cleared!"
        )


# ============================================================
# Session State
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


if "last_recs" not in st.session_state:

    st.session_state.last_recs = None


if "last_user" not in st.session_state:

    st.session_state.last_user = None


# ============================================================
# History
# ============================================================

def add_to_history(
    user_id,
    recs,
):

    st.session_state.history.append(
        {
            "user_id": user_id,
            "recs": recs,
        }
    )


# ============================================================
# Tabs
# ============================================================

(
    tab1,
    tab2,
    tab3,
    tab4,
    tab5,
) = st.tabs(

    [
        "Similar Movies",
        "Personalized For You",
        "History",
        "Fuzzy User Clusters",
        "About",
    ]

)


# ============================================================
# TAB 1
# Similar Movies
# ============================================================

with tab1:

    icon_header(
        os.path.join(
            BASE_DIR,
            "pictures",
            "icons",
            "search_icon.png",
        ),
        "Find Movies Similar to Your Favorite",
    )

    st.markdown(
        "Type a movie name or select "
        "from suggestions to get similar films."
    )

    col1, col2 = st.columns(
        [2, 1]
    )

    with col1:

        search_term = st.text_input(
            "Search movie by title:",
            placeholder=(
                "e.g., Toy Story, "
                "Star Wars..."
            ),
            key="sim_search",
        )

        selected_movie_id = None
        selected_title = ""

        if search_term:

            filtered = df_movies[
                df_movies["title"]
                .str.contains(
                    search_term,
                    case=False,
                    na=False,
                )
            ]

            if not filtered.empty:

                options = (
                    filtered[
                        [
                            "movie_id",
                            "title",
                        ]
                    ]
                    .values
                    .tolist()
                )

                selected = st.selectbox(
                    "Select matching movie:",
                    options=options,
                    format_func=lambda x: x[1],
                    key="sim_select",
                )

                if selected:

                    selected_movie_id = (
                        selected[0]
                    )

                    selected_title = (
                        selected[1]
                    )

            else:

                st.warning(
                    "No movies matched "
                    "your search."
                )

        else:

            st.info(
                "Start typing to "
                "search for a movie."
            )

    with col2:

        n_sim = st.slider(
            "Number of results:",
            3,
            10,
            5,
            key="n_sim",
        )

        content_weight_sim = st.slider(
            (
                "Content vs Collab weight "
                "(0=collab, 1=content):"
            ),
            0.0,
            1.0,
            0.6,
            key="cw_sim",
        )

    if st.button(
        "Find Similar Movies",
        type="primary",
        disabled=(
            selected_movie_id is None
        ),
    ):

        with st.spinner(
            "Finding similar movies..."
        ):

            similar = (
                hybrid.recommend_similar_movies(
                    selected_movie_id,
                    n=n_sim,
                    content_weight=(
                        content_weight_sim
                    ),
                )
            )

        if similar:

            st.success(
                f"Top {n_sim} similar to "
                f"**{selected_title}**:"
            )

            for movie in similar:

                movie_id = movie.get(
                    "movie_id"
                )

                if movie_id is None:

                    (
                        movie_id,
                        clean_title,
                        year,
                        genre,
                    ) = get_movie_meta(
                        movie["title"]
                    )

                else:

                    (
                        _,
                        clean_title,
                        year,
                        genre,
                    ) = get_movie_meta(
                        movie_id
                    )

                card_data = {

                    "title":
                        clean_title,

                    "confidence":
                        movie.get(
                            "confidence",
                            0.0,
                        ),

                    "reasons":
                        movie.get(
                            "reasons",
                            [],
                        ),

                    "year":
                        year,

                    "genre":
                        genre,

                    "poster_path":
                        get_poster_path(
                            movie_id
                        ),

                }

                show_movie_card(
                    card_data
                )

        else:

            st.warning(
                "No similar movies found. "
                "Try a different weight."
            )


# ============================================================
# TAB 2
# Neural Hybrid Personalized Recommendations
# ============================================================

with tab2:

    icon_header(
        os.path.join(
            BASE_DIR,
            "pictures",
            "icons",
            "user_icon.png",
        ),
        "Get Personalized Recommendations",
    )

    st.markdown(
        "Get recommendations using the "
        "Neural Hybrid model "
        "(ALS + Fuzzy + Neural Network)."
    )

    col1, col2 = st.columns(
        [1, 2]
    )

    with col1:

        user_id = st.number_input(
            (
                f"Your User ID "
                f"(1-{max_user_id}):"
            ),
            min_value=1,
            max_value=max_user_id,
            value=(
                1632
                if max_user_id >= 1632
                else 1
            ),
            step=1,
            key="user_id",
        )

        n_recs = st.slider(
            "Number of recommendations:",
            3,
            10,
            5,
            key="n_rec",
        )

    with col2:

        fuzzy_info = (
            get_user_fuzzy_info(
                int(user_id)
            )
        )

        if fuzzy_info:

            dominant = fuzzy_info[
                "dominant_cluster"
            ]

            percentage = fuzzy_info[
                "top_cluster_pct"
            ]

            st.info(
                f"Fuzzy User Profile: "
                f"**Cluster {dominant}** "
                f"with "
                f"**{percentage:.1%}** "
                f"membership."
            )

        else:

            st.warning(
                "This user is not available "
                "in the trained Neural Hybrid model. "
                "Cold-start recommendations will be used."
            )


    if st.button(
        "Get Recommendations",
        type="primary",
    ):

        user_id = int(user_id)

        is_known_user = (
            user_id
            in neural_model.user_map
        )

        if not is_known_user:

            with st.spinner(
                "Generating cold-start recommendations..."
            ):

                cold_recs = (
                    neural_model.recommend_for_user(
                        user_id=user_id,
                        df_ratings=df_ratings,
                        df_movies=df_movies,
                        n=n_recs,
                    )
                )

            if cold_recs:

                st.subheader(
                    "New User Recommendations"
                )

                for recommendation in cold_recs:

                    movie_id = recommendation[
                        "movie_id"
                    ]

                    (
                        _,
                        clean_title,
                        year,
                        genre,
                    ) = get_movie_meta(
                        movie_id
                    )

                    predicted_rating = (
                        recommendation.get(
                            "predicted_rating",
                            0.0,
                        )
                    )

                    card_data = {

                        "title":
                            clean_title,

                        "confidence":
                            predicted_rating / 5.0,

                        "reasons":
                            [
                                "Cold-start Popularity",
                            ],

                        "year":
                            year,

                        "genre":
                            genre,

                        "poster_path":
                            get_poster_path(
                                movie_id
                            ),

                    }

                    show_movie_card(
                        card_data
                    )

                    st.caption(
                        f"Popularity score: "
                        f"{predicted_rating:.3f}"
                    )

            else:

                st.warning(
                    "Could not generate "
                    "cold-start recommendations."
                )

        else:

            with st.spinner(
                "Generating Neural Hybrid recommendations..."
            ):

                recs = (
                    neural_model.recommend_for_user(
                        user_id=user_id,
                        df_ratings=df_ratings,
                        df_movies=df_movies,
                        n=n_recs,
                    )
                )

            if recs:

                st.session_state.last_recs = (
                    recs
                )

                st.session_state.last_user = (
                    user_id
                )

                add_to_history(
                    user_id,
                    recs,
                )

                st.subheader(
                    f"Neural Hybrid Picks "
                    f"for User #{user_id}"
                )

                fuzzy_info = (
                    get_user_fuzzy_info(
                        user_id
                    )
                )

                if fuzzy_info:

                    dominant = fuzzy_info[
                        "dominant_cluster"
                    ]

                    percentage = fuzzy_info[
                        "top_cluster_pct"
                    ]

                    st.info(
                        f"Recommendations use your "
                        f"fuzzy profile: "
                        f"Cluster {dominant} "
                        f"({percentage:.1%})"
                    )

                for recommendation in recs:

                    movie_id = recommendation[
                        "movie_id"
                    ]

                    (
                        _,
                        clean_title,
                        year,
                        genre,
                    ) = get_movie_meta(
                        movie_id
                    )

                    predicted_rating = (
                        recommendation.get(
                            "predicted_rating",
                            0.0,
                        )
                    )

                    card_data = {

                        "title":
                            clean_title,

                        "confidence":
                            predicted_rating / 5.0,

                        "reasons":
                            [
                                "Neural Hybrid",
                                "ALS Embeddings",
                                "Fuzzy User Profile",
                            ],

                        "year":
                            year,

                        "genre":
                            genre,

                        "poster_path":
                            get_poster_path(
                                movie_id
                            ),

                    }

                    show_movie_card(
                        card_data
                    )

                    st.caption(
                        f"Predicted Rating: "
                        f"**{predicted_rating:.3f} / 5**"
                    )

                st.markdown(
                    "### How was this recommendation?"
                )

                col_fb1, col_fb2, _ = (
                    st.columns(
                        [1, 1, 4]
                    )
                )

                if col_fb1.button(
                    "Like",
                    key="like",
                ):

                    st.session_state[
                        "feedback"
                    ] = "liked"

                    st.success(
                        "Thanks for your feedback!"
                    )

                if col_fb2.button(
                    "Dislike",
                    key="dislike",
                ):

                    st.session_state[
                        "feedback"
                    ] = "disliked"

                    st.info(
                        "Thanks for your feedback."
                    )

                if (
                    "feedback"
                    in st.session_state
                ):

                    st.caption(
                        "Your feedback: "
                        f"{st.session_state['feedback']}"
                    )

            else:

                st.warning(
                    "Could not generate "
                    "recommendations."
                )


# ============================================================
# TAB 3
# History
# ============================================================

with tab3:

    icon_header(
        os.path.join(
            BASE_DIR,
            "pictures",
            "icons",
            "history_icon.png",
        ),
        "Recommendation History",
    )

    if not st.session_state.history:

        st.info(
            "No history yet. "
            "Get some personalized "
            "recommendations first."
        )

    else:

        for index, entry in enumerate(
            st.session_state.history[::-1]
        ):

            session_number = (
                len(
                    st.session_state.history
                )
                - index
            )

            st.write(
                f"**Session {session_number}: "
                f"User #{entry['user_id']}**"
            )

            for recommendation in entry["recs"]:

                title = recommendation.get(
                    "title",
                    "Unknown",
                )

                score = recommendation.get(
                    "predicted_rating",
                    recommendation.get(
                        "confidence",
                        0.0,
                    ),
                )

                st.markdown(
                    f"- **{title}** "
                    f"(score: {score:.3f})"
                )

            st.markdown("---")


# ============================================================
# TAB 4
# Fuzzy User Clusters
# ============================================================

with tab4:

    icon_header(
        os.path.join(
            BASE_DIR,
            "pictures",
            "icons",
            "search_icon.png",
        ),
        "Fuzzy User Clusters",
    )

    st.markdown(
        "### Fuzzy C-Means Clusters "
        "(based on ALS user factors)"
    )

    membership = np.asarray(
        neural_model.fuzzy_u
    )

    n_clusters = membership.shape[1]

    st.write(
        f"Number of clusters: "
        f"**{n_clusters}**"
    )

    st.write(
        f"Number of clustered users: "
        f"**{membership.shape[0]}**"
    )

    top_n = st.slider(
        "Top users per cluster:",
        3,
        20,
        10,
        key="top_cluster",
    )

    cluster_tabs = st.tabs(
        [
            f"Cluster {index}"
            for index in range(
                n_clusters
            )
        ]
    )

    for cluster_id in range(
        n_clusters
    ):

        with cluster_tabs[cluster_id]:

            cluster_scores = membership[
                :,
                cluster_id
            ]

            top_indices = np.argsort(
                cluster_scores
            )[::-1][:top_n]

            rows = []

            for user_index in top_indices:

                user_id = (
                    neural_model.user_reverse.get(
                        int(user_index)
                    )
                )

                if user_id is None:

                    continue

                score = float(
                    cluster_scores[
                        user_index
                    ]
                )

                dominant_cluster = int(
                    np.argmax(
                        membership[
                            user_index
                        ]
                    )
                )

                rows.append(
                    {
                        "User ID":
                            int(user_id),

                        "Membership":
                            round(
                                score,
                                4,
                            ),

                        "Dominant Cluster":
                            (
                                f"Cluster "
                                f"{dominant_cluster}"
                            ),
                    }
                )

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
            )


    st.markdown(
        "### Borderline / Mixed Users"
    )

    if n_clusters >= 2:

        top2 = np.sort(
            membership,
            axis=1,
        )[:, -2:]

        difference = (
            top2[:, 1]
            - top2[:, 0]
        )

        border_indices = np.argsort(
            difference
        )[:10]

        border_rows = []

        for user_index in border_indices:

            user_id = (
                neural_model.user_reverse.get(
                    int(user_index)
                )
            )

            if user_id is None:

                continue

            user_membership = membership[
                user_index
            ]

            top_clusters = np.argsort(
                user_membership
            )[::-1][:2]

            border_rows.append(
                {
                    "User ID":
                        int(user_id),

                    "Top Cluster":
                        (
                            f"Cluster "
                            f"{top_clusters[0]}"
                        ),

                    "Top Membership":
                        round(
                            float(
                                user_membership[
                                    top_clusters[0]
                                ]
                            ),
                            4,
                        ),

                    "Second Cluster":
                        (
                            f"Cluster "
                            f"{top_clusters[1]}"
                        ),

                    "Second Membership":
                        round(
                            float(
                                user_membership[
                                    top_clusters[1]
                                ]
                            ),
                            4,
                        ),

                    "Difference":
                        round(
                            float(
                                difference[
                                    user_index
                                ]
                            ),
                            5,
                        ),
                }
            )

        st.dataframe(
            pd.DataFrame(
                border_rows
            ),
            use_container_width=True,
        )

    else:

        st.info(
            "Borderline analysis requires "
            "at least two clusters."
        )


# ============================================================
# TAB 5
# About
# ============================================================

with tab5:

    icon_header(
        os.path.join(
            BASE_DIR,
            "pictures",
            "icons",
            "info_icon.png",
        ),
        "About the Project",
    )

    st.markdown(
        f"""
        **Advanced Hybrid Movie Recommender System**  
        *Final Year B.Sc. Project - Computer Engineering*

        ---

        ### Dataset Overview

        - **Total Ratings:** {total_ratings:,}
        - **Total Users:** {n_users:,}
        - **Total Movies:** {n_movies:,}

        ---

        ### Features

        - **Content-Based:** Uses movie genres
          and title text (TF-IDF similarity)

        - **Collaborative Filtering:**
          ALS (Alternating Least Squares)
          with confidence weighting

        - **Fuzzy C-Means Clustering:**
          Clusters ALS user embeddings and
          models soft user-cluster memberships

        - **Neural Hybrid Recommender:**
          PyTorch model combining ALS user
          embeddings, item embeddings,
          element-wise interaction,
          and fuzzy user memberships

        - **Hybrid Similarity Engine:**
          Adjustable combination of
          Content-Based and Collaborative
          Filtering

        - **Cold-Start Handling:**
          Bayesian popularity-based
          recommendations for unknown users

        - **Evaluation Metrics:**
          Precision@k, Recall@k,
          NDCG, Coverage,
          Novelty, Diversity

        - **Interactive UI:**
          Streamlit application with
          movie search, local posters,
          recommendation history,
          Neural Hybrid recommendations,
          and fuzzy user-cluster analysis
        """
    )

    rating_distribution_path = os.path.join(
        BASE_DIR,
        "reports",
        "figures",
        "rating_distribution.png",
    )

    if os.path.exists(
        rating_distribution_path
    ):

        st.image(
            rating_distribution_path,
            caption="Rating Distribution",
            use_container_width=True,
        )


# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.caption(
    "Built with Streamlit - "
    "2026 Computer Engineering Student "
    "(Hamidreza Mirzaei) - Final Project"
)
