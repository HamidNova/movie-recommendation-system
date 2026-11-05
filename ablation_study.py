import os
import sys
import random

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

import skfuzzy as fuzz


sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

import config

from src.data_loader import (
    load_movielens,
    get_genre_matrix,
)

from src.collaborative import (
    CollaborativeRecommender,
)

from src.content_based import (
    ContentBasedRecommender,
)

from src.baseline import (
    PopularRecommender,
    RandomRecommender,
)

from src.neural_hybrid import (
    NeuralHybridRecommender,
)

from src.evaluation import (
    evaluate_model,
    evaluate_baseline,
)

from src.visualization import (
    plot_comparison,
)

from src.utils import (
    get_popularity_dict,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# Recommendation formatting helper
# ============================================================

def _ensure_predicted_rating(recs):

    formatted = []

    if not isinstance(
        recs,
        (list, tuple),
    ):
        return formatted

    for i, rec in enumerate(recs):

        if not isinstance(rec, dict):
            continue

        item = dict(rec)

        if "predicted_rating" not in item:

            value = None

            alternative_keys = [
                "ranking_score",
                "rating",
                "score",
                "similarity_score",
                "bayesian_avg",
                "pred_rating",
            ]

            for key in alternative_keys:

                if key in item:

                    try:

                        value = float(
                            item[key]
                        )

                        break

                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

            if value is None:

                value = float(
                    len(recs) - i
                )

            item["predicted_rating"] = (
                value
            )

        formatted.append(item)

    return formatted


# ============================================================
# Compatibility patches
# ============================================================

_orig_build_model = (
    CollaborativeRecommender.build_model
)


def _patched_build_model(
    self,
    df_ratings,
    df_movies=None,
    **kwargs,
):

    if (
        df_movies is None
        or not isinstance(
            df_movies,
            pd.DataFrame,
        )
    ):

        df_movies = getattr(
            CollaborativeRecommender,
            "_global_df_movies",
            None,
        )

    return _orig_build_model(
        self,
        df_ratings,
        df_movies,
    )


CollaborativeRecommender.build_model = (
    _patched_build_model
)


# ------------------------------------------------------------
# Random baseline patch
# ------------------------------------------------------------

_orig_rand_rec = getattr(
    RandomRecommender,
    "recommend",
    None,
)


if _orig_rand_rec is not None:

    def _patched_rand_rec(
        self,
        user_id,
        df_movies=None,
        n=10,
        **kwargs,
    ):

        n_value = kwargs.get(
            "n",
            n,
        )

        try:

            recs = _orig_rand_rec(
                self,
                user_id,
                df_movies,
                n=n_value,
            )

        except TypeError:

            try:

                recs = _orig_rand_rec(
                    self,
                    user_id,
                    n=n_value,
                )

            except TypeError:

                recs = _orig_rand_rec(
                    self,
                    df_movies,
                )

        return _ensure_predicted_rating(
            recs
        )


    RandomRecommender.recommend = (
        _patched_rand_rec
    )


# ------------------------------------------------------------
# Popular baseline patch
# ------------------------------------------------------------

_orig_pop_rec = getattr(
    PopularRecommender,
    "recommend",
    None,
)


if _orig_pop_rec is not None:

    def _patched_pop_rec(
        self,
        user_id,
        df_movies=None,
        n=10,
        **kwargs,
    ):

        n_value = kwargs.get(
            "n",
            n,
        )

        try:

            recs = _orig_pop_rec(
                self,
                user_id,
                df_movies,
                n=n_value,
            )

        except TypeError:

            try:

                recs = _orig_pop_rec(
                    self,
                    user_id,
                    n=n_value,
                )

            except TypeError:

                recs = _orig_pop_rec(
                    self,
                    df_movies,
                )

        return _ensure_predicted_rating(
            recs
        )


    PopularRecommender.recommend = (
        _patched_pop_rec
    )


# ------------------------------------------------------------
# Content-based patch
# ------------------------------------------------------------

_orig_cb_recommend = getattr(
    ContentBasedRecommender,
    "recommend",
    None,
)


def _patched_cb_recommend(
    self,
    user_id,
    df_movies=None,
    n=10,
    **kwargs,
):

    n_value = kwargs.get(
        "n",
        kwargs.get(
            "top_n",
            n,
        ),
    )

    if isinstance(
        n_value,
        (tuple, list),
    ):

        n_value = max(
            n_value
        )

    # Original content-based behavior
    # for title/string queries.
    if isinstance(
        user_id,
        str,
    ):

        if _orig_cb_recommend is None:
            return []

        recs = _orig_cb_recommend(
            self,
            user_id,
            n=n_value,
        )

        return _ensure_predicted_rating(
            recs
        )

    train_df = getattr(
        self,
        "_train_df",
        None,
    )

    if train_df is None:

        train_df = getattr(
            CollaborativeRecommender,
            "_global_train_df",
            None,
        )

    # Personalized content recommendations
    if (
        train_df is not None
        and getattr(
            self,
            "combined_similarity",
            None,
        ) is not None
    ):

        user_ratings = train_df[
            train_df["user_id"] == user_id
        ]

        if not user_ratings.empty:

            liked = user_ratings[
                user_ratings["rating"] >= 3.5
            ]

            if liked.empty:
                liked = user_ratings

            watched_movies = set(
                user_ratings[
                    "movie_id"
                ].values
            )

            sim_scores = np.zeros(
                len(self.df_movies),
                dtype=np.float32,
            )

            total_weight = 0.0

            for _, row in liked.iterrows():

                movie_id = int(
                    row["movie_id"]
                )

                rating = float(
                    row["rating"]
                )

                index = self.movie_index.get(
                    movie_id
                )

                if index is None:
                    continue

                weight = max(
                    rating,
                    1.0,
                )

                sim_scores += (
                    self.combined_similarity[index]
                    * weight
                )

                total_weight += weight

            if total_weight > 0:

                sim_scores /= total_weight

            ranked_indices = np.argsort(
                sim_scores
            )[::-1]

            recs = []

            for index in ranked_indices:

                row = (
                    self.df_movies.iloc[index]
                )

                movie_id = int(
                    row["movie_id"]
                )

                if movie_id in watched_movies:
                    continue

                recs.append(
                    {
                        "movie_id": movie_id,
                        "title": str(
                            row["title"]
                        ),
                        "predicted_rating": float(
                            sim_scores[index]
                        ),
                    }
                )

                if len(recs) >= n_value:
                    break

            return recs

    return []


ContentBasedRecommender.recommend = (
    _patched_cb_recommend
)


# ------------------------------------------------------------
# Collaborative recommendation patch
# ------------------------------------------------------------

_orig_collab_rec_user = getattr(
    CollaborativeRecommender,
    "recommend_for_user",
    None,
)


if _orig_collab_rec_user is not None:

    def _patched_collab_rec_user(
        self,
        user_id,
        df_ratings=None,
        df_movies=None,
        n=10,
        **kwargs,
    ):

        if df_movies is None:

            df_movies = getattr(
                CollaborativeRecommender,
                "_global_df_movies",
                None,
            )

        if df_ratings is None:

            df_ratings = getattr(
                CollaborativeRecommender,
                "_global_train_df",
                None,
            )

        recs = _orig_collab_rec_user(
            self,
            user_id,
            df_ratings,
            df_movies,
            n=n,
        )

        return _ensure_predicted_rating(
            recs
        )


    CollaborativeRecommender.recommend_for_user = (
        _patched_collab_rec_user
    )


# ------------------------------------------------------------
# Neural recommendation patch
# ------------------------------------------------------------

_orig_nn_rec_user = getattr(
    NeuralHybridRecommender,
    "recommend_for_user",
    None,
)


if _orig_nn_rec_user is not None:

    def _patched_nn_rec_user(
        self,
        user_id,
        df_ratings=None,
        df_movies=None,
        n=10,
        **kwargs,
    ):

        if df_movies is None:

            df_movies = getattr(
                CollaborativeRecommender,
                "_global_df_movies",
                None,
            )

        if df_ratings is None:

            df_ratings = getattr(
                CollaborativeRecommender,
                "_global_train_df",
                None,
            )

        recs = _orig_nn_rec_user(
            self,
            user_id,
            df_ratings,
            df_movies,
            n=n,
        )

        return _ensure_predicted_rating(
            recs
        )


    NeuralHybridRecommender.recommend_for_user = (
        _patched_nn_rec_user
    )


# ============================================================
# Genre similarity dictionary
# ============================================================

class GenreSimDict(dict):

    def __init__(
        self,
        movie_ids,
        similarity_matrix,
    ):

        super().__init__()

        self.movie_map = {
            int(movie_id): index
            for index, movie_id
            in enumerate(movie_ids)
        }

        self.similarity_matrix = (
            similarity_matrix
        )

    def get(
        self,
        key,
        default=0.0,
    ):

        if (
            isinstance(
                key,
                (tuple, list),
            )
            and len(key) == 2
        ):

            index_1 = self.movie_map.get(
                int(key[0])
            )

            index_2 = self.movie_map.get(
                int(key[1])
            )

            if (
                index_1 is not None
                and index_2 is not None
            ):

                return float(
                    self.similarity_matrix[
                        index_1,
                        index_2,
                    ]
                )

        return default

    def __getitem__(
        self,
        key,
    ):

        return self.get(
            key,
            0.0,
        )

    def __len__(self):

        return len(
            self.movie_map
        )

    def __bool__(self):

        return len(self.movie_map) > 0


# ============================================================
# ALS + Neural Ensemble
# ============================================================

class HybridEnsemble:

    def __init__(
        self,
        collab_model,
        nn_model,
        weight_collab=0.7,
    ):

        self.collab = collab_model
        self.nn = nn_model

        self.weight_collab = (
            weight_collab
        )

        self.train_user_items = {}

    def fit(
        self,
        df_ratings,
    ):

        self.train_user_items = (
            df_ratings
            .groupby("user_id")[
                "movie_id"
            ]
            .apply(set)
            .to_dict()
        )

        return self

    def recommend_for_user(
        self,
        user_id,
        df_ratings=None,
        df_movies=None,
        n=10,
    ):

        if df_movies is None:
            return []

        if (
            user_id
            not in self.nn.user_map
        ):
            return []

        if (
            self.nn.model is None
        ):
            return []

        movie_lookup = (
            df_movies
            .set_index("movie_id")[
                "title"
            ]
            .to_dict()
        )

        user_index = (
            self.nn.user_map[user_id]
        )

        # ----------------------------------------------------
        # Get candidates from ALS
        # ----------------------------------------------------

        als_ids, als_scores = (
            self.collab.model.recommend(
                userid=user_index,
                user_items=(
                    self.collab.sparse_matrix[
                        user_index
                    ]
                ),
                N=500,
                filter_already_liked_items=True,
            )
        )

        als_map = {}

        for item_index, score in zip(
            als_ids,
            als_scores,
        ):

            item_index = int(item_index)

            if (
                item_index
                in self.collab.item_reverse
            ):

                movie_id = (
                    self.collab.item_reverse[
                        item_index
                    ]
                )

                if (
                    movie_id
                    in self.nn.item_map
                ):

                    als_map[movie_id] = float(
                        score
                    )

        candidate_movie_ids = list(
            als_map.keys()
        )

        if not candidate_movie_ids:
            return []

        candidate_indices = [
            self.nn.item_map[movie_id]
            for movie_id
            in candidate_movie_ids
        ]

        # ----------------------------------------------------
        # Neural predictions
        # ----------------------------------------------------

        self.nn.model.eval()

        with torch.no_grad():

            user_embedding = torch.tensor(
                self.nn.user_factors[
                    user_index
                ],
                dtype=torch.float32,
            ).unsqueeze(0).repeat(
                len(candidate_indices),
                1,
            ).to(
                self.nn.device
            )

            item_embeddings = torch.tensor(
                self.nn.item_factors[
                    candidate_indices
                ],
                dtype=torch.float32,
            ).to(
                self.nn.device
            )

            # IMPORTANT:
            # fuzzy_u belongs to USER.
            # Therefore we repeat the same user's fuzzy
            # membership for every candidate movie.
            fuzzy_feature = torch.tensor(
                self.nn.fuzzy_u[
                    user_index
                ],
                dtype=torch.float32,
            ).unsqueeze(0).repeat(
                len(candidate_indices),
                1,
            ).to(
                self.nn.device
            )

            nn_scores = (
                self.nn.model(
                    user_embedding,
                    item_embeddings,
                    fuzzy_feature,
                )
                .cpu()
                .numpy()
            )

        # ----------------------------------------------------
        # Normalize Neural scores
        # ----------------------------------------------------

        nn_min = nn_scores.min()
        nn_max = nn_scores.max()

        if nn_max - nn_min > 1e-8:

            nn_normalized = (
                nn_scores - nn_min
            ) / (
                nn_max - nn_min
            )

        else:

            nn_normalized = np.zeros_like(
                nn_scores
            )

        # ----------------------------------------------------
        # Normalize ALS scores
        # ----------------------------------------------------

        als_array = np.array(
            [
                als_map[movie_id]
                for movie_id
                in candidate_movie_ids
            ],
            dtype=np.float32,
        )

        als_min = als_array.min()
        als_max = als_array.max()

        if als_max - als_min > 1e-8:

            als_normalized = (
                als_array - als_min
            ) / (
                als_max - als_min
            )

        else:

            als_normalized = np.zeros_like(
                als_array
            )

        # ----------------------------------------------------
        # Final ensemble score
        # ----------------------------------------------------

        final_scores = (
            self.weight_collab
            * als_normalized
            +
            (
                1.0
                - self.weight_collab
            )
            * nn_normalized
        )

        top_indices = np.argsort(
            final_scores
        )[::-1][:n]

        recommendations = []

        for index in top_indices:

            movie_id = (
                candidate_movie_ids[index]
            )

            recommendations.append(
                {
                    "movie_id": int(movie_id),
                    "title": movie_lookup.get(
                        movie_id,
                        "Unknown",
                    ),
                    "predicted_rating": round(
                        float(
                            final_scores[index]
                        ),
                        6,
                    ),
                }
            )

        return recommendations


# ============================================================
# Main Ablation Study
# ============================================================

def run_ablation_study():

    set_seed(42)

    print(
        "Loading MovieLens data..."
    )

    df_ratings, df_movies = (
        load_movielens()
    )

    CollaborativeRecommender._global_df_movies = (
        df_movies
    )

    # ========================================================
    # Genre similarity
    # ========================================================

    print(
        "Creating genre similarity matrix..."
    )

    genre_result = get_genre_matrix(
        df_movies
    )

    if isinstance(
        genre_result,
        pd.DataFrame,
    ):

        genre_matrix = (
            genre_result.values
        )

        genre_movie_ids = (
            genre_result.index.values
        )

    else:

        genre_matrix = np.asarray(
            genre_result
        )

        genre_movie_ids = (
            df_movies[
                "movie_id"
            ].values
        )

    similarity_matrix = (
        cosine_similarity(
            genre_matrix
        )
    )

    genre_similarity = GenreSimDict(
        genre_movie_ids,
        similarity_matrix,
    )

    # ========================================================
    # Train/Test Split
    # ========================================================

    print(
        "Creating train/test split..."
    )

    train_df, test_df = train_test_split(
        df_ratings,
        test_size=0.2,
        random_state=42,
    )

    train_df = train_df.reset_index(
        drop=True
    )

    test_df = test_df.reset_index(
        drop=True
    )

    CollaborativeRecommender._global_train_df = (
        train_df
    )

    item_popularity = (
        get_popularity_dict(
            train_df
        )
    )

    # ========================================================
    # 1. Random Baseline
    # ========================================================

    print(
        "\n[1/7] Evaluating Random Baseline..."
    )

    random_model = (
        RandomRecommender(
            seed=config.RANDOM_SEED
        )
        .fit(df_movies)
    )

    results_random, _ = (
        evaluate_baseline(
            baseline_model=random_model,

            # IMPORTANT:
            # evaluation.py requires BOTH
            # train_df and test_df.
            train_df=train_df,
            test_df=test_df,

            df_movies=df_movies,

            k_values=[5, 10],

            threshold=3.5,

            item_popularity=item_popularity,

            max_users=100,

            random_state=42,
        )
    )

    # ========================================================
    # 2. Popularity Baseline
    # ========================================================

    print(
        "\n[2/7] Evaluating Popularity Baseline..."
    )

    popular_model = (
        PopularRecommender(
            min_ratings=(
                config.POPULAR_MIN_RATINGS
            )
        )
        .fit(train_df)
    )

    results_popular, _ = (
        evaluate_baseline(
            baseline_model=popular_model,

            train_df=train_df,
            test_df=test_df,

            df_movies=df_movies,

            k_values=[5, 10],

            threshold=3.5,

            item_popularity=item_popularity,

            max_users=100,

            random_state=42,
        )
    )

    # ========================================================
    # 3. Content-Based
    # ========================================================

    print(
        "\n[3/7] Evaluating Content-Based Model..."
    )

    top_movie_ids = (
        pd.Series(
            item_popularity
        )
        .nlargest(10000)
        .index
    )

    df_movies_subset = (
        df_movies[
            df_movies["movie_id"].isin(
                top_movie_ids
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    content_model = (
        ContentBasedRecommender(
            df_movies_subset
        )
    )

    content_model.build_model(
        use_title=True,
        title_weight=0.2,
    )

    content_model._train_df = (
        train_df
    )

    results_content, _ = (
        evaluate_model(
            model=content_model,

            train_df=train_df,
            test_df=test_df,

            df_movies=df_movies,

            k_values=[5, 10],

            threshold=3.5,

            item_similarity_matrix=(
                genre_similarity
            ),

            item_popularity=item_popularity,

            max_users=100,

            random_state=42,
        )
    )

    # ========================================================
    # 4. Collaborative ALS
    # ========================================================

    print(
        "\n[4/7] Evaluating Collaborative Model (ALS)..."
    )

    set_seed(42)

    collab_model = (
        CollaborativeRecommender(
            alpha=2.0
        )
    )

    collab_model.build_model(
        train_df,
        df_movies,
    )

    results_collab, _ = (
        evaluate_model(
            model=collab_model,

            train_df=train_df,
            test_df=test_df,

            df_movies=df_movies,

            k_values=[5, 10],

            threshold=3.5,

            item_similarity_matrix=(
                genre_similarity
            ),

            item_popularity=item_popularity,

            max_users=100,

            random_state=42,
        )
    )

    # ========================================================
    # Extract ALS factors
    # ========================================================

    user_factors = np.asarray(
        collab_model.user_factors,
        dtype=np.float32,
    )

    item_factors = np.asarray(
        collab_model.item_factors,
        dtype=np.float32,
    )

    # ========================================================
    # IMPORTANT: Fuzzy clustering belongs to USERS
    # ========================================================
    #
    # NeuralHybridRecommender expects:
    #
    # fuzzy_u.shape[0] == number of users
    #
    # Therefore FCM must be performed on user_factors.
    # ========================================================

    print(
        "\nCreating fuzzy user memberships..."
    )

    scaler = StandardScaler()

    user_factors_scaled = (
        scaler.fit_transform(
            user_factors
        )
    )

    _, membership, _, _, _, _, _ = (
        fuzz.cluster.cmeans(
            user_factors_scaled.T,

            c=(
                config.FCM_PARAMS[
                    "n_clusters"
                ]
            ),

            m=(
                config.FCM_PARAMS["m"]
            ),

            error=1e-5,

            maxiter=300,

            seed=42,
        )
    )

    # Shape:
    # (n_clusters, n_users)
    # ->
    # (n_users, n_clusters)
    fuzzy_membership = (
        membership.T.astype(
            np.float32
        )
    )

    # ========================================================
    # 5. ALS + Neural Network WITHOUT Fuzzy
    # ========================================================

    print(
        "\n[5/7] Evaluating ALS + Neural Net (No Fuzzy)..."
    )

    # NeuralHybridRecommender requires at least one
    # fuzzy feature, so zero features are used for ablation.
    zero_fuzzy = np.zeros_like(
        fuzzy_membership,
        dtype=np.float32,
    )

    set_seed(42)

    nn_no_fuzzy = (
        NeuralHybridRecommender(
            user_factors=user_factors,

            item_factors=item_factors,

            fuzzy_u=zero_fuzzy,

            user_map=collab_model.user_map,

            item_map=collab_model.item_map,

            hidden_units=(
                config.HYBRID_NN_PARAMS[
                    "hidden_units"
                ]
            ),

            lr=(
                config.HYBRID_NN_PARAMS[
                    "learning_rate"
                ]
            ),

            epochs=(
                config.HYBRID_NN_PARAMS[
                    "epochs"
                ]
            ),

            batch_size=(
                config.HYBRID_NN_PARAMS[
                    "batch_size"
                ]
            ),

            device="cpu",

            random_state=42,
        )
    )

    nn_no_fuzzy.fit(
        train_df
    )

    results_nn_no_fuzzy, _ = (
        evaluate_model(
            model=nn_no_fuzzy,

            train_df=train_df,
            test_df=test_df,

            df_movies=df_movies,

            k_values=[5, 10],

            threshold=3.5,

            item_similarity_matrix=(
                genre_similarity
            ),

            item_popularity=item_popularity,

            max_users=100,

            random_state=42,
        )
    )

    # ========================================================
    # 6. Full Fuzzy Neural Model
    # ========================================================

    print(
        "\n[6/7] Evaluating Full Proposed Hybrid Model..."
    )

    set_seed(42)

    nn_full = (
        NeuralHybridRecommender(
            user_factors=user_factors,

            item_factors=item_factors,

            fuzzy_u=fuzzy_membership,

            user_map=collab_model.user_map,

            item_map=collab_model.item_map,

            hidden_units=(
                config.HYBRID_NN_PARAMS[
                    "hidden_units"
                ]
            ),

            lr=(
                config.HYBRID_NN_PARAMS[
                    "learning_rate"
                ]
            ),

            epochs=(
                config.HYBRID_NN_PARAMS[
                    "epochs"
                ]
            ),

            batch_size=(
                config.HYBRID_NN_PARAMS[
                    "batch_size"
                ]
            ),

            device="cpu",

            random_state=42,
        )
    )

    nn_full.fit(
        train_df
    )

    results_nn_full, _ = (
        evaluate_model(
            model=nn_full,

            train_df=train_df,
            test_df=test_df,

            df_movies=df_movies,

            k_values=[5, 10],

            threshold=3.5,

            item_similarity_matrix=(
                genre_similarity
            ),

            item_popularity=item_popularity,

            max_users=100,

            random_state=42,
        )
    )

    # ========================================================
    # 7. Final Proposed Ensemble
    # ========================================================

    print(
        "\n[7/7] Evaluating ALS + Fuzzy NN Ensemble "
        "(Proposed)..."
    )

    ensemble = HybridEnsemble(
        collab_model=collab_model,

        nn_model=nn_full,

        weight_collab=0.7,
    )

    ensemble.fit(
        train_df
    )

    results_ensemble, _ = (
        evaluate_model(
            model=ensemble,

            train_df=train_df,
            test_df=test_df,

            df_movies=df_movies,

            k_values=[5, 10],

            threshold=3.5,

            item_similarity_matrix=(
                genre_similarity
            ),

            item_popularity=item_popularity,

            max_users=100,

            random_state=42,
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    ablation_summary = {
        "Random Baseline":
            results_random,

        "Popularity Baseline":
            results_popular,

        "Content-Based":
            results_content,

        "ALS (Collaborative)":
            results_collab,

        "ALS + NN (W/O Fuzzy)":
            results_nn_no_fuzzy,

        "ALS + Fuzzy + NN (Full)":
            results_nn_full,

        "ALS + NN Ensemble (Proposed)":
            results_ensemble,
    }

    results_df = (
        pd.DataFrame(
            ablation_summary
        ).T
    )

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    results_df.to_csv(
        "reports/ablation_study_results.csv"
    )

    print(
        "\n"
        "=============================================================="
    )

    print(
        "ABLATION STUDY RESULTS"
    )

    print(
        "=============================================================="
    )

    print(
        results_df.to_string()
    )

    print(
        "=============================================================="
    )

    # ========================================================
    # Visualization
    # ========================================================

    plot_comparison(
        ablation_summary,

        metric_names=[
            "Precision@5",
            "Recall@5",
            "NDCG@5",
            "Novelty@5",
        ],

        save_dir=(
            "reports/figures"
        ),
    )

    return results_df


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    run_ablation_study()
