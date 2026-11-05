import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

import config

from src.data_loader import load_movielens, get_genre_matrix
from src.collaborative import CollaborativeRecommender
from src.content_based import ContentBasedRecommender
from src.hybrid import HybridRecommender
from src.baseline import PopularRecommender, RandomRecommender
from src.fuzzy_clustering import FuzzyClustering
from src.neural_hybrid import NeuralHybridRecommender
from src.evaluation import (
    evaluate_model,
    evaluate_baseline,
)
from src.visualization import plot_eda, plot_comparison
from src.utils import (
    save_model,
    logger,
    get_popularity_dict,
)


class GenreSimilarityRow:
    """
    Helper object for accessing movie similarities.

    Supports:
        matrix[movie_id_1][movie_id_2]
    """

    def __init__(self, parent, movie_id):
        self.parent = parent
        self.movie_id = int(movie_id)

    def __contains__(self, movie_id):
        return int(movie_id) in self.parent.mid_map

    def __getitem__(self, movie_id):
        return self.parent.get_similarity(
            self.movie_id,
            movie_id,
        )

    def get(self, movie_id, default=None):
        similarity = self.parent.get_similarity(
            self.movie_id,
            movie_id,
            default=default,
        )

        return similarity


class GenreSimDict:
    """
    Lightweight movie similarity structure.

    Supports:
        matrix[movie_id_1][movie_id_2]

    Also supports the .get() interface used by evaluation.py:
        matrix.get((movie_id_1, movie_id_2))
        matrix.get(movie_id)
    """

    def __init__(
        self,
        movie_ids,
        similarity_matrix,
    ):
        self.mid_map = {
            int(movie_id): index
            for index, movie_id
            in enumerate(movie_ids)
        }

        self.sim = np.asarray(
            similarity_matrix,
            dtype=np.float32,
        )

    def __contains__(self, movie_id):
        return int(movie_id) in self.mid_map

    def __getitem__(self, movie_id):
        movie_id = int(movie_id)

        if movie_id not in self.mid_map:
            raise KeyError(movie_id)

        return GenreSimilarityRow(
            self,
            movie_id,
        )

    def get_similarity(
        self,
        movie_id_1,
        movie_id_2,
        default=0.0,
    ):
        try:
            movie_id_1 = int(movie_id_1)
            movie_id_2 = int(movie_id_2)
        except (TypeError, ValueError):
            return default

        index_1 = self.mid_map.get(movie_id_1)
        index_2 = self.mid_map.get(movie_id_2)

        if index_1 is None or index_2 is None:
            return default

        return float(
            self.sim[index_1, index_2]
        )

    def get(
        self,
        key,
        default=None,
    ):
        """
        Support both:

            matrix.get(movie_id)

        and:

            matrix.get((movie_id_1, movie_id_2))
        """

        if isinstance(key, tuple):

            if len(key) != 2:
                return default

            return self.get_similarity(
                key[0],
                key[1],
                default=default,
            )

        try:
            movie_id = int(key)
        except (TypeError, ValueError):
            return default

        if movie_id not in self.mid_map:
            return default

        return GenreSimilarityRow(
            self,
            movie_id,
        )

    def __bool__(self):
        return len(self.mid_map) > 0


def main():

    logger.info(
        "===== Starting Advanced Movie Recommender System "
        "(ALS + Fuzzy + PyTorch) ====="
    )

    # ============================================================
    # 1. Load Data
    # ============================================================

    df_ratings, df_movies = load_movielens()

    if (
        hasattr(config, "DEMO_SAMPLE_SIZE")
        and config.DEMO_SAMPLE_SIZE
    ):
        df_ratings = df_ratings.head(
            config.DEMO_SAMPLE_SIZE
        ).copy()

    logger.info(
        "Data loaded: %d ratings, %d movies",
        len(df_ratings),
        len(df_movies),
    )

    if df_ratings.empty:
        raise ValueError(
            "Ratings dataframe is empty."
        )

    if df_movies.empty:
        raise ValueError(
            "Movies dataframe is empty."
        )

    # ============================================================
    # 2. EDA
    # ============================================================

    plot_eda(
        df_ratings,
        df_movies,
    )

    # ============================================================
    # 3. Genre Similarity Matrix
    # ============================================================

    genre_matrix = get_genre_matrix(
        df_movies
    )

    if (
        genre_matrix is not None
        and len(genre_matrix) > 0
    ):
        genre_matrix = np.asarray(
            genre_matrix,
            dtype=np.float32,
        )

        similarity_matrix = cosine_similarity(
            genre_matrix
        )

        genre_sim_matrix = GenreSimDict(
            df_movies["movie_id"].values,
            similarity_matrix,
        )

    else:
        genre_sim_matrix = None

    # ============================================================
    # 4. Train/Test Split
    # ============================================================

    train_df, test_df = train_test_split(
        df_ratings,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    # Keep only users that exist in training data.
    # This prevents evaluation of completely unseen users.
    train_user_ids = set(
        train_df["user_id"].unique()
    )

    test_df = test_df[
        test_df["user_id"].isin(
            train_user_ids
        )
    ].copy()

    if test_df.empty:
        raise ValueError(
            "Test dataframe is empty after filtering "
            "users not present in the training set."
        )

    train_users = train_df[
        "user_id"
    ].unique()

    test_users = test_df[
        "user_id"
    ].unique()

    logger.info(
        "Train set: %d ratings from %d users",
        len(train_df),
        len(train_users),
    )

    logger.info(
        "Test set: %d ratings from %d users",
        len(test_df),
        len(test_users),
    )

    # ============================================================
    # 5. Collaborative Model
    # ============================================================

    logger.info(
        "Building collaborative model (ALS)..."
    )

    collab_model = CollaborativeRecommender(
        alpha=2.0
    )

    collab_model.build_model(
        train_df,
        df_movies,
    )

    save_model(
        collab_model,
        "collaborative",
    )

    item_popularity = get_popularity_dict(
        train_df
    )

    # ============================================================
    # 6. Content-Based Model
    # ============================================================

    logger.info(
        "Building content-based model..."
    )

    content_model = ContentBasedRecommender(
        df_movies
    )

    content_model.build_model(
        use_title=True,
        title_weight=config.DEFAULT_TITLE_WEIGHT,
    )

    save_model(
        content_model,
        "content_based",
    )

    # ============================================================
    # 7. Popular Baseline
    # ============================================================

    logger.info(
        "Evaluating Popular baseline..."
    )

    popular_model = PopularRecommender(
        min_ratings=config.POPULAR_MIN_RATINGS
    )

    popular_model.fit(
        train_df
    )

    save_model(
        popular_model,
        "popular",
    )

    results_pop, _ = evaluate_baseline(
        baseline_model=popular_model,
        train_df=train_df,
        test_df=test_df,
        df_movies=df_movies,
        k_values=config.K_VALUES,
        threshold=3.5,
        item_popularity=item_popularity,
    )

    # ============================================================
    # 8. Random Baseline
    # ============================================================

    logger.info(
        "Evaluating Random baseline..."
    )

    random_model = RandomRecommender(
        seed=config.RANDOM_SEED
    )

    random_model.fit(
        df_movies
    )

    results_rand, _ = evaluate_baseline(
        baseline_model=random_model,
        train_df=train_df,
        test_df=test_df,
        df_movies=df_movies,
        k_values=config.K_VALUES,
        threshold=3.5,
        item_popularity=item_popularity,
    )

    # ============================================================
    # 9. Extract ALS Factors
    # ============================================================

    logger.info(
        "Extracting ALS user and item factors..."
    )

    user_factors = np.asarray(
        collab_model.user_factors
    )

    item_factors = np.asarray(
        collab_model.item_factors
    )

    logger.info(
        "User factors shape: %s",
        user_factors.shape,
    )

    logger.info(
        "Item factors shape: %s",
        item_factors.shape,
    )

    # ============================================================
    # 10. Fuzzy C-Means Clustering
    # ============================================================

    logger.info(
        "Running Fuzzy C-Means clustering "
        "on user factors..."
    )

    n_clusters = min(
        config.FCM_PARAMS["n_clusters"],
        max(1, user_factors.shape[0]),
    )

    fcm = FuzzyClustering(
        n_clusters=n_clusters,
        m=config.FCM_PARAMS["m"],
        max_iter=config.FCM_PARAMS["max_iter"],
        error=config.FCM_PARAMS["error"],
        random_state=config.FCM_PARAMS["random_state"],
    )

    fcm.fit(user_factors)

    fuzzy_membership = fcm.membership

    logger.info(
        "Fuzzy membership shape: %s",
        fuzzy_membership.shape,
    )

    # ============================================================
    # 11. Neural Hybrid Model
    # ============================================================

    logger.info(
        "Building and training "
        "Neural Hybrid Recommender..."
    )

    nn_model = NeuralHybridRecommender(
        user_factors=user_factors,
        item_factors=item_factors,
        fuzzy_u=fuzzy_membership,
        user_map=collab_model.user_map,
        item_map=collab_model.item_map,
        hidden_units=config.HYBRID_NN_PARAMS[
            "hidden_units"
        ],
        lr=config.HYBRID_NN_PARAMS[
            "learning_rate"
        ],
        epochs=config.HYBRID_NN_PARAMS[
            "epochs"
        ],
        batch_size=config.HYBRID_NN_PARAMS[
            "batch_size"
        ],
        device="cpu",
    )

    nn_model.fit(
        train_df
    )

    save_model(
        nn_model,
        "neural_hybrid",
    )

    # ============================================================
    # 12. Evaluate Neural Hybrid
    # ============================================================

    logger.info(
        "Evaluating Neural Hybrid model..."
    )

    results_nn, _ = evaluate_model(
        model=nn_model,
        train_df=train_df,
        test_df=test_df,
        df_movies=df_movies,
        k_values=config.K_VALUES,
        threshold=3.5,
        item_similarity_matrix=genre_sim_matrix,
        item_popularity=item_popularity,
    )

    logger.info(
        "Neural Hybrid results:"
    )

    for metric, value in results_nn.items():

        logger.info(
            "  %s: %.4f",
            metric,
            value,
        )

    # ============================================================
    # 13. Evaluate Collaborative Model
    # ============================================================

    logger.info(
        "Evaluating collaborative model (ALS)..."
    )

    results_collab, _ = evaluate_model(
        model=collab_model,
        train_df=train_df,
        test_df=test_df,
        df_movies=df_movies,
        k_values=config.K_VALUES,
        threshold=3.5,
        item_similarity_matrix=genre_sim_matrix,
        item_popularity=item_popularity,
    )

    logger.info(
        "Collaborative (ALS) results:"
    )

    for metric, value in results_collab.items():

        print(
            f"  {metric}: {value:.4f}"
        )

        logger.info(
            "  %s: %.4f",
            metric,
            value,
        )

    # ============================================================
    # 14. Compare Models
    # ============================================================

    comparison_dict = {
        "ALS": results_collab,
        "ALS+Fuzzy+NN": results_nn,
        "Popular": results_pop,
        "Random": results_rand,
    }

    plot_comparison(
        comparison_dict,
        metric_names=[
            "Precision@5",
            "Recall@5",
            "NDCG@5",
        ],
    )

    # ============================================================
    # 15. Hybrid Recommendation Demo
    # ============================================================

    hybrid = HybridRecommender(
        content_model,
        collab_model,
        train_df,
        df_movies,
    )

    test_movie_id = int(
        df_movies["movie_id"].iloc[0]
    )

    test_movie_title = str(
        df_movies["title"].iloc[0]
    )

    logger.info(
        "Testing similar movies "
        "for movie_id=%s",
        test_movie_id,
    )

    similar_movies = (
        hybrid.recommend_similar_movies(
            movie_id=test_movie_id,
            n=5,
            content_weight=0.6,
        )
    )

    print(
        f"\nMovies similar to "
        f"'{test_movie_title}' (Hybrid):"
    )

    for index, movie in enumerate(
        similar_movies,
        start=1,
    ):
        print(
            f"  {index}. "
            f"{movie['title']} "
            f"(conf: {movie['confidence']})"
        )

    # ============================================================
    # 16. Personalized Recommendation Demo
    # ============================================================

    if len(train_users) > 0:

        test_user = int(
            train_users[0]
        )

        logger.info(
            "Testing recommendations "
            "for user %s",
            test_user,
        )

        user_recommendations = hybrid.recommend(
            user_id=test_user,
            liked_movie_title=test_movie_title,
            n=5,
            content_weight=(
                config.DEFAULT_CONTENT_WEIGHT
            ),
        )

        print(
            f"\nPersonalized recommendations "
            f"for user {test_user}:"
        )

        for index, recommendation in enumerate(
            user_recommendations,
            start=1,
        ):
            print(
                f"  {index}. "
                f"{recommendation['title']} "
                f"(conf: "
                f"{recommendation['confidence']})"
            )

    # ============================================================
    # Final
    # ============================================================

    logger.info(
        "All components validated successfully."
    )

    logger.info(
        "Project completed successfully."
    )


if __name__ == "__main__":
    main()
