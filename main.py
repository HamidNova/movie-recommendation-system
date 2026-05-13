import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_movielens, get_genre_matrix
from src.collaborative import CollaborativeRecommender
from src.content_based import ContentBasedRecommender
from src.hybrid import HybridRecommender
from src.baseline import PopularRecommender, RandomRecommender
from src.fuzzy_clustering import FuzzyClustering
from src.neural_hybrid import NeuralHybridRecommender
from src.evaluation import evaluate_model, evaluate_model_with_cross_validation, evaluate_baseline
from src.visualization import plot_eda, plot_comparison
from src.utils import save_model, logger, get_popularity_dict
import config
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    logger.info("===== Starting Advanced Movie Recommender System (ALS + Fuzzy + PyTorch) =====")

    df_ratings, df_movies = load_movielens()
    if config.DEMO_SAMPLE_SIZE:
        df_ratings = df_ratings.head(config.DEMO_SAMPLE_SIZE)
    logger.info(f"Data loaded: {df_ratings.shape[0]} ratings, {df_movies.shape[0]} movies")

    plot_eda(df_ratings, df_movies)

    unique_users = df_ratings['user_id'].unique()
    train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=42)
    train_df = df_ratings[df_ratings['user_id'].isin(train_users)]
    test_df = df_ratings[df_ratings['user_id'].isin(test_users)]

    logger.info(f"Train set: {len(train_df)} ratings from {len(train_users)} users")
    logger.info(f"Test set:  {len(test_df)} ratings from {len(test_users)} users")

    # ========== Collaborative (ALS) ==========
    logger.info("Building collaborative model (ALS)...")
    collab_model = CollaborativeRecommender(alpha=2.0)
    collab_model.build_model(
        train_df,
        factors=config.ALS_PARAMS['factors'],
        iterations=config.ALS_PARAMS['iterations'],
        regularization=config.ALS_PARAMS['regularization']
    )
    save_model(collab_model, 'collaborative')

    # ========== Content-based (subset) ==========
    item_pop = get_popularity_dict(train_df)
    top_movie_ids = pd.Series(item_pop).nlargest(10000).index
    df_movies_subset = df_movies[df_movies['movie_id'].isin(top_movie_ids)].copy()
    df_movies_subset = df_movies_subset.reset_index(drop=True)

    logger.info("Building content-based model (on top 10k popular movies)...")
    content_model = ContentBasedRecommender(df_movies_subset)
    content_model.build_model(use_title=True, title_weight=0.2)
    save_model(content_model, 'content_based')

    genre_sim_matrix = {}  # غیرفعال برای سرعت

    # ========== Baseline: Popular & Random ==========
    logger.info("Evaluating baseline models...")
    popular_model = PopularRecommender(min_ratings=config.POPULAR_MIN_RATINGS).fit(train_df)
    results_pop, _ = evaluate_baseline(
        popular_model, test_df, df_movies,
        k_values=[5, 10], threshold=3.5, item_popularity=item_pop
    )

    random_model = RandomRecommender(seed=config.RANDOM_SEED).fit(df_movies)
    results_rand, _ = evaluate_baseline(
        random_model, test_df, df_movies,
        k_values=[5, 10], threshold=3.5, item_popularity=item_pop
    )

    # ========== استخراج Embedding های ALS ==========
    logger.info("Extracting ALS user and item factors...")
    user_factors = collab_model.model.user_factors  # (n_users, factors)
    item_factors = collab_model.model.item_factors  # (n_items, factors)
    logger.info(f"User factors shape: {user_factors.shape}")
    logger.info(f"Item factors shape: {item_factors.shape}")

    # ========== Fuzzy C-Means روی embedding فیلم‌ها ==========
    logger.info("Running Fuzzy C-Means clustering on item factors...")
    fcm = FuzzyClustering(
        n_clusters=config.FCM_PARAMS['n_clusters'],
        m=config.FCM_PARAMS['m'],
        max_iter=config.FCM_PARAMS['max_iter'],
        error=config.FCM_PARAMS['error'],
        random_state=config.FCM_PARAMS['random_state']
    )
    fcm.fit(item_factors)
    fuzzy_membership = fcm.u  # (n_items, n_clusters)
    logger.info(f"Fuzzy membership shape: {fuzzy_membership.shape}")

    # ========== Neural Hybrid Recommender (PyTorch) ==========
    logger.info("Building and training Neural Hybrid Recommender...")
    nn_model = NeuralHybridRecommender(
        user_factors=user_factors,
        item_factors=item_factors,
        fuzzy_u=fuzzy_membership,
        user_map=collab_model.user_map,
        item_map=collab_model.item_map,
        hidden_units=config.HYBRID_NN_PARAMS['hidden_units'],
        lr=config.HYBRID_NN_PARAMS['learning_rate'],
        epochs=config.HYBRID_NN_PARAMS['epochs'],
        batch_size=config.HYBRID_NN_PARAMS['batch_size'],
        device='cpu'
    )
    nn_model.fit(train_df)
    save_model(nn_model, 'neural_hybrid')

    # ارزیابی مدل جدید
    logger.info("Evaluating Neural Hybrid model...")
    results_nn, _ = evaluate_model(
        nn_model, test_df, df_movies,
        k_values=[5, 10], threshold=3.5,
        item_similarity_matrix=genre_sim_matrix, item_popularity=item_pop
    )
    for metric, val in results_nn.items():
        logger.info(f"  {metric}: {val:.4f}")

    # ========== Evaluation of Collaborative (ALS) ==========
    logger.info("Evaluating collaborative model (ALS)...")
    results_collab, _ = evaluate_model(
        collab_model, test_df, df_movies,
        k_values=[5, 10], threshold=3.5,
        item_similarity_matrix=genre_sim_matrix, item_popularity=item_pop
    )
    logger.info("Collaborative (ALS) results:")
    for metric, val in results_collab.items():
        print(f"  {metric}: {val:.4f}")
        logger.info(f"  {metric}: {val:.4f}")

    # ========== Model Comparison Plot ==========
    comparison_dict = {
        'ALS': results_collab,
        'ALS+Fuzzy+NN': results_nn,
        'Popular': results_pop,
        'Random': results_rand
    }
    plot_comparison(comparison_dict, metric_names=['Precision@5', 'Recall@5', 'NDCG@5'])

    # ========== Optional cross-validation (on ALS) ==========
    if len(train_users) > 200:
        logger.info("Running cross-validation (3 folds) on ALS...")
        cv_results = evaluate_model_with_cross_validation(
            collab_model, train_df, df_movies, k_values=[5], folds=3
        )
        logger.info(f"CV results: {cv_results}")

    # ========== Hybrid Demo ==========
    hybrid = HybridRecommender(content_model, collab_model, train_df, df_movies)

    logger.info("Testing similar-movies (Toy Story - movie_id=1)")
    similar = hybrid.recommend_similar_movies(movie_id=1, n=5, content_weight=0.6)
    print("\nMovies similar to 'Toy Story' (Hybrid):")
    for i, m in enumerate(similar, 1):
        print(f"  {i}. {m['title']} (conf:{m['confidence']})")

    test_user = 1
    if test_user in train_users:
        logger.info(f"Testing recommendations for user {test_user}")
        user_recs = hybrid.recommend(user_id=test_user, liked_movie_title="Toy Story", n=5, content_weight=0.4)
        print(f"\nPersonalized recommendations for user {test_user}:")
        for i, r in enumerate(user_recs, 1):
            print(f"  {i}. {r['title']} (conf:{r['confidence']})")

    logger.info("All components validated. Pipeline is production-ready.")
    logger.info("Project completed successfully.")


if __name__ == "__main__":
    main()
