import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_movielens, get_genre_matrix
from src.collaborative import CollaborativeRecommender
from src.content_based import ContentBasedRecommender
from src.hybrid import HybridRecommender
from src.evaluation import evaluate_model, evaluate_model_with_cross_validation
from src.visualization import plot_eda, plot_comparison
from src.utils import save_model, logger, get_popularity_dict
import config
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    logger.info("===== Starting Advanced Movie Recommender System =====")

    df_ratings, df_movies = load_movielens()
    logger.info(f"Data loaded: {df_ratings.shape[0]} ratings, {df_movies.shape[0]} movies")

    plot_eda(df_ratings, df_movies)

    unique_users = df_ratings['user_id'].unique()
    train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=42)
    train_df = df_ratings[df_ratings['user_id'].isin(train_users)]
    test_df = df_ratings[df_ratings['user_id'].isin(test_users)]

    logger.info(f"Train set: {len(train_df)} ratings from {len(train_users)} users")
    logger.info(f"Test set:  {len(test_df)} ratings from {len(test_users)} users")

    logger.info("Building collaborative model on training data...")
    collab_model = CollaborativeRecommender(alpha=2.0)
    collab_model.build_model(
        train_df,
        factors=config.ALS_PARAMS['factors'],
        iterations=config.ALS_PARAMS['iterations'],
        regularization=config.ALS_PARAMS['regularization']
    )
    save_model(collab_model, 'collaborative')

    logger.info("Building content-based model (metadata)...")
    content_model = ContentBasedRecommender(df_movies)
    content_model.build_model(use_title=True, title_weight=0.2)
    save_model(content_model, 'content_based')

    genre_sim_matrix = {}
    if hasattr(content_model, 'combined_similarity'):
        sim_matrix = content_model.combined_similarity
        for i, row in content_model.df_movies.iterrows():
            mid1 = row['movie_id']
            genre_sim_matrix[mid1] = {}
            for j, other in content_model.df_movies.iterrows():
                mid2 = other['movie_id']
                genre_sim_matrix[mid1][mid2] = sim_matrix[i, j]

    logger.info("Evaluating collaborative model on test data...")
    item_pop = get_popularity_dict(train_df)
    results_collab, _ = evaluate_model(
        collab_model, test_df, df_movies,
        k_values=[5, 10],
        threshold=3.5,
        item_similarity_matrix=genre_sim_matrix,
        item_popularity=item_pop
    )
    logger.info("Collaborative model evaluation results:")
    for metric, val in results_collab.items():
        logger.info(f"  {metric}: {val:.4f}")
        print(f"  {metric}: {val:.4f}")

    # Generate model comparison plot
    comparison_dict = {
        'Collaborative': results_collab
    }
    plot_comparison(comparison_dict, metric_names=['Precision@5', 'Recall@5', 'NDCG@5'])

    if len(train_users) > 200:
        logger.info("Running cross-validation (3 folds on 500 users)...")
        cv_results = evaluate_model_with_cross_validation(
            collab_model, train_df, df_movies, k_values=[5], folds=3
        )
        logger.info(f"CV results: {cv_results}")

    hybrid = HybridRecommender(content_model, collab_model, train_df, df_movies)

    logger.info("Testing similar-movies (Toy Story - movie_id=1)")
    similar = hybrid.recommend_similar_movies(movie_id=1, n=5, content_weight=0.6)
    print("\nMovies similar to 'Toy Story' (Hybrid, content_weight=0.6):")
    for i, m in enumerate(similar, 1):
        print(f"  {i}. {m['title']} (conf:{m['confidence']}, reasons:{', '.join(m['reasons'])})")

    test_user = 1
    if test_user in train_users:
        logger.info(f"Testing recommendations for user {test_user}")
        user_recs = hybrid.recommend(user_id=test_user, liked_movie_title="Toy Story", n=5, content_weight=0.4)
        print(f"\nPersonalized recommendations for user {test_user}:")
        for i, r in enumerate(user_recs, 1):
            print(f"  {i}. {r['title']} (conf:{r['confidence']})")
    else:
        logger.warning(f"User {test_user} not in training set, skipping demo.")

    logger.info("Project completed successfully.")


if __name__ == "__main__":
    main()
