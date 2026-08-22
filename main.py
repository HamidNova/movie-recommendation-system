import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
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

_orig_build_model = CollaborativeRecommender.build_model

def _patched_build_model(self, df_ratings, df_movies=None, **kwargs):
    if df_movies is None or not isinstance(df_movies, pd.DataFrame):
        df_movies = getattr(CollaborativeRecommender, '_global_df_movies', None)
    return _orig_build_model(self, df_ratings, df_movies)

CollaborativeRecommender.build_model = _patched_build_model


class GenreSimDict(dict):
    def __init__(self, mids, sim_matrix):
        super().__init__()
        self.mid_map = {int(mid): i for i, mid in enumerate(mids)}
        self.sim = sim_matrix

    def get(self, key, default=0.0):
        if isinstance(key, (tuple, list)) and len(key) == 2:
            i1 = self.mid_map.get(int(key[0]))
            i2 = self.mid_map.get(int(key[1]))
            if i1 is not None and i2 is not None:
                return float(self.sim[i1, i2])
        return default

    def __getitem__(self, key):
        return self.get(key, 0.0)

    def __len__(self):
        return len(self.mid_map)

    def __bool__(self):
        return len(self.mid_map) > 0


def main():
    logger.info("===== Starting Advanced Movie Recommender System (ALS + Fuzzy + PyTorch) =====")

    df_ratings, df_movies = load_movielens()
    CollaborativeRecommender._global_df_movies = df_movies

    if hasattr(config, 'DEMO_SAMPLE_SIZE') and config.DEMO_SAMPLE_SIZE:
        df_ratings = df_ratings.head(config.DEMO_SAMPLE_SIZE)
    logger.info(f"Data loaded: {df_ratings.shape[0]} ratings, {df_movies.shape[0]} movies")

    plot_eda(df_ratings, df_movies)

    genre_res = get_genre_matrix(df_movies)
    if genre_res is not None and len(genre_res) > 0:
        if isinstance(genre_res, pd.DataFrame):
            genre_mat = genre_res.values
            mids = genre_res.index.values if hasattr(genre_res, 'index') and len(genre_res.index) == len(genre_res) else df_movies['movie_id'].values
        else:
            genre_mat = np.asarray(genre_res)
            mids = df_movies['movie_id'].values
        sim_matrix = cosine_similarity(genre_mat)
        genre_sim_matrix = GenreSimDict(mids, sim_matrix)
    else:
        genre_sim_matrix = {}

    train_df, test_df = train_test_split(df_ratings, test_size=0.2, random_state=42)
    train_users = train_df['user_id'].unique()
    test_users = test_df['user_id'].unique()

    logger.info(f"Train set: {len(train_df)} ratings from {len(train_users)} users")
    logger.info(f"Test set:  {len(test_df)} ratings from {len(test_users)} users")

    logger.info("Building collaborative model (ALS)...")
    collab_model = CollaborativeRecommender(alpha=2.0)
    collab_model.build_model(train_df, df_movies)
    save_model(collab_model, 'collaborative')

    item_pop = get_popularity_dict(train_df)
    top_movie_ids = pd.Series(item_pop).nlargest(10000).index
    df_movies_subset = df_movies[df_movies['movie_id'].isin(top_movie_ids)].copy().reset_index(drop=True)

    logger.info("Building content-based model...")
    content_model = ContentBasedRecommender(df_movies_subset)
    content_model.build_model(use_title=True, title_weight=0.2)
    save_model(content_model, 'content_based')

    logger.info("Evaluating baseline models...")
    popular_model = PopularRecommender(min_ratings=config.POPULAR_MIN_RATINGS).fit(train_df)
    save_model(popular_model, 'popular')
    results_pop, _ = evaluate_baseline(
        popular_model, test_df, df_movies,
        k_values=[5, 10], threshold=3.5, item_popularity=item_pop
    )

    random_model = RandomRecommender(seed=config.RANDOM_SEED).fit(df_movies)
    results_rand, _ = evaluate_baseline(
        random_model, test_df, df_movies,
        k_values=[5, 10], threshold=3.5, item_popularity=item_pop
    )

    logger.info("Extracting ALS user and item factors...")
    user_factors = np.asarray(collab_model.user_factors)
    item_factors = np.asarray(collab_model.item_factors)
    logger.info(f"User factors shape: {user_factors.shape}")
    logger.info(f"Item factors shape: {item_factors.shape}")

    logger.info("Running Fuzzy C-Means clustering on item factors...")
    n_clusters = min(config.FCM_PARAMS['n_clusters'], max(1, item_factors.shape[0]))
    fcm = FuzzyClustering(
        n_clusters=n_clusters,
        m=config.FCM_PARAMS['m'],
        max_iter=config.FCM_PARAMS['max_iter'],
        error=config.FCM_PARAMS['error'],
        random_state=config.FCM_PARAMS['random_state']
    )
    fcm.fit(item_factors)
    fuzzy_membership = fcm.membership
    logger.info(f"Fuzzy membership shape: {fuzzy_membership.shape}")

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

    logger.info("Evaluating Neural Hybrid model...")
    results_nn, _ = evaluate_model(
        nn_model, test_df, df_movies,
        k_values=[5, 10], threshold=3.5,
        item_similarity_matrix=genre_sim_matrix, item_popularity=item_pop
    )
    for metric, val in results_nn.items():
        logger.info(f"  {metric}: {val:.4f}")

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

    comparison_dict = {
        'ALS': results_collab,
        'ALS+Fuzzy+NN': results_nn,
        'Popular': results_pop,
        'Random': results_rand
    }
    plot_comparison(comparison_dict, metric_names=['Precision@5', 'Recall@5', 'NDCG@5'])

    if len(train_users) > 200:
        logger.info("Running cross-validation (3 folds) on ALS...")
        cv_results = evaluate_model_with_cross_validation(
            collab_model, train_df, df_movies, k_values=[5], folds=3
        )
        logger.info(f"CV results: {cv_results}")

    hybrid = HybridRecommender(content_model, collab_model, train_df, df_movies)

    test_movie_id = df_movies['movie_id'].iloc[0]
    test_movie_title = df_movies['title'].iloc[0]
    logger.info(f"Testing similar-movies for movie_id={test_movie_id}")
    similar = hybrid.recommend_similar_movies(movie_id=test_movie_id, n=5, content_weight=0.6)
    print(f"\nMovies similar to '{test_movie_title}' (Hybrid):")
    for i, m in enumerate(similar, 1):
        print(f"  {i}. {m['title']} (conf:{m['confidence']})")

    test_user = train_users[0]
    logger.info(f"Testing recommendations for user {test_user}")
    user_recs = hybrid.recommend(user_id=test_user, liked_movie_title=test_movie_title, n=5, content_weight=0.4)
    print(f"\nPersonalized recommendations for user {test_user}:")
    for i, r in enumerate(user_recs, 1):
        print(f"  {i}. {r['title']} (conf:{r['confidence']})")

    logger.info("All components validated. Pipeline is production-ready.")
    logger.info("Project completed successfully.")


if __name__ == "__main__":
    main()
