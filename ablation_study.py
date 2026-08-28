import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_loader import load_movielens, get_genre_matrix
from src.collaborative import CollaborativeRecommender
from src.content_based import ContentBasedRecommender
from src.baseline import PopularRecommender, RandomRecommender
from src.fuzzy_clustering import FuzzyClustering
from src.neural_hybrid import NeuralHybridRecommender
from src.evaluation import evaluate_model, evaluate_baseline
from src.visualization import plot_comparison
from src.utils import get_popularity_dict


def _ensure_predicted_rating(recs):
    formatted = []
    if not isinstance(recs, (list, tuple)):
        return formatted
    for i, r in enumerate(recs):
        if not isinstance(r, dict):
            continue
        item = dict(r)
        if 'predicted_rating' not in item:
            val = None
            for alt_key in ['rating', 'score', 'similarity_score', 'bayesian_avg', 'pred_rating']:
                if alt_key in item:
                    val = float(item[alt_key])
                    break
            if val is None:
                val = float(len(recs) - i)
            item['predicted_rating'] = val
        formatted.append(item)
    return formatted


_orig_build_model = CollaborativeRecommender.build_model

def _patched_build_model(self, df_ratings, df_movies=None, **kwargs):
    if df_movies is None or not isinstance(df_movies, pd.DataFrame):
        df_movies = getattr(CollaborativeRecommender, '_global_df_movies', None)
    return _orig_build_model(self, df_ratings, df_movies)

CollaborativeRecommender.build_model = _patched_build_model


_orig_rand_rec = getattr(RandomRecommender, 'recommend', None)
if _orig_rand_rec is not None:
    def _patched_rand_rec(self, user_id, df_movies=None, n=10, **kwargs):
        n_val = kwargs.get('n', n)
        try:
            recs = _orig_rand_rec(self, user_id, df_movies, n=n_val)
        except TypeError:
            try:
                recs = _orig_rand_rec(self, user_id, n=n_val)
            except TypeError:
                recs = _orig_rand_rec(self, df_movies)
        return _ensure_predicted_rating(recs)
    RandomRecommender.recommend = _patched_rand_rec


_orig_pop_rec = getattr(PopularRecommender, 'recommend', None)
if _orig_pop_rec is not None:
    def _patched_pop_rec(self, user_id, df_movies=None, n=10, **kwargs):
        n_val = kwargs.get('n', n)
        try:
            recs = _orig_pop_rec(self, user_id, df_movies, n=n_val)
        except TypeError:
            try:
                recs = _orig_pop_rec(self, user_id, n=n_val)
            except TypeError:
                recs = _orig_pop_rec(self, df_movies)
        return _ensure_predicted_rating(recs)
    PopularRecommender.recommend = _patched_pop_rec


_orig_cb_recommend = getattr(ContentBasedRecommender, 'recommend', None)

def _patched_cb_recommend(self, user_id, df_movies=None, n=10, **kwargs):
    n_val = kwargs.get('n', kwargs.get('top_n', n))
    if isinstance(n_val, (tuple, list)):
        n_val = max(n_val)

    if isinstance(user_id, str):
        recs = _orig_cb_recommend(self, user_id, n=n_val) if _orig_cb_recommend else []
        return _ensure_predicted_rating(recs)

    train_df = getattr(self, '_train_df', None)
    if train_df is None:
        train_df = getattr(CollaborativeRecommender, '_global_train_df', None)

    if train_df is not None and getattr(self, 'combined_similarity', None) is not None:
        user_ratings = train_df[train_df['user_id'] == user_id]
        if not user_ratings.empty:
            liked = user_ratings[user_ratings['rating'] >= 3.0]
            if liked.empty:
                liked = user_ratings

            watched_mids = set(user_ratings['movie_id'].values)
            sim_scores = np.zeros(len(self.df_movies), dtype=np.float32)

            for _, row in liked.iterrows():
                mid = row['movie_id']
                rating = row['rating']
                idx = self.movie_index.get(mid)
                if idx is not None:
                    sim_scores += self.combined_similarity[idx] * (rating / 5.0)

            top_indices = np.argsort(sim_scores)[::-1]
            recs = []
            for idx in top_indices:
                row = self.df_movies.iloc[idx]
                mid = int(row['movie_id'])
                if mid not in watched_mids:
                    recs.append({
                        'movie_id': mid,
                        'title': str(row['title']),
                        'predicted_rating': float(sim_scores[idx])
                    })
                    if len(recs) >= n_val:
                        break
            return recs

    recs = []
    for idx, row in self.df_movies.head(n_val).iterrows():
        recs.append({
            'movie_id': int(row['movie_id']),
            'title': str(row['title']),
            'predicted_rating': float(n_val - idx)
        })
    return recs

ContentBasedRecommender.recommend = _patched_cb_recommend


_orig_collab_rec_user = getattr(CollaborativeRecommender, 'recommend_for_user', None)
if _orig_collab_rec_user is not None:
    def _patched_collab_rec_user(self, user_id, df_ratings=None, df_movies=None, n=10, **kwargs):
        if df_movies is None:
            df_movies = getattr(CollaborativeRecommender, '_global_df_movies', None)
        if df_ratings is None:
            df_ratings = getattr(CollaborativeRecommender, '_global_train_df', None)
        recs = _orig_collab_rec_user(self, user_id, df_ratings, df_movies, n=n)
        return _ensure_predicted_rating(recs)
    CollaborativeRecommender.recommend_for_user = _patched_collab_rec_user


_orig_nn_rec_user = getattr(NeuralHybridRecommender, 'recommend_for_user', None)
if _orig_nn_rec_user is not None:
    def _patched_nn_rec_user(self, user_id, df_ratings=None, df_movies=None, n=10, **kwargs):
        if df_movies is None:
            df_movies = getattr(CollaborativeRecommender, '_global_df_movies', None)
        if df_ratings is None:
            df_ratings = getattr(CollaborativeRecommender, '_global_train_df', None)
        recs = _orig_nn_rec_user(self, user_id, df_ratings, df_movies, n=n)
        return _ensure_predicted_rating(recs)
    NeuralHybridRecommender.recommend_for_user = _patched_nn_rec_user


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


class HybridEnsemble:
    def __init__(self, collab_model, nn_model, weight_collab=0.7):
        self.collab = collab_model
        self.nn = nn_model
        self.weight_collab = weight_collab
        self.train_user_items = {}

    def fit(self, df_ratings):
        self.train_user_items = df_ratings.groupby('user_id')['movie_id'].apply(set).to_dict()

    def recommend_for_user(self, user_id, df_ratings, df_movies, n=10):
        movie_lookup = df_movies.set_index('movie_id')['title'].to_dict() if df_movies is not None else {}
        if user_id not in self.nn.user_map:
            return []

        u_idx = self.nn.user_map[user_id]
        als_ids, als_scores = self.collab.model.recommend(
            userid=u_idx,
            user_items=self.collab.sparse_matrix[u_idx],
            N=500,
            filter_already_liked_items=True
        )
        als_map = {self.collab.item_reverse[idx]: sc for idx, sc in zip(als_ids, als_scores)
                   if idx in self.collab.item_reverse}
        candidate_mids = list(als_map.keys())
        if not candidate_mids:
            return []

        cand_idxs = [self.nn.item_map[mid] for mid in candidate_mids]
        self.nn.model.eval()
        with torch.no_grad():
            u_emb = torch.tensor(self.nn.user_factors[u_idx], dtype=torch.float32).unsqueeze(0).repeat(len(cand_idxs), 1).to(self.nn.device)
            i_emb = torch.tensor(self.nn.item_factors[cand_idxs], dtype=torch.float32).to(self.nn.device)
            f_feat = torch.tensor(self.nn.fuzzy_u[cand_idxs], dtype=torch.float32).to(self.nn.device)
            nn_scores = self.nn.model(u_emb, i_emb, f_feat).cpu().numpy()

        nn_min, nn_max = nn_scores.min(), nn_scores.max()
        if nn_max - nn_min > 1e-8:
            nn_norm = (nn_scores - nn_min) / (nn_max - nn_min)
        else:
            nn_norm = np.zeros_like(nn_scores)

        als_arr = np.array([als_map[mid] for mid in candidate_mids])
        als_min, als_max = als_arr.min(), als_arr.max()
        if als_max - als_min > 1e-8:
            als_norm = (als_arr - als_min) / (als_max - als_min)
        else:
            als_norm = np.zeros_like(als_arr)

        final_scores = self.weight_collab * als_norm + (1 - self.weight_collab) * nn_norm

        top_indices = np.argsort(final_scores)[::-1][:n]
        recs = []
        for idx in top_indices:
            mid = candidate_mids[idx]
            recs.append({
                'movie_id': int(mid),
                'title': movie_lookup.get(mid, 'Unknown'),
                'predicted_rating': round(float(final_scores[idx]), 3)
            })
        return recs


def run_ablation_study():
    df_ratings, df_movies = load_movielens()
    CollaborativeRecommender._global_df_movies = df_movies

    genre_res = get_genre_matrix(df_movies)
    if isinstance(genre_res, pd.DataFrame):
        genre_mat = genre_res.values
        mids_genre = genre_res.index.values
    else:
        genre_mat = np.asarray(genre_res)
        mids_genre = df_movies['movie_id'].values

    sim_matrix = cosine_similarity(genre_mat)
    genre_sim_matrix = GenreSimDict(mids_genre, sim_matrix)

    train_df, test_df = train_test_split(df_ratings, test_size=0.2, random_state=42)
    CollaborativeRecommender._global_train_df = train_df
    item_pop = get_popularity_dict(train_df)

    print("[1/6] Evaluating Random Baseline...")
    random_model = RandomRecommender(seed=config.RANDOM_SEED).fit(df_movies)
    results_random, _ = evaluate_baseline(
        random_model, test_df, df_movies,
        k_values=[5, 10], threshold=3.5, item_popularity=item_pop
    )

    print("[2/6] Evaluating Popularity Baseline...")
    popular_model = PopularRecommender(min_ratings=config.POPULAR_MIN_RATINGS).fit(train_df)
    results_popular, _ = evaluate_baseline(
        popular_model, test_df, df_movies,
        k_values=[5, 10], threshold=3.5, item_popularity=item_pop
    )

    print("[3/6] Evaluating Content-Based Model...")
    top_movie_ids = pd.Series(item_pop).nlargest(10000).index
    df_movies_subset = df_movies[df_movies['movie_id'].isin(top_movie_ids)].copy().reset_index(drop=True)
    content_model = ContentBasedRecommender(df_movies_subset)
    content_model.build_model(use_title=True, title_weight=0.2)
    content_model._train_df = train_df
    results_content, _ = evaluate_baseline(
        content_model, test_df, df_movies,
        k_values=[5, 10], threshold=3.5, item_popularity=item_pop
    )

    print("[4/6] Evaluating Collaborative Model (ALS)...")
    collab_model = CollaborativeRecommender(alpha=2.0)
    collab_model.build_model(train_df, df_movies)
    np.random.seed(42)
    results_collab, _ = evaluate_model(
        collab_model, test_df, df_movies,
        k_values=[5, 10], threshold=3.5,
        item_similarity_matrix=genre_sim_matrix, item_popularity=item_pop
    )

    user_factors = np.asarray(collab_model.user_factors)
    item_factors = np.asarray(collab_model.item_factors)

    from sklearn.preprocessing import StandardScaler
    import skfuzzy as fuzz

    scaler = StandardScaler()
    item_factors_scaled = scaler.fit_transform(item_factors)
    _, mem, _, _, _, _, _ = fuzz.cluster.cmeans(
        item_factors_scaled.T,
        c=config.FCM_PARAMS['n_clusters'],
        m=config.FCM_PARAMS['m'],
        error=1e-5,
        maxiter=300,
        seed=42
    )
    fuzzy_membership = mem.T.astype(np.float32)

    # 5) Neural Net without Fuzzy (zero fuzzy)
    print("[5/6] Evaluating ALS + Neural Net (No Fuzzy)...")
    zero_fuzzy = np.zeros_like(fuzzy_membership)
    nn_no_fuzzy = NeuralHybridRecommender(
        user_factors=user_factors,
        item_factors=item_factors,
        fuzzy_u=zero_fuzzy,
        user_map=collab_model.user_map,
        item_map=collab_model.item_map,
        hidden_units=config.HYBRID_NN_PARAMS['hidden_units'],
        lr=config.HYBRID_NN_PARAMS['learning_rate'],
        epochs=config.HYBRID_NN_PARAMS['epochs'],
        batch_size=config.HYBRID_NN_PARAMS['batch_size'],
        device='cpu'
    )
    nn_no_fuzzy.fit(train_df)
    np.random.seed(42)
    results_nn_no_fuzzy, _ = evaluate_model(
        nn_no_fuzzy, test_df, df_movies,
        k_values=[5, 10], threshold=3.5,
        item_similarity_matrix=genre_sim_matrix, item_popularity=item_pop
    )

    # 6) Full Hybrid Model with fuzzy weighted training
    print("[6/6] Evaluating Full Proposed Hybrid Model...")
    nn_full = NeuralHybridRecommender(
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
    nn_full.fit(train_df)
    np.random.seed(42)
    results_nn_full, _ = evaluate_model(
        nn_full, test_df, df_movies,
        k_values=[5, 10], threshold=3.5,
        item_similarity_matrix=genre_sim_matrix, item_popularity=item_pop
    )

    # Ensemble: ALS + Fuzzy NN
    print("[7/7] Evaluating ALS + Fuzzy NN Ensemble (Proposed)...")
    ensemble = HybridEnsemble(collab_model, nn_full, weight_collab=0.7)
    ensemble.fit(train_df)
    np.random.seed(42)
    results_ensemble, _ = evaluate_model(
        ensemble, test_df, df_movies,
        k_values=[5, 10], threshold=3.5,
        item_similarity_matrix=genre_sim_matrix, item_popularity=item_pop
    )

    ablation_summary = {
        'Random Baseline': results_random,
        'Popularity Baseline': results_popular,
        'Content-Based': results_content,
        'ALS (Collaborative)': results_collab,
        'ALS + NN (W/O Fuzzy)': results_nn_no_fuzzy,
        'ALS + Fuzzy + NN (Full)': results_nn_full,
        'ALS + NN Ensemble (Proposed)': results_ensemble
    }

    results_df = pd.DataFrame(ablation_summary).T
    os.makedirs('reports', exist_ok=True)
    results_df.to_csv('reports/ablation_study_results.csv')

    print("\n=================== ABLATION STUDY RESULTS ===================")
    print(results_df.to_string())
    print("==============================================================")

    plot_comparison(
        ablation_summary,
        metric_names=['Precision@5', 'Recall@5', 'NDCG@5', 'Novelty@5'],
        save_dir='reports/figures'
    )
    return results_df


if __name__ == '__main__':
    run_ablation_study()
