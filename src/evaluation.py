import numpy as np
import pandas as pd
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def ndcg_at_k(predictions_dict, user_all_test_ratings, k=5, threshold=3.5):
    ndcg_scores = []
    for user_id, user_preds in predictions_dict.items():
        top_k = sorted(user_preds, key=lambda x: x[1], reverse=True)[:k]
        relevance_top_k = [1 if actual >= threshold else 0 for (_, _, actual) in top_k]

        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_top_k))

        test_ratings = user_all_test_ratings.get(user_id, {})
        all_relevances = [1 if r >= threshold else 0 for r in test_ratings.values()]
        ideal_sorted = sorted(all_relevances, reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_sorted))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def novelty_at_k(predictions_dict, item_popularity, k=5):
    novelty_scores = []
    total_pop = sum(item_popularity.values())
    if total_pop == 0:
        return 0.0
    for user_id, user_preds in predictions_dict.items():
        top_k = sorted(user_preds, key=lambda x: x[1], reverse=True)[:k]
        if not top_k:
            continue
        novelty = 0.0
        for movie_id, _, _ in top_k:
            pop = item_popularity.get(movie_id, 1)
            novelty += -np.log2(pop / total_pop)
        novelty_scores.append(novelty / len(top_k))
    return float(np.mean(novelty_scores)) if novelty_scores else 0.0


def diversity_at_k(predictions_dict, item_similarity_matrix, k=5):
    if item_similarity_matrix is None:
        return 0.0
    diversity_scores = []
    for user_id, user_preds in predictions_dict.items():
        top_k_items = [movie_id for (movie_id, _, _) in sorted(user_preds, key=lambda x: x[1], reverse=True)[:k]]
        if len(top_k_items) < 2:
            diversity_scores.append(1.0)
            continue
        total_sim = 0.0
        count = 0
        for i in range(len(top_k_items)):
            for j in range(i + 1, len(top_k_items)):
                sim = 0.0
                m1, m2 = top_k_items[i], top_k_items[j]
                if m1 in item_similarity_matrix and m2 in item_similarity_matrix[m1]:
                    sim = item_similarity_matrix[m1][m2]
                elif m2 in item_similarity_matrix and m1 in item_similarity_matrix[m2]:
                    sim = item_similarity_matrix[m2][m1]
                total_sim += sim
                count += 1
        avg_sim = total_sim / count if count > 0 else 0.0
        diversity_scores.append(1.0 - avg_sim)
    return float(np.mean(diversity_scores)) if diversity_scores else 0.0


def precision_at_k(predictions_dict, k=5, threshold=3.5):
    precisions = []
    for user_id, user_preds in predictions_dict.items():
        top_k = sorted(user_preds, key=lambda x: x[1], reverse=True)[:k]
        if not top_k:
            continue
        relevant = sum(1 for _, _, actual in top_k if actual >= threshold)
        precisions.append(relevant / k)
    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(predictions_dict, all_test_ratings, k=5, threshold=3.5):
    recalls = []
    for user_id, user_preds in predictions_dict.items():
        test_ratings = all_test_ratings.get(user_id, {})
        total_relevant = sum(1 for r in test_ratings.values() if r >= threshold)
        if total_relevant == 0:
            continue
        top_k = sorted(user_preds, key=lambda x: x[1], reverse=True)[:k]
        relevant_in_top_k = sum(1 for _, _, actual in top_k if actual >= threshold)
        recalls.append(relevant_in_top_k / total_relevant)
    return float(np.mean(recalls)) if recalls else 0.0


def coverage(recommended_items, all_items):
    if not all_items:
        return 0.0
    return float(len(set(recommended_items)) / len(set(all_items)))

def _extract_movie_id(rec, df_movies):
    if 'movie_id' in rec:
        return rec['movie_id']
    if 'title' in rec:
        movie_row = df_movies[df_movies['title'] == rec['title']]
        if not movie_row.empty:
            return movie_row['movie_id'].iloc[0]
    return None


def evaluate_model_with_cross_validation(collab_model, df_ratings, df_movies,
                                         k_values=[5, 10], threshold=3.5, folds=3):
    users = df_ratings['user_id'].unique()
    if len(users) > 500:
        users = np.random.choice(users, 500, replace=False)
        df_sample = df_ratings[df_ratings['user_id'].isin(users)]
    else:
        df_sample = df_ratings

    unique_users = df_sample['user_id'].unique()
    np.random.shuffle(unique_users)

    fold_size = max(1, len(unique_users) // folds)
    user_folds = [unique_users[i * fold_size:(i + 1) * fold_size] for i in range(folds)]

    results_over_folds = {f'Precision@{k}': [] for k in k_values}
    results_over_folds.update({f'Recall@{k}': [] for k in k_values})
    results_over_folds.update({f'NDCG@{k}': [] for k in k_values})

    for fold_idx, test_users in enumerate(user_folds):
        if len(test_users) == 0:
            continue
        train_df = df_sample[~df_sample['user_id'].isin(test_users)]
        test_df = df_sample[df_sample['user_id'].isin(test_users)]
        if train_df.empty or test_df.empty:
            continue

        try:
            model = collab_model.__class__(alpha=getattr(collab_model, 'alpha', 2.0))
        except Exception:
            model = collab_model

        if hasattr(model, 'build_model'):
            model.build_model(train_df, factors=50, iterations=20)
        elif hasattr(model, 'fit'):
            model.fit(train_df)

        predictions_dict = {}
        all_test_ratings = {}
        for user_id in test_users:
            user_test = test_df[test_df['user_id'] == user_id]
            all_test_ratings[user_id] = dict(zip(user_test['movie_id'], user_test['rating']))

            if hasattr(model, 'recommend_for_user'):
                recs = model.recommend_for_user(user_id, train_df, df_movies, n=max(k_values))
            elif hasattr(model, 'recommend'):
                recs = model.recommend(user_id, df_movies, n=max(k_values))
            else:
                recs = []

            user_preds = []
            for rec in recs:
                movie_id = _extract_movie_id(rec, df_movies)
                if movie_id is None:
                    continue
                actual = all_test_ratings[user_id].get(movie_id, 0)
                user_preds.append((movie_id, rec['predicted_rating'], actual))
            predictions_dict[user_id] = user_preds

        for k in k_values:
            results_over_folds[f'Precision@{k}'].append(precision_at_k(predictions_dict, k, threshold))
            results_over_folds[f'Recall@{k}'].append(recall_at_k(predictions_dict, k, threshold))
            results_over_folds[f'NDCG@{k}'].append(
                ndcg_at_k(predictions_dict, all_test_ratings, k, threshold)
            )

    final_results = {}
    for metric in results_over_folds:
        vals = results_over_folds[metric]
        final_results[metric] = float(np.mean(vals)) if vals else 0.0
    return final_results


def evaluate_model(collab_model, df_ratings, df_movies, k_values=[5, 10], threshold=3.5,
                   item_similarity_matrix=None, item_popularity=None):
    all_users = df_ratings['user_id'].unique()
    if len(all_users) > 100:
        all_users = np.random.choice(all_users, 100, replace=False)

    predictions_dict = {}
    all_recommended_movies = []
    all_test_ratings = {}

    for user_id in all_users:
        if hasattr(collab_model, 'recommend_for_user'):
            recs = collab_model.recommend_for_user(user_id, df_ratings, df_movies, n=max(k_values))
        elif hasattr(collab_model, 'recommend'):
            recs = collab_model.recommend(user_id, df_movies, n=max(k_values))
        else:
            recs = []

        pred_titles = [r['title'] for r in recs if 'title' in r]
        all_recommended_movies.extend(pred_titles)

        user_ratings = df_ratings[df_ratings['user_id'] == user_id]
        actual_dict = dict(zip(user_ratings['movie_id'], user_ratings['rating']))
        all_test_ratings[user_id] = actual_dict

        user_preds = []
        for r in recs:
            movie_id = _extract_movie_id(r, df_movies)
            if movie_id is None:
                continue
            actual = actual_dict.get(movie_id, 0)
            user_preds.append((movie_id, r['predicted_rating'], actual))
        predictions_dict[user_id] = user_preds

    results = {}
    for k in k_values:
        results[f'Precision@{k}'] = precision_at_k(predictions_dict, k, threshold)
        results[f'Recall@{k}'] = recall_at_k(predictions_dict, all_test_ratings, k, threshold)
        results[f'NDCG@{k}'] = ndcg_at_k(predictions_dict, all_test_ratings, k, threshold)

    all_items = df_movies['title'].unique()
    results['Coverage'] = coverage(all_recommended_movies, all_items)

    if item_popularity is None:
        item_popularity = df_ratings.groupby('movie_id')['rating'].count().to_dict()
    results['Novelty@5'] = novelty_at_k(predictions_dict, item_popularity, k=5)

    if item_similarity_matrix is not None:
        results['Diversity@5'] = diversity_at_k(predictions_dict, item_similarity_matrix, k=5)
    else:
        results['Diversity@5'] = 0.0

    return results, predictions_dict


def evaluate_baseline(baseline_model, df_ratings_test, df_movies,
                      k_values=[5, 10], threshold=3.5, item_popularity=None):
    all_users = df_ratings_test['user_id'].unique()
    predictions_dict = {}
    all_recommended = []
    all_test_ratings = {}

    for user_id in all_users:
        if hasattr(baseline_model, 'recommend'):
            recs = baseline_model.recommend(user_id, df_movies, n=max(k_values))
        elif hasattr(baseline_model, 'recommend_for_user'):
            recs = baseline_model.recommend_for_user(user_id, df_ratings_test, df_movies, n=max(k_values))
        else:
            recs = []

        pred_titles = [rec['title'] for rec in recs if 'title' in rec]
        all_recommended.extend(pred_titles)

        user_test = df_ratings_test[df_ratings_test['user_id'] == user_id]
        actual_dict = dict(zip(user_test['movie_id'], user_test['rating']))
        all_test_ratings[user_id] = actual_dict

        user_preds = []
        for rec in recs:
            mid = _extract_movie_id(rec, df_movies)
            if mid is None:
                continue
            actual = actual_dict.get(mid, 0)
            user_preds.append((mid, rec['predicted_rating'], actual))
        predictions_dict[user_id] = user_preds

    results = {}
    for k in k_values:
        results[f'Precision@{k}'] = precision_at_k(predictions_dict, k, threshold)
        results[f'Recall@{k}'] = recall_at_k(predictions_dict, all_test_ratings, k, threshold)
        results[f'NDCG@{k}'] = ndcg_at_k(predictions_dict, all_test_ratings, k, threshold)

    results['Coverage'] = coverage(all_recommended, df_movies['title'].unique())

    if item_popularity is None:
        item_popularity = df_ratings_test.groupby('movie_id')['rating'].count().to_dict()
    results['Novelty@5'] = novelty_at_k(predictions_dict, item_popularity, k=5)
    results['Diversity@5'] = 0.0

    return results, predictions_dict
