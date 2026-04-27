# src/evaluation.py
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def ndcg_at_k(predictions_dict, user_all_test_ratings, k=5, threshold=3.5):
    """
    NDCG@k (Normalized Discounted Cumulative Gain)
    predictions_dict: {user_id: [(movie_id, predicted_score, actual_rating), ...]}
    user_all_test_ratings: {user_id: {movie_id: actual_rating}}  (همه‌ی آیتم‌های تست کاربر)
    """
    ndcg_scores = []
    for user_id, user_preds in predictions_dict.items():
        # مرتب‌سازی پیش‌بینی‌ها به صورت نزولی
        top_k = sorted(user_preds, key=lambda x: x[1], reverse=True)[:k]
        relevance_top_k = [1 if actual >= threshold else 0 for (_, _, actual) in top_k]

        # DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_top_k))

        # IDCG: از کل relevanceهای واقعی در مجموعه‌ی تست کاربر
        test_ratings = user_all_test_ratings.get(user_id, {})
        all_relevances = [1 if r >= threshold else 0 for r in test_ratings.values()]
        ideal_sorted = sorted(all_relevances, reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_sorted))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def novelty_at_k(predictions_dict, item_popularity, k=5):
    """
    Novelty@k: میانگین لگاریتم معکوس محبوبیت آیتم‌های توصیه شده
    item_popularity: دیکشنری {movie_id: تعداد ریتینگ}
    """
    novelty_scores = []
    total_pop = sum(item_popularity.values())
    for user_id, user_preds in predictions_dict.items():
        top_k = sorted(user_preds, key=lambda x: x[1], reverse=True)[:k]
        novelty = 0.0
        for movie_id, _, _ in top_k:
            pop = item_popularity.get(movie_id, 1)
            novelty += -np.log2(pop / total_pop)
        novelty_scores.append(novelty / k if k > 0 else 0)
    return np.mean(novelty_scores) if novelty_scores else 0.0


def diversity_at_k(predictions_dict, item_similarity_matrix, k=5):
    """
    Diversity@k: 1 - میانگین تشابه بین همه جفت آیتم‌های توصیه شده
    item_similarity_matrix: دیکشنری {movie_id: {movie_id: similarity}}
    """
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
            for j in range(i+1, len(top_k_items)):
                sim = 0.0
                if (top_k_items[i] in item_similarity_matrix
                        and top_k_items[j] in item_similarity_matrix[top_k_items[i]]):
                    sim = item_similarity_matrix[top_k_items[i]][top_k_items[j]]
                total_sim += sim
                count += 1
        avg_sim = total_sim / count if count > 0 else 0
        diversity_scores.append(1 - avg_sim)
    return np.mean(diversity_scores) if diversity_scores else 0.0


def precision_at_k(predictions_dict, k=5, threshold=3.5):
    """Precision@k"""
    precisions = []
    for user_id, user_preds in predictions_dict.items():
        top_k = sorted(user_preds, key=lambda x: x[1], reverse=True)[:k]
        relevant = sum(1 for _, _, actual in top_k if actual >= threshold)
        precisions.append(relevant / k)
    return np.mean(precisions) if precisions else 0.0


def recall_at_k(predictions_dict, k=5, threshold=3.5):
    """Recall@k"""
    recalls = []
    for user_id, user_preds in predictions_dict.items():
        total_relevant = sum(1 for _, _, actual in user_preds if actual >= threshold)
        if total_relevant == 0:
            continue
        top_k = sorted(user_preds, key=lambda x: x[1], reverse=True)[:k]
        relevant_in_top_k = sum(1 for _, _, actual in top_k if actual >= threshold)
        recalls.append(relevant_in_top_k / total_relevant)
    return np.mean(recalls) if recalls else 0.0


def coverage(recommended_items, all_items):
    """Coverage: نسبت آیتم‌های توصیه شده به کل آیتم‌ها"""
    return len(set(recommended_items)) / len(set(all_items))


def evaluate_model_with_cross_validation(collab_model, df_ratings, df_movies,
                                         k_values=[5, 10], threshold=3.5, folds=3):
    """
    ارزیابی با cross-validation ساده
    """
    users = df_ratings['user_id'].unique()
    if len(users) > 500:
        users = np.random.choice(users, 500, replace=False)
        df_sample = df_ratings[df_ratings['user_id'].isin(users)]
    else:
        df_sample = df_ratings

    unique_users = df_sample['user_id'].unique()
    np.random.shuffle(unique_users)
    fold_size = len(unique_users) // folds
    user_folds = [unique_users[i*fold_size:(i+1)*fold_size] for i in range(folds)]

    results_over_folds = {f'Precision@{k}': [] for k in k_values}
    results_over_folds.update({f'Recall@{k}': [] for k in k_values})
    results_over_folds.update({f'NDCG@{k}': [] for k in k_values})

    for fold_idx, test_users in enumerate(user_folds):
        train_df = df_sample[~df_sample['user_id'].isin(test_users)]
        test_df = df_sample[df_sample['user_id'].isin(test_users)]
        if train_df.empty or test_df.empty:
            continue

        model = collab_model.__class__(alpha=getattr(collab_model, 'alpha', 2.0))
        model.build_model(train_df, factors=50, iterations=20)

        predictions_dict = {}
        all_test_ratings = {}  # user_id -> {movie_id: actual_rating}
        for user_id in test_users:
            # ذخیره همه ریتینگ‌های تست کاربر
            user_test = test_df[test_df['user_id'] == user_id]
            all_test_ratings[user_id] = dict(zip(user_test['movie_id'], user_test['rating']))

            recs = model.recommend_for_user(user_id, train_df, df_movies, n=max(k_values))
            user_preds = []
            for rec in recs:
                title = rec['title']
                movie_row = df_movies[df_movies['title'] == title]
                if movie_row.empty:
                    continue
                movie_id = movie_row['movie_id'].iloc[0]
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
        final_results[metric] = np.mean(results_over_folds[metric])
    return final_results


def evaluate_model(collab_model, df_ratings, df_movies, k_values=[5, 10], threshold=3.5,
                   item_similarity_matrix=None, item_popularity=None):
    """
    ارزیابی ساده (بدون cross-validation)
    با محاسبه NDCG، Novelty، Diversity (در صورت ارائه ماتریس تشابه)
    """
    all_users = df_ratings['user_id'].unique()
    if len(all_users) > 100:
        all_users = np.random.choice(all_users, 100, replace=False)

    predictions_dict = {}
    all_recommended_movies = []
    all_recommended_ids = []
    all_test_ratings = {}  # ذخیره تمام ریتینگ‌های تست واقعی برای هر کاربر

    for user_id in all_users:
        recs = collab_model.recommend_for_user(user_id, df_ratings, df_movies, n=max(k_values))
        pred_titles = [r['title'] for r in recs]
        all_recommended_movies.extend(pred_titles)

        # جمع‌آوری ریتینگ‌های واقعی کاربر از کل دیتافریم (برای NDCG و غیره)
        user_ratings = df_ratings[df_ratings['user_id'] == user_id]
        actual_dict = dict(zip(user_ratings['movie_id'], user_ratings['rating']))
        all_test_ratings[user_id] = actual_dict  # ذخیره می‌کنیم

        user_preds = []
        for r in recs:
            movie_row = df_movies[df_movies['title'] == r['title']]
            if movie_row.empty:
                continue
            movie_id = movie_row['movie_id'].iloc[0]
            actual = actual_dict.get(movie_id, 0)
            user_preds.append((movie_id, r['predicted_rating'], actual))
        predictions_dict[user_id] = user_preds

    results = {}
    for k in k_values:
        results[f'Precision@{k}'] = precision_at_k(predictions_dict, k, threshold)
        results[f'Recall@{k}'] = recall_at_k(predictions_dict, k, threshold)
        results[f'NDCG@{k}'] = ndcg_at_k(predictions_dict, all_test_ratings, k, threshold)

    # Coverage
    all_items = df_movies['title'].unique()
    results['Coverage'] = coverage(all_recommended_movies, all_items)

    # Novelty
    if item_popularity is None:
        item_popularity = df_ratings.groupby('movie_id')['rating'].count().to_dict()
    results['Novelty@5'] = novelty_at_k(predictions_dict, item_popularity, k=5)

    # Diversity
    if item_similarity_matrix is not None:
        results['Diversity@5'] = diversity_at_k(predictions_dict, item_similarity_matrix, k=5)
    else:
        results['Diversity@5'] = 0.0

    return results, predictions_dict
