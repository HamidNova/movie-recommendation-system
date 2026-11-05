import logging

import numpy as np

from sklearn.model_selection import KFold


logger = logging.getLogger(__name__)


def _extract_movie_id(rec, df_movies):
    """
    Extract movie_id from a recommendation dictionary.

    The preferred format is:

        {
            "movie_id": ...,
            ...
        }

    If movie_id is unavailable, title is used as a fallback.
    """

    if not isinstance(rec, dict):
        return None

    movie_id = rec.get("movie_id")

    if movie_id is not None:
        try:
            return int(movie_id)
        except (TypeError, ValueError):
            return None

    title = rec.get("title")

    if title is None:
        return None

    if (
        df_movies is None
        or df_movies.empty
        or "title" not in df_movies.columns
        or "movie_id" not in df_movies.columns
    ):
        return None

    movie_row = df_movies[
        df_movies["title"] == title
    ]

    if movie_row.empty:
        return None

    try:
        return int(
            movie_row["movie_id"].iloc[0]
        )
    except (TypeError, ValueError):
        return None


def _get_predicted_score(rec):
    """
    Safely extract a ranking score from a recommendation.

    Different recommender implementations may use
    different key names.
    """

    if not isinstance(rec, dict):
        return 0.0

    score_keys = [
        "ranking_score",
        "predicted_rating",
        "score",
        "rating",
        "similarity_score",
        "bayesian_avg",
        "confidence",
    ]

    for key in score_keys:

        value = rec.get(key)

        if value is None:
            continue

        try:
            score = float(value)

            if np.isfinite(score):
                return score

        except (TypeError, ValueError):
            continue

    return 0.0


def _get_item_similarity(
    item_similarity_matrix,
    movie_id_1,
    movie_id_2,
):
    """
    Return similarity between two movies.

    Supports:

    - dict with tuple keys:
        {(movie1, movie2): similarity}

    - GenreSimDict-like objects with:
        .get((movie1, movie2))

    - nested dictionaries:
        {movie1: {movie2: similarity}}

    Returns
    -------
    float or None

    None means similarity is unavailable.
    This is different from similarity == 0.0.
    """

    if item_similarity_matrix is None:
        return None

    movie_id_1 = int(movie_id_1)
    movie_id_2 = int(movie_id_2)

    try:

        similarity = item_similarity_matrix.get(
            (movie_id_1, movie_id_2),
            None,
        )

        if similarity is not None:
            similarity = float(similarity)

            if np.isfinite(similarity):
                return similarity

    except Exception:
        pass

    try:

        similarity = item_similarity_matrix.get(
            (movie_id_2, movie_id_1),
            None,
        )

        if similarity is not None:
            similarity = float(similarity)

            if np.isfinite(similarity):
                return similarity

    except Exception:
        pass

    try:

        first_level = item_similarity_matrix.get(
            movie_id_1,
            None,
        )

        if isinstance(first_level, dict):

            similarity = first_level.get(
                movie_id_2,
                None,
            )

            if similarity is not None:
                similarity = float(similarity)

                if np.isfinite(similarity):
                    return similarity

    except Exception:
        pass

    try:

        first_level = item_similarity_matrix.get(
            movie_id_2,
            None,
        )

        if isinstance(first_level, dict):

            similarity = first_level.get(
                movie_id_1,
                None,
            )

            if similarity is not None:
                similarity = float(similarity)

                if np.isfinite(similarity):
                    return similarity

    except Exception:
        pass

    return None


def _recommend_content_for_user(
    model,
    user_id,
    train_df,
    n,
):
    """
    Create personalized recommendations for a content-based model.

    Recommendations are generated using movies that
    the user interacted with in the training set.
    """

    if n <= 0:
        return []

    required_attributes = [
        "combined_similarity",
        "movie_index",
        "df_movies",
    ]

    if not all(
        hasattr(model, attr)
        for attr in required_attributes
    ):
        return []

    if model.combined_similarity is None:
        return []

    if model.df_movies is None:
        return []

    if model.df_movies.empty:
        return []

    user_ratings = train_df[
        train_df["user_id"] == user_id
    ]

    if user_ratings.empty:
        return []

    watched_movies = set(
        user_ratings["movie_id"]
        .astype(int)
    )

    liked_movies = user_ratings[
        user_ratings["rating"] >= 3.5
    ]

    if liked_movies.empty:
        liked_movies = user_ratings

    similarity_scores = np.zeros(
        len(model.df_movies),
        dtype=np.float32,
    )

    total_weight = 0.0

    for _, row in liked_movies.iterrows():

        movie_id = int(row["movie_id"])
        rating = float(row["rating"])

        movie_index = model.movie_index.get(
            movie_id
        )

        if movie_index is None:
            continue

        weight = max(rating, 1.0)

        similarity_scores += (
            model.combined_similarity[movie_index]
            * weight
        )

        total_weight += weight

    if total_weight <= 0:
        return []

    similarity_scores /= total_weight

    ranked_indices = np.argsort(
        similarity_scores
    )[::-1]

    recommendations = []

    for index in ranked_indices:

        movie = model.df_movies.iloc[index]

        try:
            movie_id = int(
                movie["movie_id"]
            )
        except (KeyError, TypeError, ValueError):
            continue

        if movie_id in watched_movies:
            continue

        recommendations.append(
            {
                "movie_id": movie_id,
                "title": str(
                    movie["title"]
                ),
                "ranking_score": float(
                    similarity_scores[index]
                ),
            }
        )

        if len(recommendations) >= n:
            break

    return recommendations


def _get_recommendations(
    model,
    user_id,
    train_df,
    df_movies,
    n,
):
    """
    Unified recommendation interface.

    Supports:

    - Collaborative models
    - Content-based models
    - Neural models
    - Baseline models
    """

    if n <= 0:
        return []

    if hasattr(model, "recommend_for_user"):

        try:

            recommendations = model.recommend_for_user(
                user_id,
                train_df,
                df_movies,
                n=n,
            )

            return recommendations or []

        except TypeError:

            logger.debug(
                "recommend_for_user signature did not match.",
                exc_info=True,
            )

        except Exception:

            logger.exception(
                "Error while generating recommendations "
                "for user %s.",
                user_id,
            )

            return []

    if (
        hasattr(model, "combined_similarity")
        and hasattr(model, "movie_index")
        and hasattr(model, "df_movies")
    ):

        return _recommend_content_for_user(
            model=model,
            user_id=user_id,
            train_df=train_df,
            n=n,
        )

    if hasattr(model, "recommend"):

        try:

            recommendations = model.recommend(
                user_id,
                df_movies,
                n=n,
            )

            return recommendations or []

        except TypeError:

            logger.debug(
                "recommend signature did not match.",
                exc_info=True,
            )

        except Exception:

            logger.exception(
                "Error while generating recommendations "
                "for user %s.",
                user_id,
            )

            return []

    return []


def precision_at_k(
    predictions_dict,
    k=5,
    threshold=3.5,
):
    """
    Calculate average Precision@K.
    """

    if k <= 0:
        return 0.0

    precisions = []

    for user_preds in predictions_dict.values():

        top_k = sorted(
            user_preds,
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        if not top_k:
            continue

        relevant_count = sum(
            1
            for _, _, actual in top_k
            if actual >= threshold
        )

        precisions.append(
            relevant_count / len(top_k)
        )

    return (
        float(np.mean(precisions))
        if precisions
        else 0.0
    )


def recall_at_k(
    predictions_dict,
    all_test_ratings,
    k=5,
    threshold=3.5,
):
    """
    Calculate average Recall@K.
    """

    if k <= 0:
        return 0.0

    recalls = []

    for user_id, user_preds in (
        predictions_dict.items()
    ):

        test_ratings = all_test_ratings.get(
            user_id,
            {},
        )

        relevant_items = {
            movie_id
            for movie_id, rating
            in test_ratings.items()
            if rating >= threshold
        }

        total_relevant = len(
            relevant_items
        )

        if total_relevant == 0:
            continue

        top_k = sorted(
            user_preds,
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        recommended_relevant = {
            movie_id
            for movie_id, _, actual
            in top_k
            if actual >= threshold
        }

        recalls.append(
            len(recommended_relevant)
            / total_relevant
        )

    return (
        float(np.mean(recalls))
        if recalls
        else 0.0
    )


def ndcg_at_k(
    predictions_dict,
    user_all_test_ratings,
    k=5,
    threshold=3.5,
):
    """
    Calculate binary relevance NDCG@K.
    """

    if k <= 0:
        return 0.0

    ndcg_scores = []

    for user_id, user_preds in (
        predictions_dict.items()
    ):

        test_ratings = (
            user_all_test_ratings.get(
                user_id,
                {},
            )
        )

        relevant_count = sum(
            1
            for rating in test_ratings.values()
            if rating >= threshold
        )

        if relevant_count == 0:
            continue

        top_k = sorted(
            user_preds,
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        dcg = 0.0

        for rank, (
            _,
            _,
            actual_rating,
        ) in enumerate(top_k):

            if actual_rating >= threshold:

                dcg += (
                    1.0
                    / np.log2(rank + 2)
                )

        ideal_count = min(
            relevant_count,
            k,
        )

        idcg = sum(
            1.0 / np.log2(rank + 2)
            for rank in range(ideal_count)
        )

        if idcg > 0:

            ndcg_scores.append(
                dcg / idcg
            )

    return (
        float(np.mean(ndcg_scores))
        if ndcg_scores
        else 0.0
    )


def novelty_at_k(
    predictions_dict,
    item_popularity,
    k=5,
):
    """
    Calculate average novelty using self-information.

    Less popular recommended items produce
    higher novelty scores.
    """

    if k <= 0:
        return 0.0

    if not item_popularity:
        return 0.0

    total_popularity = sum(
        max(float(value), 0.0)
        for value in item_popularity.values()
    )

    if total_popularity <= 0:
        return 0.0

    novelty_scores = []

    for user_preds in predictions_dict.values():

        top_k = sorted(
            user_preds,
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        if not top_k:
            continue

        novelty = 0.0

        for movie_id, _, _ in top_k:

            popularity = max(
                float(
                    item_popularity.get(
                        movie_id,
                        1,
                    )
                ),
                1.0,
            )

            probability = (
                popularity / total_popularity
            )

            novelty += -np.log2(
                max(
                    probability,
                    1e-12,
                )
            )

        novelty_scores.append(
            novelty / len(top_k)
        )

    return (
        float(np.mean(novelty_scores))
        if novelty_scores
        else 0.0
    )


def diversity_at_k(
    predictions_dict,
    item_similarity_matrix,
    k=5,
):
    """
    Calculate intra-list diversity.

    Diversity = 1 - average pairwise similarity.

    Pairs without available similarity are ignored
    instead of being incorrectly treated as similarity 0.
    """

    if k <= 1:
        return 0.0

    if item_similarity_matrix is None:
        return 0.0

    diversity_scores = []

    for user_preds in predictions_dict.values():

        top_k = sorted(
            user_preds,
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        movie_ids = [
            int(movie_id)
            for movie_id, _, _ in top_k
        ]

        if len(movie_ids) < 2:
            continue

        similarities = []

        for i in range(len(movie_ids)):

            for j in range(
                i + 1,
                len(movie_ids),
            ):

                similarity = (
                    _get_item_similarity(
                        item_similarity_matrix,
                        movie_ids[i],
                        movie_ids[j],
                    )
                )

                if similarity is None:
                    continue

                similarities.append(
                    float(
                        np.clip(
                            similarity,
                            0.0,
                            1.0,
                        )
                    )
                )

        if not similarities:
            continue

        average_similarity = float(
            np.mean(similarities)
        )

        diversity_scores.append(
            1.0 - average_similarity
        )

    return (
        float(np.mean(diversity_scores))
        if diversity_scores
        else 0.0
    )


def coverage_at_k(
    predictions_dict,
    all_items,
    k=5,
):
    """
    Calculate catalog coverage@K.

    Coverage measures how much of the available
    catalog appears in at least one user's top-K list.
    """

    if k <= 0:
        return 0.0

    all_items = set(
        int(item)
        for item in all_items
    )

    if not all_items:
        return 0.0

    recommended_items = set()

    for user_preds in predictions_dict.values():

        top_k = sorted(
            user_preds,
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        for movie_id, _, _ in top_k:

            recommended_items.add(
                int(movie_id)
            )

    return float(
        len(
            recommended_items & all_items
        )
        / len(all_items)
    )


def evaluate_model(
    model,
    train_df,
    test_df,
    df_movies,
    k_values=None,
    threshold=3.5,
    item_similarity_matrix=None,
    item_popularity=None,
    max_users=100,
    random_state=42,
):
    """
    Evaluate a trained recommender.

    Parameters
    ----------
    model:
        Already-trained recommendation model.

    train_df:
        Data used to train the model.

    test_df:
        Unseen interactions used for evaluation.

    df_movies:
        Movie metadata.

    k_values:
        List of K values, for example [5, 10].

    threshold:
        Rating threshold for relevant items.

    Notes
    -----
    Only users and items known to the training data
    are evaluated. This prevents cold-start users/items
    from unfairly reducing ranking metrics.
    """

    if k_values is None:
        k_values = [5, 10]

    if not k_values:
        raise ValueError(
            "k_values must contain at least one value."
        )

    if any(k <= 0 for k in k_values):
        raise ValueError(
            "All k_values must be greater than 0."
        )

    required_rating_columns = {
        "user_id",
        "movie_id",
        "rating",
    }

    if not required_rating_columns.issubset(
        train_df.columns
    ):
        raise ValueError(
            "train_df must contain user_id, "
            "movie_id, and rating columns."
        )

    if not required_rating_columns.issubset(
        test_df.columns
    ):
        raise ValueError(
            "test_df must contain user_id, "
            "movie_id, and rating columns."
        )

    if (
        df_movies is None
        or df_movies.empty
        or "movie_id" not in df_movies.columns
    ):
        raise ValueError(
            "df_movies must contain movie_id."
        )

    train_users = set(
        train_df["user_id"]
        .astype(int)
        .unique()
    )

    train_items = set(
        train_df["movie_id"]
        .astype(int)
        .unique()
    )

    evaluable_test_df = test_df[
        test_df["user_id"]
        .astype(int)
        .isin(train_users)
        &
        test_df["movie_id"]
        .astype(int)
        .isin(train_items)
    ].copy()

    if evaluable_test_df.empty:

        logger.warning(
            "No evaluable test interactions remain "
            "after removing cold-start users/items."
        )

        return {}, {}

    rng = np.random.RandomState(
        random_state
    )

    all_users = (
        evaluable_test_df["user_id"]
        .unique()
    )

    if (
        max_users is not None
        and len(all_users) > max_users
    ):

        all_users = rng.choice(
            all_users,
            size=max_users,
            replace=False,
        )

    predictions_dict = {}
    all_test_ratings = {}

    max_k = max(k_values)

    for user_id in all_users:

        user_id = int(user_id)

        user_test = evaluable_test_df[
            evaluable_test_df["user_id"]
            == user_id
        ]

        actual_dict = {
            int(movie_id): float(rating)
            for movie_id, rating in zip(
                user_test["movie_id"],
                user_test["rating"],
            )
        }

        if not actual_dict:
            continue

        all_test_ratings[user_id] = (
            actual_dict
        )

        recommendations = (
            _get_recommendations(
                model=model,
                user_id=user_id,
                train_df=train_df,
                df_movies=df_movies,
                n=max_k,
            )
        )

        user_preds = []

        seen_recommendations = set()

        for rec in recommendations:

            movie_id = _extract_movie_id(
                rec,
                df_movies,
            )

            if movie_id is None:
                continue

            movie_id = int(movie_id)

            if movie_id in seen_recommendations:
                continue

            seen_recommendations.add(
                movie_id
            )

            predicted_score = (
                _get_predicted_score(rec)
            )

            actual_rating = actual_dict.get(
                movie_id,
                0.0,
            )

            user_preds.append(
                (
                    movie_id,
                    predicted_score,
                    float(actual_rating),
                )
            )

            if len(user_preds) >= max_k:
                break

        predictions_dict[user_id] = (
            user_preds
        )

    if not predictions_dict:

        logger.warning(
            "No predictions were generated."
        )

        return {}, {}

    results = {}

    all_items = set(
        int(movie_id)
        for movie_id in train_items
    )

    for k in k_values:

        results[f"Precision@{k}"] = (
            precision_at_k(
                predictions_dict,
                k=k,
                threshold=threshold,
            )
        )

        results[f"Recall@{k}"] = (
            recall_at_k(
                predictions_dict,
                all_test_ratings,
                k=k,
                threshold=threshold,
            )
        )

        results[f"NDCG@{k}"] = (
            ndcg_at_k(
                predictions_dict,
                all_test_ratings,
                k=k,
                threshold=threshold,
            )
        )

        results[f"Coverage@{k}"] = (
            coverage_at_k(
                predictions_dict,
                all_items,
                k=k,
            )
        )

    if item_popularity is None:

        item_popularity = (
            train_df
            .groupby("movie_id")
            .size()
            .to_dict()
        )

    for k in k_values:

        results[f"Novelty@{k}"] = (
            novelty_at_k(
                predictions_dict,
                item_popularity,
                k=k,
            )
        )

        results[f"Diversity@{k}"] = (
            diversity_at_k(
                predictions_dict,
                item_similarity_matrix,
                k=k,
            )
        )

    return results, predictions_dict


def evaluate_baseline(
    baseline_model,
    train_df,
    test_df,
    df_movies,
    k_values=None,
    threshold=3.5,
    item_popularity=None,
    max_users=100,
    random_state=42,
):
    """
    Evaluate Popular and Random baseline recommenders.
    """

    return evaluate_model(
        model=baseline_model,
        train_df=train_df,
        test_df=test_df,
        df_movies=df_movies,
        k_values=k_values,
        threshold=threshold,
        item_similarity_matrix=None,
        item_popularity=item_popularity,
        max_users=max_users,
        random_state=random_state,
    )


def evaluate_model_with_cross_validation(
    model_class,
    df_ratings,
    df_movies,
    k_values=None,
    threshold=3.5,
    folds=3,
    max_users=100,
    random_state=42,
):
    """
    Evaluate a recommender using interaction-level K-Fold
    cross validation.

    A fresh model is created and trained for every fold.

    Cold-start users and items are removed from each
    fold's test set before evaluation.
    """

    if k_values is None:
        k_values = [5, 10]

    if not isinstance(
        model_class,
        type,
    ):
        raise TypeError(
            "model_class must be a model class, "
            "for example CollaborativeRecommender, "
            "not an already-created model instance."
        )

    if folds < 2:
        raise ValueError(
            "folds must be at least 2."
        )

    if len(df_ratings) < folds:
        raise ValueError(
            "Number of ratings must be greater than "
            "or equal to the number of folds."
        )

    required_columns = {
        "user_id",
        "movie_id",
        "rating",
    }

    if not required_columns.issubset(
        df_ratings.columns
    ):
        raise ValueError(
            "df_ratings must contain user_id, "
            "movie_id, and rating columns."
        )

    kfold = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )

    metric_names = []

    for k in k_values:

        metric_names.extend(
            [
                f"Precision@{k}",
                f"Recall@{k}",
                f"NDCG@{k}",
                f"Coverage@{k}",
                f"Novelty@{k}",
                f"Diversity@{k}",
            ]
        )

    results_over_folds = {
        metric: []
        for metric in metric_names
    }

    completed_folds = 0

    for fold_index, (
        train_indices,
        test_indices,
    ) in enumerate(
        kfold.split(df_ratings),
        start=1,
    ):

        logger.info(
            "Running cross-validation fold %d/%d",
            fold_index,
            folds,
        )

        train_df = (
            df_ratings
            .iloc[train_indices]
            .copy()
        )

        test_df = (
            df_ratings
            .iloc[test_indices]
            .copy()
        )

        train_users = set(
            train_df["user_id"]
            .astype(int)
            .unique()
        )

        train_items = set(
            train_df["movie_id"]
            .astype(int)
            .unique()
        )

        test_df = test_df[
            test_df["user_id"]
            .astype(int)
            .isin(train_users)
            &
            test_df["movie_id"]
            .astype(int)
            .isin(train_items)
        ].copy()

        if test_df.empty:

            logger.warning(
                "Fold %d has no evaluable interactions. "
                "Skipping.",
                fold_index,
            )

            continue

        try:

            model = model_class(
                alpha=2.0
            )

        except TypeError:

            model = model_class()

        if hasattr(model, "build_model"):

            model.build_model(
                train_df,
                df_movies,
            )

        elif hasattr(model, "fit"):

            model.fit(train_df)

        else:

            raise ValueError(
                "Model has neither "
                "build_model() nor fit()."
            )

        fold_results, _ = evaluate_model(
            model=model,
            train_df=train_df,
            test_df=test_df,
            df_movies=df_movies,
            k_values=k_values,
            threshold=threshold,
            max_users=max_users,
            random_state=(
                random_state + fold_index
            ),
        )

        if not fold_results:

            logger.warning(
                "Fold %d produced no evaluation results.",
                fold_index,
            )

            continue

        completed_folds += 1

        for metric in metric_names:

            if metric in fold_results:

                results_over_folds[
                    metric
                ].append(
                    fold_results[metric]
                )

    if completed_folds == 0:

        raise RuntimeError(
            "Cross validation could not evaluate "
            "any fold."
        )

    final_results = {}

    for metric, values in (
        results_over_folds.items()
    ):

        final_results[metric] = (
            float(np.mean(values))
            if values
            else 0.0
        )

    final_results["EvaluatedFolds"] = (
        completed_folds
    )

    return final_results
