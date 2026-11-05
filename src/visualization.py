import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set,
    dataframe_name: str,
):
    """
    Validate that a DataFrame contains all required columns.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"{dataframe_name} must be a pandas DataFrame."
        )

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        missing_columns_text = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            f"{dataframe_name} is missing required columns: "
            f"{missing_columns_text}"
        )


def plot_eda(
    df_ratings: pd.DataFrame,
    df_movies: pd.DataFrame,
    save_dir="reports/figures",
):
    """
    Generate and save exploratory data analysis plots.

    Required rating columns:
    - user_id
    - movie_id
    - rating

    Required movie columns:
    - movie_id
    - title
    """

    _validate_columns(
        dataframe=df_ratings,
        required_columns={
            "user_id",
            "movie_id",
            "rating",
        },
        dataframe_name="df_ratings",
    )

    _validate_columns(
        dataframe=df_movies,
        required_columns={
            "movie_id",
            "title",
        },
        dataframe_name="df_movies",
    )

    if df_ratings.empty or df_movies.empty:
        return

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    saved_plots = 0

    # ============================================================
    # 1. Rating Distribution
    # ============================================================

    fig, ax = plt.subplots()

    rating_order = sorted(
        df_ratings["rating"]
        .dropna()
        .unique()
    )

    sns.countplot(
        data=df_ratings,
        x="rating",
        order=rating_order,
        color="steelblue",
        ax=ax,
    )

    ax.set_title(
        "Distribution of User Ratings",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")

    for patch in ax.patches:

        height = patch.get_height()

        if height > 0:

            ax.annotate(
                f"{int(height)}",
                (
                    patch.get_x()
                    + patch.get_width() / 2.0,
                    height,
                ),
                ha="center",
                va="bottom",
            )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            save_dir,
            "rating_distribution.png",
        ),
        dpi=150,
    )

    plt.close(fig)

    saved_plots += 1

    # ============================================================
    # 2. Ratings per User
    # ============================================================

    user_ratings = (
        df_ratings
        .groupby("user_id")["rating"]
        .count()
    )

    fig, ax = plt.subplots()

    bins = min(
        50,
        max(1, len(user_ratings)),
    )

    ax.hist(
        user_ratings,
        bins=bins,
        edgecolor="black",
        alpha=0.7,
    )

    ax.set_title(
        "Number of Ratings per User",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Number of Users")

    median_ratings = (
        user_ratings.median()
        if not user_ratings.empty
        else 0
    )

    ax.axvline(
        median_ratings,
        color="red",
        linestyle="--",
        label=(
            f"Median: "
            f"{median_ratings:.0f}"
        ),
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            save_dir,
            "ratings_per_user.png",
        ),
        dpi=150,
    )

    plt.close(fig)

    saved_plots += 1

    # ============================================================
    # 3. Most Popular Movies
    # ============================================================

    top_movies = (
        df_ratings
        .groupby("movie_id")["rating"]
        .count()
        .sort_values(ascending=False)
        .head(10)
    )

    titles_map = (
        df_movies
        .drop_duplicates(
            subset=["movie_id"],
            keep="first",
        )
        .set_index("movie_id")["title"]
        .astype(str)
        .to_dict()
    )

    top_movie_names = [
        titles_map.get(
            movie_id,
            f"Movie {movie_id}",
        )
        for movie_id in top_movies.index
    ]

    fig, ax = plt.subplots()

    ax.barh(
        top_movie_names,
        top_movies.values,
        color="teal",
    )

    ax.set_xlabel("Number of Ratings")

    ax.set_title(
        "Top 10 Most Watched Movies",
        fontsize=14,
        fontweight="bold",
    )

    ax.invert_yaxis()

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            save_dir,
            "popular_movies.png",
        ),
        dpi=150,
    )

    plt.close(fig)

    saved_plots += 1

    # ============================================================
    # 4. Average Rating Distribution
    # ============================================================

    average_ratings = (
        df_ratings
        .groupby("movie_id")["rating"]
        .mean()
    )

    fig, ax = plt.subplots()

    bins = min(
        20,
        max(1, len(average_ratings)),
    )

    ax.hist(
        average_ratings,
        bins=bins,
        edgecolor="black",
        alpha=0.7,
        color="coral",
    )

    ax.set_title(
        "Distribution of Average Ratings per Movie",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Number of Movies")

    average_rating_mean = (
        average_ratings.mean()
        if not average_ratings.empty
        else 0
    )

    ax.axvline(
        average_rating_mean,
        color="blue",
        linestyle="--",
        label=(
            f"Mean: "
            f"{average_rating_mean:.2f}"
        ),
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            save_dir,
            "avg_rating_dist.png",
        ),
        dpi=150,
    )

    plt.close(fig)

    saved_plots += 1

    # ============================================================
    # 5. Sample Ratings Heatmap
    # ============================================================

    unique_users = (
        df_ratings["user_id"]
        .dropna()
        .unique()
    )

    unique_movies = (
        df_ratings["movie_id"]
        .dropna()
        .unique()
    )

    if (
        len(unique_users) > 0
        and len(unique_movies) > 0
    ):

        rng = np.random.default_rng(42)

        sample_users = rng.choice(
            unique_users,
            size=min(
                20,
                len(unique_users),
            ),
            replace=False,
        )

        sample_movies = rng.choice(
            unique_movies,
            size=min(
                20,
                len(unique_movies),
            ),
            replace=False,
        )

        sample = df_ratings[
            (
                df_ratings["user_id"]
                .isin(sample_users)
            )
            &
            (
                df_ratings["movie_id"]
                .isin(sample_movies)
            )
        ]

        if not sample.empty:

            pivot = sample.pivot_table(
                index="user_id",
                columns="movie_id",
                values="rating",
                aggfunc="mean",
            )

            if not pivot.empty:

                fig, ax = plt.subplots(
                    figsize=(14, 10)
                )

                sns.heatmap(
                    pivot,
                    cmap="coolwarm",
                    annot=True,
                    fmt=".1f",
                    linewidths=0.5,
                    ax=ax,
                )

                ax.set_title(
                    (
                        "Sample Ratings Heatmap "
                        f"({pivot.shape[0]} users x "
                        f"{pivot.shape[1]} movies)"
                    ),
                    fontsize=14,
                    fontweight="bold",
                )

                fig.tight_layout()

                fig.savefig(
                    os.path.join(
                        save_dir,
                        "heatmap_sample.png",
                    ),
                    dpi=150,
                )

                plt.close(fig)

                saved_plots += 1

    print(
        f"{saved_plots} EDA plots saved in "
        f"{save_dir}"
    )


def plot_comparison(
    results_dict: dict,
    metric_names: list,
    save_dir="reports/figures",
):
    """
    Create and save a bar chart comparing model results.

    Parameters
    ----------
    results_dict:
        Dictionary where keys are model names and values
        are dictionaries containing evaluation metrics.

    metric_names:
        List of metric names to include in the comparison.

    save_dir:
        Directory where the plot will be saved.
    """

    if not isinstance(results_dict, dict):
        raise TypeError(
            "results_dict must be a dictionary."
        )

    if not isinstance(metric_names, list):
        raise TypeError(
            "metric_names must be a list."
        )

    if not results_dict:
        return

    if not metric_names:
        return

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    results_df = pd.DataFrame(
        results_dict
    ).T

    valid_metrics = [
        metric
        for metric in metric_names
        if metric in results_df.columns
    ]

    if not valid_metrics:
        return

    results_df = results_df[
        valid_metrics
    ]

    results_df = results_df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    results_df = results_df.dropna(
        axis=1,
        how="all",
    )

    if results_df.empty:
        return

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    results_df.plot(
        kind="bar",
        ax=ax,
        colormap="viridis",
    )

    ax.set_title(
        "Model Comparison",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_ylabel("Score")

    ax.legend(
        loc="best"
    )

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.3f",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            save_dir,
            "model_comparison.png",
        ),
        dpi=150,
    )

    plt.close(fig)

    print(
        f"Comparison plot saved in "
        f"{save_dir}"
    )
