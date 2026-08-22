import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12


def plot_eda(df_ratings: pd.DataFrame, df_movies: pd.DataFrame, save_dir='reports/figures'):
    if df_ratings.empty or df_movies.empty:
        return

    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    ax = sns.countplot(x='rating', hue='rating', data=df_ratings, palette='viridis', legend=False)
    plt.title('Distribution of User Ratings', fontsize=14, fontweight='bold')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f'{int(h)}', (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rating_distribution.png'), dpi=150)
    plt.close()

    user_ratings = df_ratings.groupby('user_id')['rating'].count()
    plt.figure()
    plt.hist(user_ratings, bins=min(50, max(1, len(user_ratings))), edgecolor='black', alpha=0.7)
    plt.title('Number of Ratings per User', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    med = user_ratings.median() if not user_ratings.empty else 0
    plt.axvline(med, color='red', linestyle='--', label=f'Median: {med:.0f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ratings_per_user.png'), dpi=150)
    plt.close()

    top_movies = df_ratings.groupby('movie_id')['rating'].count().sort_values(ascending=False).head(10)
    titles_map = df_movies.drop_duplicates(subset=['movie_id']).set_index('movie_id')['title'].to_dict()
    top_movies_names = [titles_map.get(mid, f"Movie {mid}") for mid in top_movies.index]

    plt.figure()
    plt.barh(top_movies_names, top_movies.values, color='teal')
    plt.xlabel('Number of Ratings')
    plt.title('Top 10 Most Watched Movies', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'popular_movies.png'), dpi=150)
    plt.close()

    avg_ratings = df_ratings.groupby('movie_id')['rating'].mean()
    plt.figure()
    plt.hist(avg_ratings, bins=min(20, max(1, len(avg_ratings))), edgecolor='black', alpha=0.7, color='coral')
    plt.title('Distribution of Average Ratings per Movie', fontsize=14, fontweight='bold')
    plt.xlabel('Average Rating')
    plt.ylabel('Number of Movies')
    avg_mean = avg_ratings.mean() if not avg_ratings.empty else 0
    plt.axvline(avg_mean, color='blue', linestyle='--', label=f'Mean: {avg_mean:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'avg_rating_dist.png'), dpi=150)
    plt.close()

    unique_users = df_ratings['user_id'].unique()
    unique_movies = df_ratings['movie_id'].unique()

    sample_users = np.random.choice(unique_users, min(20, len(unique_users)), replace=False)
    sample_movies = np.random.choice(unique_movies, min(20, len(unique_movies)), replace=False)

    sample = df_ratings[df_ratings['user_id'].isin(sample_users) & df_ratings['movie_id'].isin(sample_movies)]
    if not sample.empty:
        pivot = sample.pivot_table(index='user_id', columns='movie_id', values='rating')
        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot, cmap='coolwarm', annot=True, fmt='.0f', linewidths=0.5)
        plt.title('Sample Ratings Heatmap (20 users x 20 movies)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'heatmap_sample.png'), dpi=150)
        plt.close()

    print(f"5 EDA plots saved in {save_dir}")


def plot_comparison(results_dict: dict, metric_names: list, save_dir='reports/figures'):
    if not results_dict:
        return

    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(results_dict).T

    valid_metrics = [m for m in metric_names if m in df.columns]
    if not valid_metrics:
        return

    df = df[valid_metrics]
    ax = df.plot(kind='bar', figsize=(10, 6), colormap='viridis')
    plt.title('Model Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Score')
    plt.legend(loc='best')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150)
    plt.close()

    print(f"Comparison plot saved in {save_dir}")
