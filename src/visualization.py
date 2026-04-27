# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Optional

# تنظیم استایل
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12


def plot_eda(df_ratings: pd.DataFrame, df_movies: pd.DataFrame, save_dir='reports/figures'):
    """۵ نمودار EDA حرفه‌ای"""
    os.makedirs(save_dir, exist_ok=True)

    # 1. توزیع ریتینگ‌ها
    plt.figure()
    ax = sns.countplot(x='rating', data=df_ratings, palette='viridis')
    plt.title('Distribution of User Ratings', fontsize=14, fontweight='bold')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    # اضافه کردن عدد روی میله‌ها
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rating_distribution.png'), dpi=150)
    plt.close()

    # 2. تعداد ریتینگ به ازای کاربر
    user_ratings = df_ratings.groupby('user_id')['rating'].count()
    plt.figure()
    plt.hist(user_ratings, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Number of Ratings per User', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.axvline(user_ratings.median(), color='red', linestyle='--', label=f'Median: {user_ratings.median():.0f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ratings_per_user.png'), dpi=150)
    plt.close()

    # 3. ۱۰ فیلم پرمخاطب
    top_movies = df_ratings.groupby('movie_id')['rating'].count().sort_values(ascending=False).head(10)
    top_movies_names = df_movies.set_index('movie_id').loc[top_movies.index]['title']
    plt.figure()
    plt.barh(top_movies_names, top_movies.values, color='teal')
    plt.xlabel('Number of Ratings')
    plt.title('Top 10 Most Watched Movies', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'popular_movies.png'), dpi=150)
    plt.close()

    # 4. توزیع میانگین ریتینگ فیلم‌ها
    avg_ratings = df_ratings.groupby('movie_id')['rating'].mean()
    plt.figure()
    plt.hist(avg_ratings, bins=20, edgecolor='black', alpha=0.7, color='coral')
    plt.title('Distribution of Average Ratings per Movie', fontsize=14, fontweight='bold')
    plt.xlabel('Average Rating')
    plt.ylabel('Number of Movies')
    plt.axvline(avg_ratings.mean(), color='blue', linestyle='--', label=f'Mean: {avg_ratings.mean():.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'avg_rating_dist.png'), dpi=150)
    plt.close()

    # 5. هیت مپ نمونه (۲۰ کاربر و ۲۰ فیلم تصادفی)
    sample_users = np.random.choice(df_ratings['user_id'].unique(), min(20, df_ratings['user_id'].nunique()),
                                    replace=False)
    sample_movies = np.random.choice(df_ratings['movie_id'].unique(), min(20, df_ratings['movie_id'].nunique()),
                                     replace=False)
    sample = df_ratings[df_ratings['user_id'].isin(sample_users) & df_ratings['movie_id'].isin(sample_movies)]
    if not sample.empty:
        pivot = sample.pivot_table(index='user_id', columns='movie_id', values='rating')
        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot, cmap='coolwarm', annot=True, fmt='.0f', linewidths=0.5)
        plt.title('Sample Ratings Heatmap (20 users x 20 movies)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'heatmap_sample.png'), dpi=150)
        plt.close()

    print(f"✅ 5 EDA plots saved in {save_dir}")


def plot_comparison(results_dict: dict, metric_names: list, save_dir='reports/figures'):
    """
    مقایسه چند مدل (مثلاً collaborative, content, hybrid) روی معیارهای مختلف
    results_dict: {'model_name': {'Precision@5': 0.4, 'Recall@5': 0.3, ...}}
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(results_dict).T
    df = df[metric_names]
    ax = df.plot(kind='bar', figsize=(10, 6), colormap='viridis')
    plt.title('Model Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150)
    plt.close()
    print(f"✅ Comparison plot saved in {save_dir}")
