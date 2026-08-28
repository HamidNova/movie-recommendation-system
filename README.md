# Movie Recommender System with Fuzzy C-Means

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/HamidNova/movie-recommendation-system?style=social)](https://github.com/HamidNova/movie-recommendation-system)

A complete hybrid movie recommendation engine, built as a final-year B.Sc. project in Computer Engineering.  
It combines:

- **Collaborative filtering** with ALS
- **Content-based filtering** using genres and title TF-IDF
- **Fuzzy C-Means clustering** on latent item factors
- **Neural hybrid recommender** trained with fuzzy-weighted samples
- **Ensemble blending** of ALS and the fuzzy-aware neural network

The project includes a full evaluation suite, an interactive Streamlit interface, and a separate fuzzy clustering visualization page.

---

## What's New

- **Fuzzy C-Means (FCM)** clustering on ALS item embeddings
- **Neural hybrid recommender** that uses fuzzy membership vectors as features
- **Fuzzy-weighted training** to focus the network on items that match the user’s fuzzy profile
- **Ensemble model** that combines ALS scores with fuzzy-aware neural scores
- **Ablation study** comparing ALS, content-based, neural without fuzzy, fuzzy neural, and the final ensemble
- **Streamlit UI improvements**
  - Fuzzy cluster membership shown for every recommended movie
  - Dedicated **Fuzzy Clusters** page with top movies per cluster and borderline/mixed movies
- **Corrected Recall@k** metric (previously overestimated because denominator used only candidate items)

---

## Features

- Content-based filtering using 19 genres + TF-IDF on movie titles
- Collaborative filtering with ALS (implicit feedback, confidence weighting)
- Neural hybrid recommender built on PyTorch, using ALS user/item embeddings and fuzzy membership vectors
- Fuzzy C-Means clustering with standardized latent factors and cosine-aware preprocessing
- Hybrid engine with adjustable content-collaborative weight and per-recommendation reasons
- Cold-start handling using Bayesian average popularity
- Evaluation metrics:
  - Precision@k
  - Recall@k
  - NDCG@k
  - Coverage
  - Novelty
  - Diversity
- Visualization:
  - EDA plots
  - Model comparison charts
  - Fuzzy cluster membership display
- Interactive Streamlit app:
  - Movie search
  - Personalized recommendations
  - Surprise Me mode
  - Recommendation history
  - Fuzzy cluster explorer
  - Poster gallery (local assets)

---

## Project Structure

```
MovieRecommender/
├── data/                     # Auto-created dataset folder
├── assets/posters/           # Local movie posters
├── models/                   # Saved trained models
├── logs/                     # Execution logs
├── reports/figures/          # EDA and comparison plots
├── pictures/                 # Icons and screenshots
│
├── src/
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── content_based.py      # Content-based recommender
│   ├── collaborative.py      # ALS collaborative recommender
│   ├── fuzzy_clustering.py   # Fuzzy C-Means clustering
│   ├── neural_hybrid.py      # Neural hybrid recommender with fuzzy features
│   ├── baseline.py           # Popular / Random baselines
│   ├── hybrid.py             # Hybrid combination engine
│   ├── evaluation.py         # Metrics and evaluation
│   ├── visualization.py      # Plots and charts
│   └── utils.py              # Logger, save/load, cold-start helpers
│
├── app/
│   └── streamlit_app.py      # Interactive UI
│
├── config.py                 # Hyperparameters and configuration
├── main.py                   # Full training pipeline
├── ablation_study.py         # Model comparison and fuzzy impact analysis
├── download_posters_ddg.py   # Optional poster downloader
├── requirements.txt
├── pytest.ini                # Pytest configuration
├── .gitignore
└── README.md
```

---

## Installation & Usage

### Local

```bash
git clone https://github.com/Zero-Day-Hero/MovieRecommender.git
cd MovieRecommender
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train and evaluate models
python main.py

# Run ablation study (ALS vs fuzzy hybrid vs ensemble, etc.)
python ablation_study.py

# Launch interactive UI
streamlit run app/streamlit_app.py
```

The Streamlit app will be available at `http://localhost:8501`.

### Docker

```bash
docker-compose up --build
```

After the container starts, open `http://localhost:8501`.

---

## Dataset

The project uses the **MovieLens 100K** dataset.  
It contains:

- 100,000 ratings
- 943 users
- 1,682 movies

Each movie has 19 binary genre features and a title.  
Ratings are on a scale from 1 to 5.

---

## Methodology

### 1. Collaborative Filtering (ALS)

- Builds an implicit feedback matrix using confidence scores.
- Trains an Alternating Least Squares model to produce user and item latent factors.

### 2. Fuzzy C-Means Clustering

- Applied to the item latent factors from ALS.
- Data is standardized and normalized before clustering.
- The result is a fuzzy membership matrix `U` where each movie belongs to every cluster with a degree between 0 and 1.
- This gives a soft, interpretable grouping of movies.

### 3. Neural Hybrid Recommender

- Inputs:
  - User latent vector from ALS
  - Item latent vector from ALS
  - Element-wise product interaction
  - Fuzzy membership vector of the item
- The network is trained with **fuzzy-weighted MSE**:
  - Samples where the item’s fuzzy profile is closer to the user’s favorite cluster get higher weight.
  - This forces the network to pay more attention to content/taste similarity.

### 4. Final Ensemble

The final recommendation score for an item is a weighted blend:

```
final_score = 0.7 * normalized_ALS_score + 0.3 * normalized_neural_score
```

This preserves ALS ranking strength while allowing fuzzy information to act as a personalized tie-breaker and boost relevant items.

---

## Evaluation Results

The evaluation was performed on a hold-out test set (20% of ratings) with threshold `3.5`.  
The table below shows the final results after correcting Recall@k.

| Model                         | Precision@5 | Recall@5 | NDCG@5 | Precision@10 | Recall@10 | NDCG@10 |
|-------------------------------|-------------|----------|--------|--------------|-----------|---------|
| Random Baseline               | 0.004       | 0.001    | 0.005  | 0.004        | 0.002     | 0.005   |
| Popularity Baseline           | 0.055       | 0.020    | 0.050  | 0.050        | 0.035     | 0.052   |
| Content-Based                 | 0.036       | 0.014    | 0.038  | 0.032        | 0.023     | 0.037   |
| ALS (Collaborative)           | 0.308       | 0.155    | 0.350  | 0.256        | 0.220     | 0.338   |
| ALS + NN (W/O Fuzzy)          | 0.268       | 0.136    | 0.307  | 0.228        | 0.186     | 0.298   |
| ALS + Fuzzy + NN (Full)       | 0.282       | 0.137    | 0.318  | 0.227        | 0.179     | 0.296   |
| **ALS + NN Ensemble (Final)** | **0.322**   | 0.153    | **0.364** | **0.270**    | **0.222** | **0.351** |

**Key observations:**

- The final ensemble improves **Precision@5** from `0.308` to `0.322`.
- **NDCG@5** improves from `0.350` to `0.364`.
- This demonstrates that adding Fuzzy C-Means information through the proposed ensemble has a positive impact on recommendation quality.
- The neural network alone suffers from the regression-to-mean problem, which is why the ensemble approach is used.

---

## Fuzzy Clusters in the UI

The Streamlit app now includes a dedicated **Fuzzy Clusters** page.

### Top movies per cluster

For each cluster, the interface shows the movies with the highest membership values.  
Example:

| Movie | Membership |
|-------|------------|
| Fright Night Part II (1989) | 0.999 |
| Prom Night (1980) | 0.999 |
| Howling II: Your Sister Is a Werewolf (1985) | 0.999 |

### Borderline / Mixed Movies

Movies that belong to two or more clusters with similar membership values are displayed as borderline examples.  
This makes the fuzzy nature of the clustering directly visible.

Example of borderline movies:

| Movie | Top Cluster | Top Membership | Second Cluster | Second Membership | Difference |
|-------|-------------|----------------|----------------|-------------------|------------|
| Bonnie and Clyde (1967) | Cluster 1 | 0.221 | Cluster 4 | 0.221 | 0.000 |
| Rumble in the Bronx (1995) | Cluster 2 | 0.235 | Cluster 4 | 0.235 | 0.000 |
| Gone with the Wind (1939) | Cluster 4 | 0.220 | Cluster 0 | 0.219 | 0.001 |

---

## Screenshots

![Rating Distribution](reports/figures/rating_distribution.png)  
![Model Comparison](reports/figures/model_comparison.png)  
![Streamlit App](pictures/screenshot_streamlit.png)

---

## License & Contact

MIT License.

**Author:** Hamidreza Mirzaei  
**Email:** hamidrezamirzaei8363@gmail.com  
**GitHub:** [github.com/HamidNova](https://github.com/Zero-Day-Hero)