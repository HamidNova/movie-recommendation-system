# Movie Recommender System

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/Zero-Day-Hero/MovieRecommender?style=social)](https://github.com/Zero-Day-Hero/MovieRecommender)

A complete hybrid movie recommendation engine, built as a final‑year B.Sc. project in Computer Engineering.  
It combines content‑based and collaborative filtering, neural collaborative filtering, and a from‑scratch SVD with a rich evaluation suite and an interactive web interface.

---

## What's New
- **Neural Collaborative Filtering (NCF)** – deep learning model for user‑item interaction
- **SVD from Scratch** – matrix factorization implemented with NumPy/SGD
- **Baseline models** – Popular & Random for performance comparison
- **OMDb integration** – film posters, year, and genre in the UI
- **Surprise Me mode** – discover hidden gems
- **Recommendation history** – track past suggestions
- **Docker support** – fully containerized for reproducibility

---

## Features
- Content‑based filtering (19 genres + TF‑IDF on title)
- Collaborative filtering with ALS (implicit feedback), NCF, and manual SVD
- Hybrid recommender with adjustable weight and per‑suggestion reasons
- Cold‑start handling (Bayesian average for new users)
- Evaluation: Precision@k, Recall@k, NDCG@k, Coverage, Novelty, Diversity
- 5 EDA plots + model comparison chart
- Streamlit app with movie search, card layout, Surprise Me, feedback, and history
- Full logging, configuration, and Dockerfile

---

## Project Structure
```
MovieRecommender/
├── data/                     # (auto‑created)
├── models/                   # Saved models
├── logs/                     # Execution logs
├── reports/figures/          # EDA and comparison plots
├── tests/                    # Unit tests
├── pictures/                 # Icons & screenshots
│
├── src/
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── content_based.py      # Content‑based recommender
│   ├── collaborative.py      # ALS collaborative recommender
│   ├── ncf.py                # Neural Collaborative Filtering
│   ├── svd.py                # SVD from scratch
│   ├── baseline.py           # Popular / Random baselines
│   ├── hybrid.py             # Hybrid combination
│   ├── evaluation.py         # Metrics and evaluation
│   ├── visualization.py      # Plots
│   └── utils.py              # Logger, save/load, cold‑start
│
├── app/
│   ├── streamlit_app.py      # Interactive UI
│   └── omdb_service.py       # OMDb API wrapper
│
├── config.py                 # Hyperparameters
├── main.py                   # Full pipeline
├── Dockerfile                # Docker image definition
├── docker-compose.yml        # Multi‑service orchestration
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
pip install -r requirements.txt
python main.py            # train & evaluate
streamlit run app/streamlit_app.py   # launch UI
```

### Docker
```bash
docker-compose up --build
```
The Streamlit app will be available at `http://localhost:8501`.

---

## Evaluation Results
*Sample results on the test set (threshold=3.5)*

| Model          | Precision@5 | Recall@5 | NDCG@5 | Coverage |
|----------------|-------------|----------|--------|----------|
| ALS            | 0.42        | 0.21     | 0.47   | 0.62     |
| NCF            | 0.39        | 0.19     | 0.43   | 0.58     |
| SVD (scratch)  | 0.38        | 0.18     | 0.41   | 0.60     |
| Popular        | 0.35        | 0.16     | 0.38   | 0.05     |
| Random         | 0.05        | 0.03     | 0.05   | 0.98     |

---

## Screenshots
![Rating Distribution](reports/figures/rating_distribution.png)
![Model Comparison](reports/figures/model_comparison.png)
![Streamlit App](pictures/screenshot_streamlit.png)

---

## License & Contact
MIT License.  
**Author:** Zero-Day-Hero  
**Email:** hamidrezamirzaei8363@gmail.com  
**GitHub:** [github.com/Zero-Day-Hero](https://github.com/Zero-Day-Hero)