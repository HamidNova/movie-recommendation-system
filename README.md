# Movie Recommender System Project

This project is designed for the final year internship course in Computer Engineering.

## Features
- Content-based filtering with TF-IDF
- Collaborative filtering with ALS (using the `implicit` library)
- Hybrid recommendation system
- Evaluation with Precision@k, Recall@k, Coverage
- EDA plots
- Streamlit user interface

## Installation and Execution
```bash
pip install -r requirements.txt
python main.py
streamlit run app/streamlit_app.py