import numpy as np
from src.evaluation import precision_at_k, recall_at_k, ndcg_at_k

def test_precision_at_k():
    preds = {
        1: [(101, 0.9, 5), (102, 0.8, 4), (103, 0.7, 3)],
    }
    p = precision_at_k(preds, k=2, threshold=3.5)
    assert p == 1.0  # both relevant

def test_recall_at_k():
    preds = {
        1: [(101, 0.9, 5), (102, 0.8, 4), (103, 0.7, 3)],
    }
    r = recall_at_k(preds, k=2, threshold=3.5)
    assert r == 1.0  # all relevant in top-2

def test_ndcg_at_k():
    preds = {
        1: [(101, 0.9, 5), (102, 0.8, 4), (103, 0.7, 3)],
    }
    test_ratings = {1: {101: 5, 102: 4, 103: 3}}
    ndcg = ndcg_at_k(preds, test_ratings, k=3, threshold=3.5)
    assert ndcg == 1.0  # all relevant and perfect order
