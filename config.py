MODEL_PATH = 'models/'
DATA_PATH = 'data/'
LOG_PATH = 'logs/'

ALS_PARAMS = {
    'factors': 100,
    'iterations': 30,
    'regularization': 0.05,
    'random_state': 42
}

TEST_SIZE = 0.2
K_VALUES = [5, 10]
CROSS_VALIDATION_FOLDS = 5

DEFAULT_CONTENT_WEIGHT = 0.4

POPULAR_MIN_RATINGS = 10
RANDOM_SEED = 42

# Fuzzy C-Means parameters
FCM_PARAMS = {
    'n_clusters': 15,
    'm': 2.0,
    'max_iter': 100,
    'error': 0.005,
    'random_state': 42
}

# Hybrid Neural Network parameters (PyTorch)
HYBRID_NN_PARAMS = {
    'hidden_units': [128, 64],
    'learning_rate': 0.001,
    'epochs': 10,
    'batch_size': 64
}

OMDB_API_KEY = "1da7b640"
DEMO_SAMPLE_SIZE = 500000
