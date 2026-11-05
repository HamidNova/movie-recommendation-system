MODEL_PATH = 'models/'
DATA_PATH = 'data/'
LOG_PATH = 'logs/'


# ============================================================
# Movie Genre Columns
# ============================================================

GENRE_COLUMNS = [
    'unknown',
    'Action',
    'Adventure',
    'Animation',
    "Children's",
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Fantasy',
    'Film-Noir',
    'Horror',
    'Musical',
    'Mystery',
    'Romance',
    'Sci-Fi',
    'Thriller',
    'War',
    'Western'
]


# ============================================================
# General Recommendation Settings
# ============================================================

DEFAULT_TOP_K = 5
DEFAULT_TITLE_WEIGHT = 0.2
DEFAULT_CONTENT_WEIGHT = 0.4


# ============================================================
# Collaborative Filtering (ALS)
# ============================================================

ALS_PARAMS = {
    'factors': 100,
    'iterations': 30,
    'regularization': 0.05,
    'random_state': 42
}


# ============================================================
# Evaluation Settings
# ============================================================

TEST_SIZE = 0.2
K_VALUES = [5, 10]
CROSS_VALIDATION_FOLDS = 3


# ============================================================
# Baseline Settings
# ============================================================

POPULAR_MIN_RATINGS = 10
RANDOM_SEED = 42


# ============================================================
# Fuzzy C-Means Settings
# ============================================================

FCM_PARAMS = {
    'n_clusters': 5,
    'm': 1.5,
    'max_iter': 300,
    'error': 1e-5,
    'random_state': 42
}


# ============================================================
# Neural Hybrid Model Settings
# ============================================================

HYBRID_NN_PARAMS = {
    'hidden_units': [128, 64],
    'learning_rate': 0.001,
    'epochs': 20,
    'batch_size': 128
}