# Project configuration
MODEL_PATH = 'models/'
DATA_PATH = 'data/'
LOG_PATH = 'logs/'

# ALS hyperparameters (replacement for SVD)
ALS_PARAMS = {
    'factors': 100,
    'iterations': 30,
    'regularization': 0.05,
    'random_state': 42
}

# Evaluation settings
TEST_SIZE = 0.2
K_VALUES = [5, 10]
CROSS_VALIDATION_FOLDS = 5

# Default content weight in hybrid recommender
DEFAULT_CONTENT_WEIGHT = 0.4
