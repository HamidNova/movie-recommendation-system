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
