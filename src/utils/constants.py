from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Combine stopwords from NLTK and scikit-learn
STOPWORDS = set(ENGLISH_STOP_WORDS).union(set(stopwords.words('english')))

# List of contractions
CONTRACTIONS = {
    'arent', 'couldnt', 'didnt', 'doesnt', 'dont', 'hadnt', 'hasnt', 'havent',
    'hed', 'hell', 'hes', 'id', 'ill', 'im', 'isnt', 'itd', 'itll', 'ive',
    'mightnt', 'mustnt', 'neednt', 'shant', 'shed', 'shell', 'shes',
    'shouldnt', 'shouldve', 'thatll', 'theyd', 'theyll', 'theyre', 'theyve',
    'wasnt', 'wed', 'well', 'werent', 'weve', 'wont', 'wouldnt', 'youd',
    'youll', 'youre', 'youve'
}

# Add contractions to stopwords
STOPWORDS.update(CONTRACTIONS)

# Default text column name
DEFAULT_TEXT_COLUMN = 'text'

# Default target column name
DEFAULT_TARGET_COLUMN = 'target'

# Default random state for reproducibility
RANDOM_STATE = 42

# Default test size for train-test split
TEST_SIZE = 0.2

# Default validation size for train-validation split
VAL_SIZE = 0.1

# Value indicating disaster tweet
IS_DISASTER = 1

# List of meta-features for analysis
METAFEATURES = [
    'word_count',
    'unique_word_count',
    'stop_word_count',
    'url_count',
    'mean_word_length',
    'char_count',
    'punctuation_count',
    'hashtag_count',
    'mention_count',
    'vader_neg',
    'vader_neu',
    'vader_pos',
    'vader_compound',
    'emoji_count',
    'caps_count',
    'excl_count'
]

# Number of top n-grams to analyze
TOP_NGRAMS_COUNT = 25

# Logistic Regression default parameters
LOGISTIC_REGRESSION_PARAMS = {
    'C': 1.0,  # Inverse of regularization strength
    'penalty': 'l2',  # Regularization penalty
    'solver': 'liblinear',  # Algorithm to use in optimization
    'max_iter': 1000,  # Maximum number of iterations
    'random_state': RANDOM_STATE,  # Random state for reproducibility
    'n_jobs': -1,  # Use all available cores
    'class_weight': 'balanced'  # Adjust weights inversely proportional to class frequencies
}

# Hyperparameter Search Space Constants
TFIDF_PARAM_RANGES = {
    'max_features': (1000, 15000, 1000),  # (min, max, step)
    'ngram_range': [(1, 1), (1, 2), (1, 3)],
    'min_df': (1, 10),
    'max_df': (0.7, 1.0)
}

# Logistic Regression parameter ranges for hyperparameter tuning
LOGISTIC_REGRESSION_PARAM_RANGES = {
    'C': (1e-3, 100.0, True),  # (min, max, log=True)
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg', 'sag'],
    'max_iter': (100, 2000, 100),  # (min, max, step)
    'class_weight': [None, 'balanced'],
    'l1_ratio': (0.0, 1.0)  # для elasticnet
}

# Model Training Constants
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_JOBS = -1
DEFAULT_CV_FOLDS = 5
DEFAULT_INNER_CV_FOLDS = 3
DEFAULT_OUTER_CV_FOLDS = 5
DEFAULT_N_TRIALS = 20
DEFAULT_SCORING = 'f1'

# Cross-validation strategies
CV_STRATEGIES = {
    'stratified': True,  # Use stratified cross-validation by default
    'shuffle': True,     # Shuffle data before splitting
}

# Metrics to calculate
MODEL_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc'
]

# Model file extension
MODEL_FILE_EXTENSION = '.joblib'

# Text Classification Constants
DEFAULT_VECTORIZER_PARAMS = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.95,
    'sublinear_tf': True
}

DEFAULT_CLASSIFIER_PARAMS = {
    'C': 1.0,
    'max_iter': 1000,
    'n_jobs': DEFAULT_N_JOBS,
    'random_state': DEFAULT_RANDOM_STATE
}

# Feature importance constants
DEFAULT_TOP_FEATURES = 20

# Hyperparameter Tuning Constants
DEFAULT_OPTIMIZATION_DIRECTION = 'maximize'
DEFAULT_OPTIMIZATION_TRIALS = 100
DEFAULT_OPTIMIZATION_STORAGE = None
DEFAULT_OPTIMIZATION_LOAD_IF_EXISTS = False
DEFAULT_OPTIMIZATION_STUDY_NAME = 'optimization'
DEFAULT_OPTIMIZATION_SHOW_PROGRESS = True 