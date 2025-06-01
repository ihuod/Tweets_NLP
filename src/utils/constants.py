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
TOP_NGRAMS_COUNT = 100 