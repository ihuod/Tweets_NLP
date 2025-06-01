import os

# Get project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Paths to data
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Paths to files
TRAIN_PATH = os.path.join(RAW_DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(RAW_DATA_DIR, 'test.csv') 