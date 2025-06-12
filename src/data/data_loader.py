import pandas as pd
import sys
import os
from IPython.display import display
import logging
import numpy as np

from src.utils.paths import TRAIN_PATH, TEST_PATH, PROJECT_ROOT

# Get project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def load_data(train_path: str = TRAIN_PATH, test_path: str = TEST_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads training and test datasets.
    
    Args:
        train_path (str): Path to training dataset
        test_path (str): Path to test dataset
        
    Returns:
        tuple: (df_train, df_test) - tuple containing training and test datasets
    """
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    
    return df_train, df_test

def process_ids(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.Series, 
                                                                        pd.Series, 
                                                                        pd.DataFrame, 
                                                                        pd.DataFrame
                                                                        ]:
    """
    Extracts and removes ID columns from datasets.
    
    Args:
        df_train (pd.DataFrame): Training dataset
        df_test (pd.DataFrame): Test dataset
        
    Returns:
        tuple: (id_train, id_test, df_train, df_test) - IDs and processed datasets
    """
    id_train = df_train['id']
    id_test = df_test['id']
    
    df_train.drop(columns=['id'], inplace=True)
    df_test.drop(columns=['id'], inplace=True)
    
    return id_train, id_test, df_train, df_test

def display_data_info(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """
    Displays information about loaded datasets.
    
    Args:
        df_train (pd.DataFrame): Training dataset
        df_test (pd.DataFrame): Test dataset
    """
    logging.info("\n === Train dataset ===\n")
    logging.info(f"Shape: {df_train.shape}")
    logging.info("\nFirst 5 rows:")
    logging.info(df_train.head())
    logging.info(df_train.info())
    display(df_train.head())
    
    logging.info("\n === Test dataset ===\n")
    logging.info(f"Shape: {df_test.shape}")
    logging.info("\nFirst 5 rows:")
    logging.info(df_test.head())
    logging.info(df_test.info())
    display(df_test.head())

def save_submission(
    id_test: pd.Series,
    y_pred: np.ndarray,
    filename: str = 'submission.csv',
    submission_dir: str = 'data/submission'
) -> None:
    """
    Save model predictions to a submission file.
    
    Args:
        id_test (pd.Series): Test dataset IDs
        y_pred (np.ndarray): Model predictions
        filename (str): Name of the submission file. Default is 'submission.csv'
        submission_dir (str): Directory to save submission file. Default is 'data/submission'
        
    Returns:
        None
    """
    # Create submission directory if it doesn't exist
    submission_dir = os.path.join(PROJECT_ROOT, submission_dir)
    os.makedirs(submission_dir, exist_ok=True)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': id_test,
        'target': y_pred
    })
    
    # Save to CSV
    submission_path = os.path.join(submission_dir, filename)
    submission.to_csv(submission_path, index=False)
    
    logging.info(f"Submission saved to {submission_path}")
    logging.info(f"Submission shape: {submission.shape}")
    logging.info("\nFirst 5 rows of submission:")
    logging.info(submission.head())
    display(submission.head())