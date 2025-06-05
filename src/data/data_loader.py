import pandas as pd
import sys
import os
from IPython.display import display
import logging

# Add src to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.paths import TRAIN_PATH, TEST_PATH

def load_data():
    """
    Loads training and test datasets.
    
    Returns:
        tuple: (df_train, df_test) - tuple containing training and test datasets
    """
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    
    return df_train, df_test

def process_ids(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.Series, 
                                                                        pd.Series, 
                                                                        pd.DataFrame, 
                                                                        pd.DataFrame]:
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