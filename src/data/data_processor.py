import pandas as pd
import numpy as np
from typing import Tuple, List, Union
import re

def preprocess_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Preprocess text column in dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column to preprocess
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    df = df.copy()
    df[text_column] = clean_texts(df[text_column])
    return df

def fill_missing_values(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill missing values in keyword and location columns.
    
    Args:
        df_train (pd.DataFrame): Training dataset
        df_test (pd.DataFrame): Test dataset
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed datasets
    """
    df_train = df_train.copy()
    df_test = df_test.copy()
    
    for df in [df_train, df_test]:
        for col in ['keyword', 'location']:
            df[col] = df[col].fillna(f'no_{col}')
            
    return df_train, df_test

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows from dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        subset (List[str], optional): List of columns to consider for duplicates
        
    Returns:
        Tuple[pd.DataFrame, int]: Processed dataframe and number of removed duplicates
    """
    initial_len = len(df)
    df = df.drop_duplicates(subset=subset, keep='first')
    removed_count = initial_len - len(df)
    return df, removed_count

def split_features_target(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of categorical columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        List[str]: List of categorical column names
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def get_numerical_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of numerical columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        List[str]: List of numerical column names
    """
    return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def clean_text(text: Union[str, float]) -> str:
    """
    Clean text by removing URLs, mentions, hashtags, special characters and unknown symbols.
    
    Args:
        text (Union[str, float]): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""
        
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Remove special characters and unknown symbols
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove unknown symbols (squares and U-like characters)
    text = re.sub(r'[\uFFFD\uFFFE\uFFFF]', '', text)  # Remove replacement characters
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)  # Remove zero-width spaces
    text = re.sub(r'[\u202A-\u202E]', '', text)  # Remove directional formatting
    text = re.sub(r'[\u2060-\u2064]', '', text)  # Remove word joiners and invisible separators
    
    return text.strip()

def clean_texts(texts: Union[pd.Series, List[str]]) -> pd.Series:
    """
    Clean multiple texts.
    
    Args:
        texts (Union[pd.Series, List[str]]): Texts to clean
        
    Returns:
        pd.Series: Cleaned texts
    """
    if isinstance(texts, list):
        texts = pd.Series(texts)
    return texts.apply(clean_text)