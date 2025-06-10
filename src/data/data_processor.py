import pandas as pd
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

def fill_missing_values(
    datasets: dict[str, pd.DataFrame],
    columns: list[str] | None = None,
    default_prefix: str = 'no_'
) -> dict[str, pd.DataFrame]:
    """
    Fill missing values in specified columns across multiple datasets.
    
    Args:
        datasets (dict[str, pd.DataFrame]): Dictionary with dataset names as keys and DataFrames as values
        columns (list[str] | None): List of columns to fill missing values in.
            If None, uses ['keyword', 'location']
        default_prefix (str): Prefix for default values. Default is 'no_'
        
    Returns:
        dict[str, pd.DataFrame]: Dictionary with processed datasets with filled missing values
        
    Example:
        >>> datasets = {
        ...     'train': df_train,
        ...     'test': df_test,
        ...     'validation': df_val
        ... }
        >>> processed_datasets = fill_missing_values(datasets)
    """
    # Use default columns if none provided
    columns = columns or ['keyword', 'location']
    
    # Create dictionary for filling missing values
    fill_vals = {col: f'{default_prefix}{col}' for col in columns}
    
    # Fill missing values in specified columns for each dataset using pandas Series
    return pd.Series(datasets).apply(lambda df: df.fillna(fill_vals)).to_dict()

def remove_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows from dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        subset (list[str], optional): list of columns to consider for duplicates
        
    Returns:
        tuple[pd.DataFrame, int]: Processed dataframe and number of removed duplicates
    """
    initial_len = len(df)
    df = df.drop_duplicates(subset=subset, keep='first')
    removed_count = initial_len - len(df)
    return df, removed_count

def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        
    Returns:
        tuple[pd.DataFrame, pd.Series]: Features and target
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """
    Get list of categorical columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        list[str]: list of categorical column names
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def get_numerical_columns(df: pd.DataFrame) -> list[str]:
    """
    Get list of numerical columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        list[str]: list of numerical column names
    """
    return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def clean_text(text: str | float) -> str:
    """
    Clean text by removing URLs, mentions, hashtags, special characters and unknown symbols.
    
    Args:
        text (str | float): Text to clean
        
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

def clean_texts(texts: pd.Series | list[str]) -> pd.Series:
    """
    Clean multiple texts.
    
    Args:
        texts (pd.Series | list[str]): Texts to clean
        
    Returns:
        pd.Series: Cleaned texts
    """
    if isinstance(texts, list):
        texts = pd.Series(texts)
    return texts.apply(clean_text)