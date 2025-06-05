import numpy as np
import string
import emoji
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from src.utils.constants import (
    STOPWORDS, 
    DEFAULT_TEXT_COLUMN, 
    METAFEATURES, 
    IS_DISASTER
)

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

def add_word_count(df: pd.DataFrame, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    """
    Add word count to the text.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column
        
    Returns:
        pd.DataFrame: Dataframe with new word_count column
    """
    df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
    return df

def add_unique_word_count(df: pd.DataFrame, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    """
    Add unique word count to the text.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column
        
    Returns:
        pd.DataFrame: Dataframe with new unique_word_count column
    """
    df['unique_word_count'] = df[text_column].apply(lambda x: len(set(str(x).split())))
    return df

def add_stop_word_count(df: pd.DataFrame, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    """
    Add stop word count to the text.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column
        
    Returns:
        pd.DataFrame: Dataframe with new stop_word_count column
    """
    df['stop_word_count'] = df[text_column].apply(
        lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    return df

def add_url_count(df: pd.DataFrame, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    """
    Add URL count to the text.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column
        
    Returns:
        pd.DataFrame: Dataframe with new url_count column
    """
    df['url_count'] = df[text_column].apply(
        lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
    return df

def add_mean_word_length(df: pd.DataFrame, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    """
    Add mean word length to the text.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column
        
    Returns:
        pd.DataFrame: Dataframe with new mean_word_length column
    """
    df['mean_word_length'] = df[text_column].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]))
    return df

def add_char_count(df: pd.DataFrame, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    """
    Add character count to the text.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column
        
    Returns:
        pd.DataFrame: Dataframe with new char_count column
    """
    df['char_count'] = df[text_column].apply(lambda x: len(str(x)))
    return df

def add_punctuation_count(df: pd.DataFrame, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    """
    Add punctuation count to the text.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column
        
    Returns:
        pd.DataFrame: Dataframe with new punctuation_count column
    """
    df['punctuation_count'] = df[text_column].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation]))
    return df

def add_hashtag_count(df: pd.DataFrame, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    """
    Add hashtag count to the text.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column
        
    Returns:
        pd.DataFrame: Dataframe with new hashtag_count column
    """
    df['hashtag_count'] = df[text_column].apply(lambda x: len([c for c in str(x) if c == '#']))
    return df

def add_mention_count(df: pd.DataFrame, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    """
    Add mention count to the text.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column
        
    Returns:
        pd.DataFrame: Dataframe with new mention_count column
    """
    df['mention_count'] = df[text_column].apply(lambda x: len([c for c in str(x) if c == '@']))
    return df

def add_excl_count(df: pd.DataFrame, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    """
    Add exclamation mark count to the text.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column
        
    Returns:
        pd.DataFrame: Dataframe with new excl_count column
    """
    df['excl_count'] = df[text_column].apply(lambda x: str(x).count('!'))
    return df

def add_vader_scores(df: pd.DataFrame, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    """
    Add VADER sentiment scores to the text.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column
        
    Returns:
        pd.DataFrame: Dataframe with new vader_neg, vader_neu, vader_pos, vader_compound columns
    """
    sentiments = df[text_column].apply(lambda x: sid.polarity_scores(str(x)))
    df['vader_neg'] = sentiments.apply(lambda x: x['neg'])
    df['vader_neu'] = sentiments.apply(lambda x: x['neu'])
    df['vader_pos'] = sentiments.apply(lambda x: x['pos'])
    df['vader_compound'] = sentiments.apply(lambda x: x['compound'])
    return df

def add_emoji_count(df: pd.DataFrame, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    """
    Add emoji count to the text.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column
        
    Returns:
        pd.DataFrame: Dataframe with new emoji_count column
    """
    df['emoji_count'] = df[text_column].apply(lambda x: emoji.emoji_count(str(x)))
    return df

def add_caps_count(df: pd.DataFrame, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    """
    Add count of words in capital letters to the text.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column
        
    Returns:
        pd.DataFrame: Dataframe with new caps_count column
    """
    df['caps_count'] = df[text_column].apply(
        lambda x: len([w for w in str(x).split() if w.isupper() and len(w) > 1]))
    return df

def add_all_text_features(df: pd.DataFrame, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    """
    Add all text features to the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column
        
    Returns:
        pd.DataFrame: Dataframe with all added features
    """
    df = add_word_count(df, text_column)
    df = add_unique_word_count(df, text_column)
    df = add_stop_word_count(df, text_column)
    df = add_url_count(df, text_column)
    df = add_mean_word_length(df, text_column)
    df = add_char_count(df, text_column)
    df = add_punctuation_count(df, text_column)
    df = add_hashtag_count(df, text_column)
    df = add_mention_count(df, text_column)
    df = add_excl_count(df, text_column)
    df = add_vader_scores(df, text_column)
    df = add_emoji_count(df, text_column)
    df = add_caps_count(df, text_column)
    
    return df 

def plot_metafeatures_distribution(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """
    Plot distribution of meta-features for disaster and non-disaster tweets,
    and compare distributions between training and test sets.
    
    Args:
        df_train (pd.DataFrame): Training dataframe
        df_test (pd.DataFrame): Test dataframe
        
    Returns:
        None: Displays the plots
    """
    disaster_tweets = df_train['target'] == IS_DISASTER
    
    fig, axes = plt.subplots(ncols=2, nrows=len(METAFEATURES), figsize=(40, 100), dpi=100)
    
    for i, feature in enumerate(METAFEATURES):
        # Plot distribution for disaster vs non-disaster tweets
        sns.displot(df_train.loc[~disaster_tweets][feature], label='Not Disaster', ax=axes[i][0], color='green')
        sns.displot(df_train.loc[disaster_tweets][feature], label='Disaster', ax=axes[i][0], color='red')
        
        # Plot distribution for training vs test sets
        sns.displot(df_train[feature], label='Training', ax=axes[i][1])
        sns.displot(df_test[feature], label='Test', ax=axes[i][1])
        
        # Customize plots
        for j in range(2):
            axes[i][j].set_xlabel('')
            axes[i][j].tick_params(axis='x', labelsize=20)
            axes[i][j].tick_params(axis='y', labelsize=20)
            axes[i][j].legend()
        
        axes[i][0].set_title(f'{feature} Target Distribution in Training Set', fontsize=20)
        axes[i][1].set_title(f'{feature} Training & Test Set Distribution', fontsize=20)
    
    plt.tight_layout()
    plt.show() 