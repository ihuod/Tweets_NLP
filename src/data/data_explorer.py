import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, Tuple
from .data_processor import fill_missing_values
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from src.utils.constants import STOPWORDS, IS_DISASTER, TOP_NGRAMS_COUNT

def analyze_missing_values(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze missing values in datasets.
    
    Args:
        df_train (pd.DataFrame): Training dataset
        df_test (pd.DataFrame): Test dataset
        
    Returns:
        Dict[str, float]: Dictionary with missing values statistics
    """
    missing_stats = {}
    
    for name, df in [('train', df_train), ('test', df_test)]:
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        missing_stats[f'{name}_missing'] = missing_percent.to_dict()
        
    return missing_stats

def analyze_unique_values(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, int]:
    """
    Analyze unique values in datasets.
    
    Args:
        df_train (pd.DataFrame): Training dataset
        df_test (pd.DataFrame): Test dataset
        
    Returns:
        Dict[str, int]: Dictionary with unique values statistics
    """
    unique_stats = {}
    
    for name, df in [('train', df_train), ('test', df_test)]:
        for col in ['keyword', 'location']:
            unique_stats[f'{name}_{col}_unique'] = df[col].nunique()
            
    return unique_stats

def print_keyword_location_analysis_results(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """
    Print comprehensive analysis of datasets with visualizations.
    
    Args:
        df_train (pd.DataFrame): Training dataset
        df_test (pd.DataFrame): Test dataset
    """
    # Get initial statistics
    missing_stats = analyze_missing_values(df_train, df_test)
    unique_stats = analyze_unique_values(df_train, df_test)
    
    # Print initial statistics
    print("Missing Values Analysis:")
    for name, stats in missing_stats.items():
        print(f"\n{name.upper()}:")
        for col, percent in stats.items():
            print(f"{col}: {percent:.2f}%")
            
    print("\nUnique Values Analysis:")
    for name, count in unique_stats.items():
        print(f"{name}: {count}")
    
    # Fill missing values
    df_train, df_test = fill_missing_values(df_train, df_test)
    
    # Create visualizations
    # Top 20 keywords
    keyword_counts = df_train['keyword'].value_counts().head(20)
    fig_keywords = px.bar(
        x=keyword_counts.index,
        y=keyword_counts.values,
        title='Top 20 Keywords in Training Set',
        labels={'x': 'Keyword', 'y': 'Count'},
        color=keyword_counts.values,
        color_continuous_scale='Viridis'
    )
    fig_keywords.update_layout(
        xaxis_tickangle=-45,
        showlegend=False
    )
    fig_keywords.show()
    
    # Top 20 locations
    location_counts = df_train['location'].value_counts().head(20)
    fig_locations = px.bar(
        x=location_counts.index,
        y=location_counts.values,
        title='Top 20 Locations in Training Set',
        labels={'x': 'Location', 'y': 'Count'},
        color=location_counts.values,
        color_continuous_scale='Viridis'
    )
    fig_locations.update_layout(
        xaxis_tickangle=-45,
        showlegend=False
    )
    fig_locations.show()
    
    # Print final counts
    print("\nAfter filling missing values:")
    print("\nKeyword counts (top 10):")
    print(keyword_counts[:10])
    print("\nLocation counts (top 10):")
    print(location_counts[:10]) 

    # Target distribution in train dataset by keywords
    df_train['target_mean'] = df_train.groupby('keyword')['target'].transform('mean')

    df_sorted = df_train.sort_values(by='target_mean', ascending=False)

    fig = px.histogram(df_sorted, 
                    y = 'keyword',
                    color = 'target',
                    orientation = 'h',
                    height = 3000,
                    width = 800,
                    title = 'Target Distribution in Keywords',
                    color_discrete_sequence = px.colors.qualitative.Pastel)

    fig.update_layout(
        yaxis = dict(autorange = "reversed", tickfont = dict(size = 10)),
        legend = dict(x = 1, y = 1),
        xaxis = dict(tickfont = dict(size = 15))
    )

    df_train.drop(columns = ['target_mean'], inplace = True)

    fig.show()

def generate_ngrams(text, n_gram=1, stopwords=None):
    """
    Generate n-grams from text.
    
    Args:
        text (str): Input text
        n_gram (int): Size of n-grams (1 for unigrams, 2 for bigrams, etc.)
        stopwords (set): Set of stopwords to exclude
        
    Returns:
        list: List of n-grams
    """
    if stopwords is None:
        stopwords = set()
    tokens = [token for token in text.lower().split(' ') if token != '' and token not in stopwords]
    ngrams = zip(*[tokens[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]

def get_top_ngrams(tweets, n_gram=1, stopwords=None, n_top=TOP_NGRAMS_COUNT):
    """
    Get top n-grams from a list of tweets.
    
    Args:
        tweets (pd.Series): Series of tweets
        n_gram (int): Size of n-grams
        stopwords (set): Set of stopwords to exclude
        n_top (int): Number of top n-grams to return
        
    Returns:
        pd.DataFrame: DataFrame with top n-grams and their counts
    """
    ngram_counts = defaultdict(int)
    for tweet in tweets:
        for ngram in generate_ngrams(tweet, n_gram, stopwords):
            ngram_counts[ngram] += 1
    sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
    return pd.DataFrame(sorted_ngrams[:n_top], columns=['ngram', 'count'])

def plot_ngrams_comparison(
        disaster_ngrams_df,
        nondisaster_ngrams_df,
        ngram_type, 
        n_cols=2, 
        figsize=(18, 50), 
        dpi=100, 
        disaster_color='red', 
        nondisaster_color='green', 
        x_labelsize=13, 
        y_labelsize=13, 
        title_fontsize=15, 
        y_labelsize_trigrams=11
):
    """
    Plot comparison of n-grams between disaster and non-disaster tweets.
    
    Args:
        disaster_ngrams_df (pd.DataFrame): DataFrame with disaster n-grams
        nondisaster_ngrams_df (pd.DataFrame): DataFrame with non-disaster n-grams
        ngram_type (str): Type of n-grams ('unigrams', 'bigrams', 'trigrams')
        n_cols (int): Number of columns in the plot
        figsize (tuple): Figure size
        dpi (int): DPI of the figure
        disaster_color (str): Color for disaster n-grams
        nondisaster_color (str): Color for non-disaster n-grams
        x_labelsize (int): Font size for x-axis labels
        y_labelsize (int): Font size for y-axis labels
        title_fontsize (int): Font size for titles
        y_labelsize_trigrams (int): Font size for y-axis labels in trigrams plot
        
    Returns:
        None: Displays the plot
    """
    fig, axes = plt.subplots(ncols=n_cols, figsize=figsize, dpi=dpi)
    plt.tight_layout()
    
    y_tick_size = y_labelsize_trigrams if ngram_type == 'trigrams' else y_labelsize
    
    sns.barplot(
        y='ngram', 
        x='count', 
        data=disaster_ngrams_df.head(TOP_NGRAMS_COUNT), 
        ax=axes[0], 
        color=disaster_color
    )
    sns.barplot(
        y='ngram', 
        x='count', 
        data=nondisaster_ngrams_df.head(TOP_NGRAMS_COUNT), 
        ax=axes[1], 
        color=nondisaster_color
    )
    
    for i in range(n_cols):
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=x_labelsize)
        axes[i].tick_params(axis='y', labelsize=y_tick_size)
    
    axes[0].set_title(f'Top {TOP_NGRAMS_COUNT} most common {ngram_type} in Disaster Tweets', fontsize=title_fontsize)
    axes[1].set_title(f'Top {TOP_NGRAMS_COUNT} most common {ngram_type} in Non-disaster Tweets', fontsize=title_fontsize)
    
    plt.show()

def analyze_ngrams(df_train):
    """
    Analyze n-grams in disaster and non-disaster tweets.
    
    Args:
        df_train (pd.DataFrame): Training dataframe
        
    Returns:
        tuple: DataFrames with top unigrams, bigrams, and trigrams for both disaster and non-disaster tweets
    """
    disaster_tweets = df_train['target'] == IS_DISASTER
    
    # Get top n-grams
    df_disaster_unigrams = get_top_ngrams(df_train[disaster_tweets]['text'], n_gram=1, stopwords=STOPWORDS)
    df_nondisaster_unigrams = get_top_ngrams(df_train[~disaster_tweets]['text'], n_gram=1, stopwords=STOPWORDS)
    
    df_disaster_bigrams = get_top_ngrams(df_train[disaster_tweets]['text'], n_gram=2, stopwords=STOPWORDS)
    df_nondisaster_bigrams = get_top_ngrams(df_train[~disaster_tweets]['text'], n_gram=2, stopwords=STOPWORDS)
    
    df_disaster_trigrams = get_top_ngrams(df_train[disaster_tweets]['text'], n_gram=3, stopwords=STOPWORDS)
    df_nondisaster_trigrams = get_top_ngrams(df_train[~disaster_tweets]['text'], n_gram=3, stopwords=STOPWORDS)
    
    # Plot comparisons
    plot_ngrams_comparison(df_disaster_unigrams, df_nondisaster_unigrams, 'unigrams')
    plot_ngrams_comparison(df_disaster_bigrams, df_nondisaster_bigrams, 'bigrams')
    plot_ngrams_comparison(
        df_disaster_trigrams, 
        df_nondisaster_trigrams, 
        'trigrams', 
        figsize=(20, 50), 
        y_labelsize_trigrams=11
    )
    
    return (
        df_disaster_unigrams, df_nondisaster_unigrams,
        df_disaster_bigrams, df_nondisaster_bigrams,
        df_disaster_trigrams, df_nondisaster_trigrams
    )