import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .data_processor import fill_missing_values
from collections import defaultdict
from src.utils.constants import STOPWORDS, IS_DISASTER, TOP_NGRAMS_COUNT, TOP_KEYWORDS_COUNT, TOP_LOCATIONS_COUNT

def analyze_missing_values_single_df(
    df: pd.DataFrame,
    columns: list[str] | None = None
) -> dict[str, float]:
    """
    Analyze missing values in specified columns of a single dataset.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        columns (list[str] | None): List of columns to analyze. If None, analyzes all columns.
        
    Returns:
        dict[str, float]: Dictionary with missing value statistics in percentages
    """
    # Use all columns if none specified
    if columns is None:
        columns = df.columns
    
    # Analyze missing values
    return df[columns].apply(
        lambda x: (x.isna().sum() / len(x)) * 100,
        raw=True
    ).to_dict()

def analyze_missing_values(
    datasets: dict[str, pd.DataFrame],
    columns: list[str] | None = None
) -> dict[str, float]:
    """
    Analyze missing values in specified columns across multiple datasets.
    
    Args:
        datasets (dict[str, pd.DataFrame]): Dictionary with dataset names as keys and DataFrames as values
        columns (list[str] | None): List of columns to analyze. If None, analyzes common columns across all datasets.
        
    Returns:
        dict[str, float]: Dictionary with missing value statistics in percentages
        
    Example:
        >>> datasets = {
        ...     'train': df_train,
        ...     'test': df_test,
        ...     'validation': df_val
        ... }
        >>> missing_stats = analyze_missing_values(datasets)
    """
    # Use common columns if none specified
    if columns is None:
        columns = list(set.intersection(*[set(df.columns) for df in datasets.values()]))
    
    # Convert datasets to Series and calculate missing values
    missing_stats = (
        pd.Series(datasets)
        .apply(lambda df: (df[columns].isnull().sum() / len(df) * 100).to_dict())
        .to_dict()
    )
    
    # Flatten nested dictionary and add dataset prefix
    return {
        f'{dataset}_{col}_missing': percent 
        for dataset, stats in missing_stats.items() 
        for col, percent in stats.items()
    }

def analyze_unique_values_single_df(
    df: pd.DataFrame,
    columns: list[str] | None = None
) -> dict[str, int]:
    """
    Analyze unique values in specified columns of a single dataset.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        columns (list[str] | None): List of columns to analyze.
            If None, uses ['keyword', 'location']
        
    Returns:
        dict[str, int]: Dictionary with unique values statistics
    """
    # Use default columns if none provided
    unique_counts = df[columns].apply(
        lambda x: len(np.unique(x)),
        raw=True
    )
    
    return {f'{col}_unique': count
            for col, count in unique_counts.items()
            if col in df.columns}

def analyze_unique_values(
    datasets: dict[str, pd.DataFrame],
    columns: list[str] | None = None
) -> dict[str, int]:
    """
    Analyze unique values in specified columns across multiple datasets.
    
    Args:
        datasets (dict[str, pd.DataFrame]): Dictionary with dataset names as keys and DataFrames as values
        columns (list[str] | None): List of columns to analyze.
            If None, uses ['keyword', 'location']
        
    Returns:
        dict[str, int]: Dictionary with unique values statistics
        
    Example:
        >>> datasets = {
        ...     'train': df_train,
        ...     'test': df_test,
        ...     'validation': df_val
        ... }
        >>> unique_stats = analyze_unique_values(datasets)
    """
    # Use default columns if none provided
    columns = columns or ['keyword', 'location']
    
    # Convert datasets to Series and calculate unique values
    unique_stats = (
        pd.Series(datasets)
        .apply(lambda df: df[columns].nunique().to_dict())
        .to_dict()
    )
    
    # Flatten nested dictionary and add dataset prefix
    return {
        f'{dataset}_{col}': count 
        for dataset, stats in unique_stats.items() 
        for col, count in stats.items()
    }

def plot_keyword_distribution(df: pd.DataFrame, n_top: int = TOP_KEYWORDS_COUNT) -> None:
    """
    Plot distribution of top keywords.
    
    Args:
        df (pd.DataFrame): DataFrame with keyword column
        n_top (int): Number of top keywords to show
    """
    keyword_counts = df['keyword'].value_counts().head(n_top)
    fig = px.bar(
        x=keyword_counts.index,
        y=keyword_counts.values,
        title=f'Top {n_top} Keywords',
        labels={'x': 'Keyword', 'y': 'Count'},
        color=keyword_counts.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False
    )
    fig.show()
    return keyword_counts

def plot_location_distribution(df: pd.DataFrame, n_top: int = TOP_LOCATIONS_COUNT) -> None:
    """
    Plot distribution of top locations.
    
    Args:
        df (pd.DataFrame): DataFrame with location column
        n_top (int): Number of top locations to show
    """
    location_counts = df['location'].value_counts().head(n_top)
    fig = px.bar(
        x=location_counts.index,
        y=location_counts.values,
        title=f'Top {n_top} Locations',
        labels={'x': 'Location', 'y': 'Count'},
        color=location_counts.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False
    )
    fig.show()
    return location_counts

def plot_target_distribution_by_keywords(df: pd.DataFrame) -> None:
    """
    Plot target distribution by keywords.
    
    Args:
        df (pd.DataFrame): DataFrame with keyword and target columns
    """
    df = df.copy()
    df['target_mean'] = df.groupby('keyword')['target'].transform('mean')
    df_sorted = df.sort_values(by='target_mean', ascending=False)

    fig = px.histogram(
        df_sorted, 
        y='keyword',
        color='target',
        orientation='h',
        height=3000,
        width=800,
        title='Target Distribution in Keywords',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        legend=dict(x=1, y=1),
        xaxis=dict(tickfont=dict(size=15))
    )
    fig.show()

def print_keyword_location_analysis_results(
    datasets: dict[str, pd.DataFrame],
    train_dataset_name: str = 'train'
) -> None:
    """
    Print comprehensive analysis of datasets with visualizations.
    
    Args:
        datasets (dict[str, pd.DataFrame]): Dictionary with dataset names as keys and DataFrames as values
        train_dataset_name (str): Name of the training dataset in the dictionary. Default is 'train'
        
    Example:
        >>> datasets = {
        ...     'train': df_train,
        ...     'test': df_test,
        ...     'validation': df_val
        ... }
        >>> print_keyword_location_analysis_results(datasets)
    """
    # Get initial statistics
    missing_stats = analyze_missing_values(datasets)
    unique_stats = analyze_unique_values(datasets)
    
    # Print initial statistics
    print("Missing Values Analysis:")
    for key, value in missing_stats.items():
        dataset, col, _ = key.split('_')
        print(f"\n{dataset.upper()}:")
        print(f"{col}: {value:.2f}%")
            
    print("\nUnique Values Analysis:")
    for key, value in unique_stats.items():
        print(f"{key}: {value}")
    
    # Fill missing values
    processed_datasets = fill_missing_values(datasets)
    df_train = processed_datasets[train_dataset_name]
    
    # Create visualizations
    keyword_counts = plot_keyword_distribution(df_train)
    location_counts = plot_location_distribution(df_train)
    
    # Print final counts
    print("\nAfter filling missing values:")
    print(f"\nKeyword counts (top {TOP_KEYWORDS_COUNT}):")
    print(keyword_counts[:TOP_KEYWORDS_COUNT])
    print(f"\nLocation counts (top {TOP_LOCATIONS_COUNT}):")
    print(location_counts[:TOP_LOCATIONS_COUNT]) 

    # Plot target distribution
    plot_target_distribution_by_keywords(df_train)

def generate_ngrams(text: str, n_gram: int = 1, stopwords: set | None = None) -> list[str]:
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

def get_top_ngrams(tweets: pd.Series, n_gram: int = 1, stopwords: set | None = None, n_top: int = TOP_NGRAMS_COUNT) -> pd.DataFrame:
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

def create_ngram_bar_trace(
    df: pd.DataFrame,
    name: str,
    color: str,
    n_top: int = TOP_NGRAMS_COUNT
) -> go.Bar:
    """
    Create a bar trace for n-grams visualization.
    
    Args:
        df (pd.DataFrame): DataFrame with n-grams data
        name (str): Name of the trace
        color (str): Color for the bars
        n_top (int): Number of top n-grams to show
        
    Returns:
        go.Bar: Bar trace for plotly
    """
    return go.Bar(
        y=df.head(n_top)['ngram'],
        x=df.head(n_top)['count'],
        name=name,
        marker_color=color,
        orientation='h'
    )

def create_ngrams_subplot(
    disaster_ngrams_df: pd.DataFrame,
    nondisaster_ngrams_df: pd.DataFrame,
    ngram_type: str,
    disaster_color: str = 'red',
    nondisaster_color: str = 'green'
) -> go.Figure:
    """
    Create a subplot comparing disaster and non-disaster n-grams.
    
    Args:
        disaster_ngrams_df (pd.DataFrame): DataFrame with disaster n-grams
        nondisaster_ngrams_df (pd.DataFrame): DataFrame with non-disaster n-grams
        ngram_type (str): Type of n-grams ('unigrams', 'bigrams', 'trigrams')
        disaster_color (str): Color for disaster n-grams
        nondisaster_color (str): Color for non-disaster n-grams
        
    Returns:
        go.Figure: Plotly figure with subplots
    """
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=(
            f'Top {TOP_NGRAMS_COUNT} {ngram_type} in Disaster Tweets',
            f'Top {TOP_NGRAMS_COUNT} {ngram_type} in Non-disaster Tweets'
        ),
        horizontal_spacing=0.25
    )

    # Add traces
    fig.add_trace(
        create_ngram_bar_trace(disaster_ngrams_df, 'Disaster', disaster_color),
        row=1, col=1
    )
    fig.add_trace(
        create_ngram_bar_trace(nondisaster_ngrams_df, 'Non-disaster', nondisaster_color),
        row=1, col=2
    )

    return fig

def update_ngrams_layout(
    fig: go.Figure,
    ngram_type: str,
    figsize: tuple[int, int] = (18, 50),
    y_labelsize: int = 13,
    title_fontsize: int = 15,
    y_labelsize_trigrams: int = 11
) -> None:
    """
    Update layout of n-grams plot.
    
    Args:
        fig (go.Figure): Plotly figure to update
        ngram_type (str): Type of n-grams ('unigrams', 'bigrams', 'trigrams')
        figsize (tuple): Figure size
        y_labelsize (int): Font size for y-axis labels
        title_fontsize (int): Font size for titles
        y_labelsize_trigrams (int): Font size for y-axis labels in trigrams plot
    """
    # Update layout
    fig.update_layout(
        height=figsize[1] * 10,
        width=figsize[0] * 50,
        showlegend=False,
        title_font_size=title_fontsize,
        font=dict(size=y_labelsize_trigrams if ngram_type in ['trigrams', 'bigrams'] else y_labelsize),
        margin=dict(l=20, r=20, t=40, b=20),
        bargap=0.2,
        bargroupgap=0.1
    )

    # Update axes
    for col in [1, 2]:
        fig.update_xaxes(
            title_text='Count',
            row=1,
            col=col,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        )
        fig.update_yaxes(
            title_text='',
            row=1,
            col=col,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        )

def plot_ngrams_comparison(
    disaster_ngrams_df: pd.DataFrame,
    nondisaster_ngrams_df: pd.DataFrame,
    ngram_type: str, 
    n_cols: int = 2, 
    figsize: tuple[int, int] = (18, 50), 
    dpi: int = 100, 
    disaster_color: str = 'red', 
    nondisaster_color: str = 'green', 
    x_labelsize: int = 13, 
    y_labelsize: int = 13, 
    title_fontsize: int = 15, 
    y_labelsize_trigrams: int = 11
) -> None:
    """
    Plot comparison of n-grams between disaster and non-disaster tweets using Plotly.
    
    Args:
        disaster_ngrams_df (pd.DataFrame): DataFrame with disaster n-grams
        nondisaster_ngrams_df (pd.DataFrame): DataFrame with non-disaster n-grams
        ngram_type (str): Type of n-grams ('unigrams', 'bigrams', 'trigrams')
        n_cols (int): Number of columns in the plot
        figsize (tuple): Figure size (not used in Plotly)
        dpi (int): DPI of the figure (not used in Plotly)
        disaster_color (str): Color for disaster n-grams
        nondisaster_color (str): Color for non-disaster n-grams
        x_labelsize (int): Font size for x-axis labels
        y_labelsize (int): Font size for y-axis labels
        title_fontsize (int): Font size for titles
        y_labelsize_trigrams (int): Font size for y-axis labels in trigrams plot
    """
    # Create subplot
    fig = create_ngrams_subplot(
        disaster_ngrams_df,
        nondisaster_ngrams_df,
        ngram_type,
        disaster_color,
        nondisaster_color
    )
    
    # Update layout
    update_ngrams_layout(
        fig,
        ngram_type,
        figsize,
        y_labelsize,
        title_fontsize,
        y_labelsize_trigrams
    )
    
    # Show the plot
    fig.show()

def analyze_ngrams(df_train: pd.DataFrame) -> tuple[pd.DataFrame, 
                                                    pd.DataFrame, 
                                                    pd.DataFrame, 
                                                    pd.DataFrame, 
                                                    pd.DataFrame, 
                                                    pd.DataFrame
                                                    ]:
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