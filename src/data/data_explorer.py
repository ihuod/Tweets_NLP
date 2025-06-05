import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .data_processor import fill_missing_values
from collections import defaultdict
from src.utils.constants import STOPWORDS, IS_DISASTER, TOP_NGRAMS_COUNT

def analyze_missing_values(df_train: pd.DataFrame, df_test: pd.DataFrame, columns: list[str] | None = None) -> dict[str, float]:
    """
    Analyze missing values in specified columns of train and test datasets.
    
    Args:
        df_train (pd.DataFrame): Training dataset
        df_test (pd.DataFrame): Test dataset
        columns (list[str] | None): List of columns to analyze. If None, analyzes all columns.
        
    Returns:
        dict[str, float]: Dictionary with missing value statistics in percentages
    """
    # Use all columns if none specified
    if columns is None:
        columns = list(set(df_train.columns) & set(df_test.columns))
    
    datasets = {'train': df_train, 'test': df_test}
    
    # Analyze missing values using functional approach
    missing_stats = (
        pd.Series(datasets)
        .apply(lambda df: {
            col: (df[col].isnull().sum() / len(df)) * 100 
            for col in columns
            if col in df.columns  # Проверяем наличие колонки
        })
        .to_dict()
    )
    
    # Flatten nested dictionary and add dataset prefix
    return {
        f'{dataset}_{col}_missing': percent 
        for dataset, stats in missing_stats.items() 
        for col, percent in stats.items()
    }

def analyze_unique_values(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame,
    columns: list[str] | None = None
) -> dict[str, int]:
    """
    Analyze unique values in specified columns of datasets using functional approach.
    
    Args:
        df_train (pd.DataFrame): Training dataset
        df_test (pd.DataFrame): Test dataset
        columns (list[str] | None): List of columns to analyze.
            If None, uses ['keyword', 'location']
        
    Returns:
        dict[str, int]: Dictionary with unique values statistics
    """
    # Use default columns if none provided
    columns = columns or ['keyword', 'location']
    
    # Create dictionary with dataset names and dataframes
    datasets = {'train': df_train, 'test': df_test}
    
    # Analyze unique values using functional approach
    unique_stats = (
        pd.Series(datasets)
        .apply(lambda df: {f'{col}_unique': df[col].nunique() for col in columns})
        .to_dict()
    )
    
    # Flatten nested dictionary and add dataset prefix
    return {
        f'{dataset}_{col}': count 
        for dataset, stats in unique_stats.items() 
        for col, count in stats.items()
    }

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
    for key, value in missing_stats.items():
        dataset, col, _ = key.split('_')
        print(f"\n{dataset.upper()}:")
        print(f"{col}: {value:.2f}%")
            
    print("\nUnique Values Analysis:")
    for key, value in unique_stats.items():
        print(f"{key}: {value}")
    
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
        
    Returns:
        None: Displays the plot
    """
    # Create subplot figure with proper spacing
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=(
            f'Top {TOP_NGRAMS_COUNT} {ngram_type} in Disaster Tweets',
            f'Top {TOP_NGRAMS_COUNT} {ngram_type} in Non-disaster Tweets'
        ),
        horizontal_spacing=0.25
    )

    # Add disaster n-grams bar chart
    fig.add_trace(
        go.Bar(
            y=disaster_ngrams_df.head(TOP_NGRAMS_COUNT)['ngram'],
            x=disaster_ngrams_df.head(TOP_NGRAMS_COUNT)['count'],
            name='Disaster',
            marker_color=disaster_color,
            orientation='h'
        ),
        row=1, col=1
    )

    # Add non-disaster n-grams bar chart
    fig.add_trace(
        go.Bar(
            y=nondisaster_ngrams_df.head(TOP_NGRAMS_COUNT)['ngram'],
            x=nondisaster_ngrams_df.head(TOP_NGRAMS_COUNT)['count'],
            name='Non-disaster',
            marker_color=nondisaster_color,
            orientation='h'
        ),
        row=1, col=2
    )

    # Update layout with proper margins and spacing
    fig.update_layout(
        height=figsize[1] * 10,
        width=figsize[0] * 50,
        showlegend=False,
        title_font_size=title_fontsize,
        font=dict(size=y_labelsize_trigrams if (ngram_type == 'trigrams' or ngram_type == 'bigrams') else y_labelsize),
        margin=dict(l=20, r=20, t=40, b=20),
        bargap=0.2,
        bargroupgap=0.1
    )

    # Update axes with proper spacing and labels
    fig.update_xaxes(
        title_text='Count',
        row=1,
        col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray'
    )
    fig.update_xaxes(
        title_text='Count',
        row=1,
        col=2,
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray'
    )
    fig.update_yaxes(
        title_text='',
        row=1,
        col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray'
    )
    fig.update_yaxes(
        title_text='',
        row=1,
        col=2,
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray'
    )

    # Show the plot
    fig.show()

def analyze_ngrams(df_train: pd.DataFrame) -> tuple[pd.DataFrame, 
                                                    pd.DataFrame, 
                                                    pd.DataFrame, 
                                                    pd.DataFrame, 
                                                    pd.DataFrame, 
                                                    pd.DataFrame]:
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