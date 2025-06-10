"""
Package for feature extraction and analysis.
Contains functions for:
- Extracting text features
- Analyzing sentiment
- Counting text statistics
"""

from .feature_engineering import (
    add_word_count,
    add_unique_word_count,
    add_stop_word_count,
    add_url_count,
    add_mean_word_length,
    add_char_count,
    add_punctuation_count,
    add_hashtag_count,
    add_mention_count,
    add_excl_count,
    add_vader_scores,
    add_emoji_count,
    add_caps_count,
    add_all_text_features,
    plot_metafeatures_distribution
)

from .feature_selection import FeatureImportanceAnalyzer

__all__ = [
    'add_word_count',
    'add_unique_word_count',
    'add_stop_word_count',
    'add_url_count',
    'add_mean_word_length',
    'add_char_count',
    'add_punctuation_count',
    'add_hashtag_count',
    'add_mention_count',
    'add_excl_count',
    'add_vader_scores',
    'add_emoji_count',
    'add_caps_count',
    'add_all_text_features',
    'plot_metafeatures_distribution',
    'FeatureImportanceAnalyzer'
] 