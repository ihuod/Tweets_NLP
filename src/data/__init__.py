"""
Data loading and analysis module.
Contains functions for:
- Loading data (data_loader.py)
- Processing data (data_processor.py)
- Exploring data (data_explorer.py)
"""

# Import commonly used functions
from .data_loader import load_data, process_ids, display_data_info
from .data_processor import preprocess_dataframe, remove_duplicates, fill_missing_values
from .data_explorer import print_keyword_location_analysis_results, analyze_ngrams

# Define what to import with "from src.data import *"
__all__ = [
    'load_data',
    'process_ids',
    'display_data_info',
    'preprocess_dataframe',
    'remove_duplicates',
    'fill_missing_values',
    'print_keyword_location_analysis_results',
    'analyze_ngrams'
] 