"""
Module for text classification using scikit-learn Pipeline.
"""

import numpy as np
from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.utils.constants import (
    DEFAULT_VECTORIZER_PARAMS,
    DEFAULT_CLASSIFIER_PARAMS,
    DEFAULT_TOP_FEATURES,
    DEFAULT_N_JOBS
)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Base class for text preprocessing.
    You can inherit from it to create your own preprocessors.
    """
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'TextPreprocessor':
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

def create_text_classification_pipeline(
    vectorizer: BaseEstimator | None = None,
    classifier: ClassifierMixin | None = None,
    preprocessors: list[tuple[str, BaseEstimator]] | None = None,
    vectorizer_params: dict[str, Any] | None = None,
    classifier_params: dict[str, Any] | None = None,
    n_jobs: int = DEFAULT_N_JOBS
) -> Pipeline:
    """
    Creates a universal pipeline for text classification.
    
    Args:
        vectorizer (Optional[BaseEstimator]): Text vectorizer (e.g. TfidfVectorizer)
        classifier (Optional[ClassifierMixin]): Classifier
        preprocessors (Optional[list[tuple[str, BaseEstimator]]]): list of preprocessors in format (name, object)
        vectorizer_params (Optional[dict[str, Any]]): Parameters for vectorizer
        classifier_params (Optional[dict[str, Any]]): Parameters for classifier
        n_jobs (int): Number of parallel processes
        
    Returns:
        Pipeline: Pipeline for text classification
    """
    # Use TfidfVectorizer by default, if no other vectorizer is specified
    if vectorizer is None:
        params = DEFAULT_VECTORIZER_PARAMS.copy()
        if vectorizer_params:
            params.update(vectorizer_params)
        vectorizer = TfidfVectorizer(**params)
    elif vectorizer_params:
        vectorizer.set_params(**vectorizer_params)
    
    # Use LogisticRegression by default, if no other classifier is specified
    if classifier is None:
        params = DEFAULT_CLASSIFIER_PARAMS.copy()
        if classifier_params:
            params.update(classifier_params)
        classifier = LogisticRegression(**params)
    elif classifier_params:
        classifier.set_params(**classifier_params)
    
    # Create pipeline steps
    steps = []
    
    # Add preprocessors, if they are specified
    if preprocessors:
        steps.extend(preprocessors)
    
    # Add vectorizer
    steps.append(('vectorizer', vectorizer))
    
    # Add classifier
    steps.append(('classifier', classifier))
    
    return Pipeline(steps)

def get_feature_names(pipeline: Pipeline) -> list[str]:
    """
    Get feature names from pipeline.
    
    Args:
        pipeline (Pipeline): Trained pipeline
        
    Returns:
        list[str]: list of feature names
    """
    return pipeline.named_steps['vectorizer'].get_feature_names_out()

def get_feature_importance(
    pipeline: Pipeline,
    top_n: int = DEFAULT_TOP_FEATURES
) -> dict[str, float]:
    """
    Get feature importance from pipeline.
    
    Args:
        pipeline (Pipeline): Trained pipeline
        top_n (int): Number of top features
        
    Returns:
        dict[str, float]: dictionary with feature importance
    """
    # Get feature names and coefficients
    feature_names = get_feature_names(pipeline)
    classifier = pipeline.named_steps['classifier']
    
    # Check if classifier has coef_ attribute
    if hasattr(classifier, 'coef_'):
        coefficients = classifier.coef_[0]
        # Create dictionary with feature importance
        feature_importance = dict(zip(feature_names, np.abs(coefficients)))
        
        # Sort by importance and take top N
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return dict(sorted_features)
    else:
        raise AttributeError(
            f"Classifier {classifier.__class__.__name__} does not have feature importance"
        )
