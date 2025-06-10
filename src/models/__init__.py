"""
Package for text classification models.
Contains functions for:
- Creating text classification pipelines
- Hyperparameter tuning
- Model evaluation
"""

from .text_classifier import (
    create_text_classification_pipeline,
    get_feature_names,
    get_feature_importance,
    TextPreprocessor
)

from .hyperparameter_tuning import (
    create_study,
    optuna_tune,
    find_optimal_text_classifier_params,
    optimize_hyperparameters
)

from .model_evaluation import (
    cross_validate_model,
    nested_cross_validate_model
)

__all__ = [
    # text_classifier
    'create_text_classification_pipeline',
    'get_feature_names',
    'get_feature_importance',
    'TextPreprocessor',
    
    # hyperparameter_tuning
    'create_study',
    'optuna_tune',
    'find_optimal_text_classifier_params',
    'optimize_hyperparameters',

    # model_evaluation
    'cross_validate_model',
    'nested_cross_validate_model'
] 