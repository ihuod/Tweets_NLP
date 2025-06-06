"""
Пакет для работы с моделями классификации текста.
"""

from .text_classifier import (
    create_text_classification_pipeline,
    get_feature_names,
    get_feature_importance,
    TextPreprocessor
)

from .model_trainer import ModelTrainer

from .hyperparameter_tuning import (
    create_study,
    optuna_tune,
    find_optimal_text_classifier_params
)

__all__ = [
    # text_classifier
    'create_text_classification_pipeline',
    'get_feature_names',
    'get_feature_importance',
    'TextPreprocessor',
    
    # model_trainer
    'ModelTrainer',
    
    # hyperparameter_tuning
    'create_study',
    'optuna_tune',
    'find_optimal_text_classifier_params'
] 