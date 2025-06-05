"""
Module for hyperparameter tuning using Optuna.
"""

import optuna
from typing import Any, Callable
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
from src.utils.constants import (
    DEFAULT_OPTIMIZATION_DIRECTION,
    DEFAULT_OPTIMIZATION_TRIALS,
    DEFAULT_OPTIMIZATION_STORAGE,
    DEFAULT_OPTIMIZATION_LOAD_IF_EXISTS,
    DEFAULT_OPTIMIZATION_STUDY_NAME,
    DEFAULT_OPTIMIZATION_SHOW_PROGRESS,
    DEFAULT_CV_FOLDS,
    DEFAULT_SCORING,
    DEFAULT_N_JOBS,
    CV_STRATEGIES,
    DEFAULT_RANDOM_STATE
)
from src.models.text_classifier import create_text_classification_pipeline

def create_study(
    study_name: str = DEFAULT_OPTIMIZATION_STUDY_NAME,
    direction: str = DEFAULT_OPTIMIZATION_DIRECTION,
    storage: str | None = DEFAULT_OPTIMIZATION_STORAGE,
    load_if_exists: bool = DEFAULT_OPTIMIZATION_LOAD_IF_EXISTS
) -> optuna.Study:
    """Create or load an Optuna study."""
    return optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=load_if_exists
    )

def optuna_tune(
    objective: Callable,
    n_trials: int = DEFAULT_OPTIMIZATION_TRIALS,
    study_name: str = DEFAULT_OPTIMIZATION_STUDY_NAME,
    direction: str = DEFAULT_OPTIMIZATION_DIRECTION,
    storage: str | None = DEFAULT_OPTIMIZATION_STORAGE,
    load_if_exists: bool = DEFAULT_OPTIMIZATION_LOAD_IF_EXISTS,
    show_progress_bar: bool = DEFAULT_OPTIMIZATION_SHOW_PROGRESS
) -> tuple[optuna.Study, dict[str, Any]]:
    """
    Optimize hyperparameters using Optuna.
    
    Args:
        objective (Callable): Objective function for Optuna
        n_trials (int): Number of trials
        study_name (str): Name of the study
        direction (str): Direction of optimization
        storage (str | None): Database URL for storing study results
        load_if_exists (bool): Whether to load existing study
        show_progress_bar (bool): Show progress bar
        
    Returns:
        tuple[optuna.Study, dict[str, Any]]: Study and best parameters
    """
    study = create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=load_if_exists
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar
    )
    
    return study, study.best_params

def pipeline_factory(params: dict[str, Any]) -> Any:
    """Create a text classification pipeline with given parameters."""
    return create_text_classification_pipeline(
        vectorizer_params={
            'max_features': params['max_features'],
            'ngram_range': params['ngram_range'],
            'min_df': params['min_df'],
            'max_df': params['max_df']
        },
        classifier_params={
            'C': params['C'],
            'max_iter': params['max_iter'],
            'class_weight': params['class_weight'],
            'random_state': 42
        },
        n_jobs=-1
    )

def param_space(trial: optuna.Trial) -> dict[str, Any]:
    """Define parameter space for Optuna optimization."""
    # TF-IDF parameters
    max_features = trial.suggest_int('max_features', 1000, 15000, step=1000)
    ngram_range = trial.suggest_categorical('ngram_range', [(1, 1), (1, 2), (1, 3)])
    min_df = trial.suggest_int('min_df', 1, 10)
    max_df = trial.suggest_float('max_df', 0.7, 1.0)
    
    # Logistic Regression parameters
    C = trial.suggest_float('C', 1e-3, 100.0, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', 'none'])
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga', 'lbfgs', 'newton-cg', 'sag'])
    max_iter = trial.suggest_int('max_iter', 100, 2000, step=100)
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
    l1_ratio = None
    
    # l1_ratio only for penalty='elasticnet' and solver='saga'
    if penalty == 'elasticnet' and solver == 'saga':
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
    
    params = {
        'max_features': max_features,
        'ngram_range': ngram_range,
        'min_df': min_df,
        'max_df': max_df,
        'C': C,
        'penalty': penalty,
        'solver': solver,
        'max_iter': max_iter,
        'class_weight': class_weight
    }
    if l1_ratio is not None:
        params['l1_ratio'] = l1_ratio
    return params

def find_optimal_text_classifier_params(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = DEFAULT_OPTIMIZATION_TRIALS,
    cv: int = DEFAULT_CV_FOLDS,
    scoring: str = DEFAULT_SCORING,
    n_jobs: int = DEFAULT_N_JOBS,
    show_progress_bar: bool = DEFAULT_OPTIMIZATION_SHOW_PROGRESS
) -> tuple[optuna.Study, dict[str, Any]]:
    """Find optimal hyperparameters for text classifier using Optuna."""
    
    def objective(trial: optuna.Trial) -> float:
        # Get parameters
        params = param_space(trial)
        
        # Create pipeline
        pipeline = pipeline_factory(params)

        cv_split = StratifiedKFold(
            n_splits=cv,
            shuffle=CV_STRATEGIES['shuffle'],
            random_state=DEFAULT_RANDOM_STATE
        )
        
        # Evaluate pipeline
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv_split,
            scoring=scoring,
            n_jobs=n_jobs
        )
        
        return np.mean(scores)
    
    # Create and optimize study
    study, best_params = optuna_tune(
        objective=objective,
        n_trials=n_trials,
        study_name='text_classifier_optimization',
        direction=DEFAULT_OPTIMIZATION_DIRECTION,
        storage=DEFAULT_OPTIMIZATION_STORAGE,
        load_if_exists=DEFAULT_OPTIMIZATION_LOAD_IF_EXISTS,
        show_progress_bar=show_progress_bar
    )
    
    # Print results
    print("\nOptimal hyperparameters:")
    print("\nTF-IDF parameters:")
    for param in ['max_features', 'ngram_range', 'min_df', 'max_df']:
        if param in best_params:
            print(f"{param}: {best_params[param]}")
    
    print("\nLogistic Regression parameters:")
    for param in ['C', 'penalty', 'solver', 'max_iter', 'class_weight', 'l1_ratio']:
        if param in best_params:
            print(f"{param}: {best_params[param]}")
    
    print(f"\nBest {scoring} score: {study.best_value:.4f}")
    
    return study, best_params 