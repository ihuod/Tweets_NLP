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

def find_optimal_text_classifier_params(
    X: np.ndarray,
    y: np.ndarray,
    pipeline_factory: Callable[[dict[str, Any]], Any],
    param_space: Callable[[optuna.Trial], dict[str, Any]],
    n_trials: int = DEFAULT_OPTIMIZATION_TRIALS,
    cv: int = DEFAULT_CV_FOLDS,
    scoring: str = DEFAULT_SCORING,
    n_jobs: int = DEFAULT_N_JOBS,
    show_progress_bar: bool = DEFAULT_OPTIMIZATION_SHOW_PROGRESS
) -> tuple[optuna.Study, dict[str, Any]]:
    """
    Find optimal hyperparameters using Optuna.
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Target
        pipeline_factory (Callable): Function that creates pipeline from parameters
        param_space (Callable): Function that defines parameter space for Optuna
        n_trials (int): Number of trials
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric
        n_jobs (int): Number of jobs for parallel processing
        show_progress_bar (bool): Show progress bar
        
    Returns:
        tuple[optuna.Study, dict[str, Any]]: Study and best parameters
    """
    def objective(trial: optuna.Trial) -> float:
        # Get parameters
        params = param_space(trial)
        
        # Create pipeline
        pipeline = pipeline_factory(params)
        
        # Evaluate pipeline
        cv_split = StratifiedKFold(
            n_splits=cv,
            shuffle=CV_STRATEGIES['shuffle'],
            random_state=DEFAULT_RANDOM_STATE
        )
        
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
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    print(f"\nBest {scoring} score: {study.best_value:.4f}")
    
    return study, best_params 