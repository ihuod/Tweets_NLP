"""
Module for model evaluation using cross-validation and hyperparameter tuning.
"""

import numpy as np
from typing import Any, Callable
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator
from .hyperparameter_tuning import optimize_hyperparameters, optuna_tune
from src.utils.constants import (   DEFAULT_CV_FOLDS,
                                    DEFAULT_INNER_CV_FOLDS,
                                    DEFAULT_OUTER_CV_FOLDS,
                                    DEFAULT_N_TRIALS,
                                    DEFAULT_SCORING,
                                    DEFAULT_N_JOBS,
                                    CV_STRATEGIES,
                                    DEFAULT_RANDOM_STATE
                                )

def cross_validate_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = DEFAULT_CV_FOLDS,
    scoring: str = DEFAULT_SCORING,
    n_jobs: int = DEFAULT_N_JOBS,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratified: bool = CV_STRATEGIES['stratified']
) -> np.ndarray:
    """
    Perform cross-validation.
    
    Args:
        model (BaseEstimator): Model to evaluate
        X (np.ndarray): Features
        y (np.ndarray): Target values
        cv (int): Number of folds
        scoring (str): Scoring metric
        n_jobs (int): Number of jobs for parallel processing
        random_state (int): Random state for reproducibility
        stratified (bool): Whether to use stratified cross-validation
        
    Returns:
        np.ndarray: Cross-validation scores
    """
    # Choose cross-validation strategy
    if stratified:
        cv_split = StratifiedKFold(
            n_splits=cv,
            shuffle=CV_STRATEGIES['shuffle'],
            random_state=random_state
        )
    else:
        cv_split = KFold(
            n_splits=cv,
            shuffle=CV_STRATEGIES['shuffle'],
            random_state=random_state
        )
    
    # Calculate cross-validation scores
    return cross_val_score(
        model,
        X,
        y,
        cv=cv_split,
        scoring=scoring,
        n_jobs=n_jobs
    )

def nested_cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    pipeline_factory: Callable[[dict[str, Any]], Any],
    param_space: Callable[[Any], dict[str, Any]],
    n_trials: int = DEFAULT_N_TRIALS,
    inner_cv: int = DEFAULT_INNER_CV_FOLDS,
    outer_cv: int = DEFAULT_OUTER_CV_FOLDS,
    scoring: str = DEFAULT_SCORING,
    n_jobs: int = DEFAULT_N_JOBS,
    random_state: int = DEFAULT_RANDOM_STATE
) -> np.ndarray:
    """
    Nested cross-validation with hyperparameter tuning.
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Target values
        pipeline_factory (Callable): Function that creates pipeline from parameters
        param_space (Callable): Function that defines parameter space
        n_trials (int): Number of trials
        inner_cv (int): Folds for inner CV
        outer_cv (int): Folds for outer CV
        scoring (str): Metric
        n_jobs (int): Number of jobs for parallel processing
        random_state (int): Random state for reproducibility
        
    Returns:
        np.ndarray: Scores on outer folds
    """
    outer_scores = []
    outer_cv_split = StratifiedKFold(
        n_splits=outer_cv,
        shuffle=CV_STRATEGIES['shuffle'],
        random_state=random_state
    )

    for fold, (train_idx, test_idx) in enumerate(outer_cv_split.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Optimize hyperparameters for this fold
        best_params, best_score = optimize_hyperparameters(
            X=X_train,
            y=y_train,
            pipeline_factory=pipeline_factory,
            param_space=param_space,
            n_trials=n_trials,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            study_name=f'fold_optimization_{fold}'
        )
        
        # Train and evaluate pipeline with best parameters
        best_pipeline = pipeline_factory(best_params)
        best_pipeline.fit(X_train, y_train)
        score = best_pipeline.score(X_test, y_test)
    
        print(f"Fold {fold+1} {scoring.upper()}: {score:.4f}")
        print(f"Best params: {best_params}")
        outer_scores.append(score)

    return np.array(outer_scores)