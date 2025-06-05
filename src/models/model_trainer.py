import numpy as np
import joblib
from typing import Any, Callable
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.base import BaseEstimator
from .hyperparameter_tuning import optuna_tune
from src.utils.constants import (
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_N_JOBS,
    DEFAULT_CV_FOLDS,
    DEFAULT_INNER_CV_FOLDS,
    DEFAULT_OUTER_CV_FOLDS,
    DEFAULT_N_TRIALS,
    DEFAULT_SCORING,
    CV_STRATEGIES,
    MODEL_METRICS,
    MODEL_FILE_EXTENSION
)

class ModelTrainer:
    """
    Class for training and evaluating models.
    """
    
    def __init__(
        self,
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int | None = DEFAULT_RANDOM_STATE,
        n_jobs: int = DEFAULT_N_JOBS
    ):
        """
        Initialize the model trainer.
        
        Args:
            test_size (float): Proportion of the dataset to include in the test split
            random_state (Optional[int]): Random state for reproducibility
            n_jobs (int): Number of jobs to run in parallel
        """
        self.test_size = test_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.metrics = {}
        
    def train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target values
            
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Train and test sets
        """
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> 'ModelTrainer':
        """
        Train the model.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Target values
            
        Returns:
            ModelTrainer: Self
        """
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)
        
        # Train model
        self.model = self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return self
    
    def cross_validate(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = DEFAULT_CV_FOLDS,
        scoring: str = DEFAULT_SCORING,
        stratified: bool = CV_STRATEGIES['stratified']
    ) -> np.ndarray:
        """
        Perform cross-validation.
        
        Args:
            model (BaseEstimator): Model to evaluate
            X (np.ndarray): Training features
            y (np.ndarray): Target values
            cv (int): Number of folds
            scoring (str): Scoring metric
            stratified (bool): Whether to use stratified cross-validation
            
        Returns:
            np.ndarray: Cross-validation scores
        """
        # Choose cross-validation strategy
        if stratified:
            cv_split = StratifiedKFold(
                n_splits=cv,
                shuffle=CV_STRATEGIES['shuffle'],
                random_state=self.random_state
            )
        else:
            cv_split = KFold(
                n_splits=cv,
                shuffle=CV_STRATEGIES['shuffle'],
                random_state=self.random_state
            )
        
        # Calculate cross-validation scores
        return cross_val_score(
            model,
            X,
            y,
            cv=cv_split,
            scoring=scoring,
            n_jobs=self.n_jobs
        )
    
    def get_metrics(self) -> dict[str, float]:
        """
        Get model metrics.
        
        Returns:
            dict[str, float]: Model metrics
        """
        return self.metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X (np.ndarray): Features to predict
            
        Returns:
            np.ndarray: Predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X (np.ndarray): Features to predict
            
        Returns:
            np.ndarray: Class probabilities
        """
        return self.model.predict_proba(X)
        
    def save_model(self, path: str) -> None:
        """
        Save model to file.
        
        Args:
            path (str): Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() method first.")
        
        if not path.endswith(MODEL_FILE_EXTENSION):
            path += MODEL_FILE_EXTENSION
        joblib.dump(self.model, path)
        
    def load_model(self, path: str) -> None:
        """
        Load model from file.
        
        Args:
            path (str): Path to load model from
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() method first.")
        
        if not path.endswith(MODEL_FILE_EXTENSION):
            path += MODEL_FILE_EXTENSION
        self.model = joblib.load(path)

    def nested_optuna_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pipeline_factory: Callable[[dict[str, Any]], Any],
        param_space: Callable[[Any], dict[str, Any]],
        n_trials: int = DEFAULT_N_TRIALS,
        inner_cv: int = DEFAULT_INNER_CV_FOLDS,
        outer_cv: int = DEFAULT_OUTER_CV_FOLDS,
        scoring: str = DEFAULT_SCORING,
        random_state: int = DEFAULT_RANDOM_STATE
    ) -> np.ndarray:
        """
        Nested cross-validation with hyperparameter tuning using Optuna.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target values
            pipeline_factory (Callable): Pipeline factory
            param_space (Callable): Function returning parameters for trial
            n_trials (int): Number of Optuna trials
            inner_cv (int): Folds for inner CV
            outer_cv (int): Folds for outer CV
            scoring (str): Metric
            random_state (int): Random state
            
        Returns:
            np.ndarray: Scores on outer folds
        """
        outer_scores = []
        outer_cv_split = StratifiedKFold(
            n_splits=outer_cv,
            shuffle=CV_STRATEGIES['shuffle'],
            random_state=random_state
        )
        inner_cv_split = StratifiedKFold(
            n_splits=inner_cv,
            shuffle=CV_STRATEGIES['shuffle'],
            random_state=random_state
        )

        for fold, (train_idx, test_idx) in enumerate(outer_cv_split.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            def objective(trial):
                params = param_space(trial)
                pipeline = pipeline_factory(params)
                score = cross_val_score(
                    pipeline, X_train, y_train,
                    cv=inner_cv_split,
                    scoring=scoring,
                    n_jobs=self.n_jobs
                ).mean()
                return score

            study, best_params = optuna_tune(objective, n_trials=n_trials)
            pipeline = pipeline_factory(best_params)
            pipeline.fit(X_train, y_train)
            score = pipeline.score(X_test, y_test)
            print(f"Fold {fold+1} {scoring.upper()}: {score:.4f}")
            print(f"Best params: {best_params}")
            outer_scores.append(score)

        return np.array(outer_scores) 