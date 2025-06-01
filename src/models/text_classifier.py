"""
Text classifier model using TF-IDF and Logistic Regression.
"""

import numpy as np
from typing import Any, Dict, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .base_model import BaseModel

class TextClassifier(BaseModel):
    """
    Text classifier using TF-IDF and Logistic Regression.
    """
    
    def __init__(
        self,
        vectorizer_params: Optional[Dict[str, Any]] = None,
        classifier_params: Optional[Dict[str, Any]] = None,
        n_jobs: int = -1
    ):
        """
        Initialize the text classifier.
        
        Args:
            vectorizer_params (Optional[Dict[str, Any]]): Parameters for TfidfVectorizer
            classifier_params (Optional[Dict[str, Any]]): Parameters for LogisticRegression
            n_jobs (int): Number of jobs to run in parallel
        """
        super().__init__(n_jobs=n_jobs)
        self._vectorizer_params = vectorizer_params or {}
        self._classifier_params = classifier_params or {}
        self._model = None
        
    @property
    def model(self) -> Pipeline:
        """
        Get the model pipeline.
        
        Returns:
            Pipeline: Model pipeline
        """
        if self._model is None:
            self._model = self._create_pipeline()
        return self._model
        
    def _create_pipeline(self) -> Pipeline:
        """
        Create the model pipeline.
        
        Returns:
            Pipeline: Model pipeline
        """
        return Pipeline([
            ('vectorizer', TfidfVectorizer(**self._vectorizer_params)),
            ('classifier', LogisticRegression(n_jobs=self.n_jobs, **self._classifier_params))
        ])
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TextClassifier':
        """
        Fit the model.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Target values
            
        Returns:
            TextClassifier: Self
        """
        self.model.fit(X, y)
        return self
        
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
        
    def get_feature_names(self) -> List[str]:
        """
        Get feature names.
        
        Returns:
            List[str]: Feature names
        """
        return self.model.named_steps['vectorizer'].get_feature_names_out()
        
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            Dict[str, float]: Feature importance
        """
        # Get feature names and coefficients
        feature_names = self.get_feature_names()
        coefficients = self.model.named_steps['classifier'].coef_[0]
        
        # Calculate feature importance
        feature_importance = dict(zip(feature_names, np.abs(coefficients)))
        
        # Sort by importance and get top N
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return dict(sorted_features)
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        
        Args:
            deep (bool): If True, will return the parameters for this estimator and
                        contained subobjects that are estimators.
                        
        Returns:
            Dict[str, Any]: Parameter names mapped to their values
        """
        params = {
            'vectorizer_params': self._vectorizer_params,
            'classifier_params': self._classifier_params,
            'n_jobs': self.n_jobs
        }
        if deep and self._model is not None:
            params['model'] = self._model
        return params
        
    def set_params(self, **params: Any) -> 'TextClassifier':
        """
        Set the parameters of this estimator.
        
        Args:
            **params (Any): Estimator parameters
            
        Returns:
            TextClassifier: Self
        """
        if 'vectorizer_params' in params:
            self._vectorizer_params = params.pop('vectorizer_params')
        if 'classifier_params' in params:
            self._classifier_params = params.pop('classifier_params')
        if 'n_jobs' in params:
            self.n_jobs = params.pop('n_jobs')
            
        # Reset model to force recreation with new parameters
        self._model = None
        
        # Set any remaining parameters
        for param, value in params.items():
            setattr(self, param, value)
            
        return self 