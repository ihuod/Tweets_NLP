from abc import ABC, abstractmethod
import numpy as np
from typing import Any

class BaseModel(ABC):
    """
    Abstract base class for all models.
    """
    
    def __init__(self, n_jobs: int = -1):
        """
        Initialize the base model.
        
        Args:
            n_jobs (int): Number of jobs to run in parallel
        """
        self.n_jobs = n_jobs
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Fit the model.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Target values
            
        Returns:
            BaseModel: Self
        """
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X (np.ndarray): Features to predict
            
        Returns:
            np.ndarray: Predictions
        """
        pass
        
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X (np.ndarray): Features to predict
            
        Returns:
            np.ndarray: Class probabilities
        """
        pass
        
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get parameters for this estimator.
        
        Args:
            deep (bool): If True, will return the parameters for this estimator and
                        contained subobjects that are estimators.
                        
        Returns:
            dict[str, Any]: Parameter names mapped to their values
        """
        return {'n_jobs': self.n_jobs}
        
    def set_params(self, **params: Any) -> 'BaseModel':
        """
        Set the parameters of this estimator.
        
        Args:
            **params (Any): Estimator parameters
            
        Returns:
            BaseModel: Self
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self 