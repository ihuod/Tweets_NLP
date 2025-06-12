"""
Module for feature selection and importance analysis using SHAP.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from typing import Any
from src.utils.constants import (LOGISTIC_REGRESSION_PARAMS, 
                                 DEFAULT_RANDOM_STATE, 
                                 TOP_FEATURES_COUNT 
                                )

class FeatureImportanceAnalyzer:
    """
    Class for analyzing feature importance using SHAP values.
    Supports various model types including linear models and tree-based models.
    """
    
    def __init__(
        self,
        model: BaseEstimator | None = None,
        model_type: str = 'logistic_regression',
        model_params: dict[str, Any] | None = None,
        random_state: int = DEFAULT_RANDOM_STATE
    ):
        """
        Initialize the feature importance analyzer.
        
        Args:
            model (BaseEstimator | None): Pre-initialized model instance
            model_type (str): Type of model to use ('logistic_regression', 'random_forest', etc.)
            model_params (Dict[str, Any] | None): Parameters for model initialization
            random_state (int): Random state for reproducibility
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.random_state = random_state
        
        if model is not None:
            self.model = model
        else:
            self.model = self._initialize_model()
            
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
    def _initialize_model(self) -> BaseEstimator:
        """
        Initialize model based on model_type and parameters.
        
        Returns:
            BaseEstimator: Initialized model
        """
        if self.model_type == 'logistic_regression':
            params = LOGISTIC_REGRESSION_PARAMS.copy()
            params.update(self.model_params)
            params['random_state'] = self.random_state
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureImportanceAnalyzer':
        """
        Fit the model and calculate SHAP values.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target values
            
        Returns:
            FeatureImportanceAnalyzer: Self
        """
        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()
        
        # Initialize appropriate SHAP explainer based on model type
        if self.model_type == 'logistic_regression':
            self.explainer = shap.LinearExplainer(self.model, X)
        else:
            # For tree-based models
            self.explainer = shap.TreeExplainer(self.model)
            
        self.shap_values = self.explainer.shap_values(X)
        
        return self
        
    def plot_summary(self, max_display: int = TOP_FEATURES_COUNT) -> None:
        """
        Plot SHAP summary plot.
        
        Args:
            max_display (int): Maximum number of features to display
        """
        if self.shap_values is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before plotting.")
            
        shap.summary_plot(
            self.shap_values,
            self.feature_names,
            max_display=max_display
        )
        
    def plot_dependence(
        self,
        feature: str,
        interaction_index: str | None = None
    ) -> None:
        """
        Plot SHAP dependence plot for a specific feature.
        
        Args:
            feature (str): Feature name to plot
            interaction_index (str | None): Feature to use for coloring
        """
        if self.shap_values is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before plotting.")
            
        shap.dependence_plot(
            feature,
            self.shap_values,
            self.feature_names,
            interaction_index=interaction_index
        )
        
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if self.shap_values is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before getting feature importance.")
            
        # Calculate mean absolute SHAP values
        importance = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame with feature names and importance scores
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
        
    def plot_feature_importance(self, top_n: int = TOP_FEATURES_COUNT) -> None:
        """
        Plot feature importance scores using Plotly Express.
        
        Args:
            top_n (int): Number of top features to display
        """
        importance_df = self.get_feature_importance()
        fig = px.bar(
            importance_df.head(top_n),
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Feature Importance',
            labels={'importance': 'Mean |SHAP value|', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        fig.show()