"""
Module for feature selection and importance analysis using SHAP.
"""

import numpy as np
import pandas as pd
import shap
from typing import List, Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureImportanceAnalyzer:
    """
    Class for analyzing feature importance using SHAP values.
    """
    
    def __init__(
        self,
        model: Optional[LogisticRegression] = None,
        random_state: int = 42,
        max_iter: int = 1000,
        class_weight: str = 'balanced'
    ):
        """
        Initialize the feature importance analyzer.
        
        Args:
            model (Optional[LogisticRegression]): Logistic regression model
            random_state (int): Random state for reproducibility
            max_iter (int): Maximum number of iterations for logistic regression
            class_weight (str): Class weight strategy
        """
        self.model = model or LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight=class_weight
        )
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
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
        
        # Initialize SHAP explainer
        self.explainer = shap.LinearExplainer(self.model, X)
        self.shap_values = self.explainer.shap_values(X)
        
        return self
        
    def plot_summary(self, max_display: int = 20) -> None:
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
        interaction_index: Optional[str] = None
    ) -> None:
        """
        Plot SHAP dependence plot for a specific feature.
        
        Args:
            feature (str): Feature name to plot
            interaction_index (Optional[str]): Feature to use for coloring
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
        
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """
        Plot feature importance scores.
        
        Args:
            top_n (int): Number of top features to display
        """
        importance_df = self.get_feature_importance()
        
        # Plot top N features
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=importance_df.head(top_n),
            x='importance',
            y='feature'
        )
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Mean |SHAP value|')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()