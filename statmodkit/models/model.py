import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

from ..data.preprocessing import DataPreprocessor
from ..models.feature_selection import FeatureSelector
from ..visualization.plots import StatisticalVisualizer

class StatisticalModel:
    """A comprehensive statistical modeling class that integrates preprocessing, 
    feature selection, and visualization capabilities."""
    
    def __init__(self, 
                 model: BaseEstimator,
                 target_column: str,
                 problem_type: str = 'classification',
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize the StatisticalModel.
        
        Args:
            model: Scikit-learn compatible model
            target_column: Name of the target column
            problem_type: Type of problem ('classification' or 'regression')
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.target_column = target_column
        self.problem_type = problem_type
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.feature_selector = FeatureSelector()
        self.visualizer = StatisticalVisualizer()
        
        # Store data and results
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_importance = None
        self.metrics = None
        
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the model to the data.
        
        Args:
            df: Input DataFrame
        """
        # Preprocess data
        df_processed = self.preprocessor.preprocess(df)
        
        # Split data
        X = df_processed.drop(columns=[self.target_column])
        y = df_processed[self.target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Select features
        self.feature_selector.fit(self.X_train, self.y_train)
        self.X_train = self.feature_selector.transform(self.X_train)
        self.X_test = self.feature_selector.transform(self.X_test)
        
        # Fit model
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.X_train.columns,
                self.model.feature_importances_
            ))
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = dict(zip(
                self.X_train.columns,
                np.abs(self.model.coef_)
            ))
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Array of predictions
        """
        # Preprocess and select features
        X_processed = self.preprocessor.preprocess(X)
        X_processed = self.feature_selector.transform(X_processed)
        
        return self.model.predict(X_processed)
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Returns:
            Dictionary of performance metrics
        """
        y_pred = self.model.predict(self.X_test)
        
        if self.problem_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred)
            }
        else:
            metrics = {
                'mse': mean_squared_error(self.y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                'r2': r2_score(self.y_test, y_pred)
            }
            
        self.metrics = metrics
        return metrics
    
    def visualize(self) -> None:
        """Create visualizations of the model and data."""
        # Feature distributions
        for feature in self.X_train.columns:
            self.visualizer.feature_distribution_plot(
                pd.concat([self.X_train, self.y_train], axis=1),
                feature,
                self.target_column
            )
            
        # Correlation heatmap
        self.visualizer.correlation_heatmap(
            pd.concat([self.X_train, self.y_train], axis=1),
            self.target_column
        )
        
        # Feature importance
        if self.feature_importance is not None:
            self.visualizer.feature_importance_plot(self.feature_importance)
            
        # Residual plot
        if self.problem_type == 'regression':
            y_pred = self.model.predict(self.X_test)
            self.visualizer.residual_plot(self.y_test, y_pred)
            
        # Model performance
        if self.metrics is not None:
            self.visualizer.plot_model_performance(self.metrics)
            
    def get_summary(self) -> pd.DataFrame:
        """
        Get a comprehensive summary of the data and model.
        
        Returns:
            Summary DataFrame
        """
        return self.visualizer.create_summary_table(
            pd.concat([self.X_train, self.y_train], axis=1),
            self.target_column
        )
        
    def save(self, path: str) -> None:
        """
        Save the model and its components.
        
        Args:
            path: Directory to save the model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, os.path.join(path, 'model.joblib'))
        
        # Save preprocessor
        joblib.dump(self.preprocessor, os.path.join(path, 'preprocessor.joblib'))
        
        # Save feature selector
        joblib.dump(self.feature_selector, os.path.join(path, 'feature_selector.joblib'))
        
        # Save metadata
        metadata = {
            'target_column': self.target_column,
            'problem_type': self.problem_type,
            'test_size': self.test_size,
            'random_state': self.random_state
        }
        joblib.dump(metadata, os.path.join(path, 'metadata.joblib'))
        
    @classmethod
    def load(cls, path: str) -> 'StatisticalModel':
        """
        Load a saved model and its components.
        
        Args:
            path: Directory containing the saved model
            
        Returns:
            Loaded StatisticalModel instance
        """
        # Load metadata
        metadata = joblib.load(os.path.join(path, 'metadata.joblib'))
        
        # Create instance
        model = cls(
            model=joblib.load(os.path.join(path, 'model.joblib')),
            target_column=metadata['target_column'],
            problem_type=metadata['problem_type'],
            test_size=metadata['test_size'],
            random_state=metadata['random_state']
        )
        
        # Load components
        model.preprocessor = joblib.load(os.path.join(path, 'preprocessor.joblib'))
        model.feature_selector = joblib.load(os.path.join(path, 'feature_selector.joblib'))
        
        return model 