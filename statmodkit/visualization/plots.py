import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union
import statsmodels.api as sm
from scipy import stats

class StatisticalVisualizer:
    """A class for creating statistical visualizations and summary tables."""
    
    def __init__(self, style: str = 'seaborn'):
        """
        Initialize the StatisticalVisualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.figsize = (12, 8)
        
    def feature_distribution_plot(self, df: pd.DataFrame, feature: str, 
                                target: Optional[str] = None) -> None:
        """
        Create distribution plots for a feature.
        
        Args:
            df: Input DataFrame
            feature: Feature to plot
            target: Optional target variable for conditional plots
        """
        plt.figure(figsize=self.figsize)
        
        if target is None:
            # Simple distribution plot
            if df[feature].dtype in ['int64', 'float64']:
                sns.histplot(data=df, x=feature, kde=True)
            else:
                sns.countplot(data=df, x=feature)
        else:
            # Conditional distribution plot
            if df[feature].dtype in ['int64', 'float64']:
                sns.boxplot(data=df, x=target, y=feature)
            else:
                sns.countplot(data=df, x=feature, hue=target)
                
        plt.title(f'Distribution of {feature}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def correlation_heatmap(self, df: pd.DataFrame, 
                          target: Optional[str] = None) -> None:
        """
        Create a correlation heatmap.
        
        Args:
            df: Input DataFrame
            target: Optional target variable to include
        """
        plt.figure(figsize=self.figsize)
        
        if target is not None:
            # Calculate correlations with target
            corr = df.corr()[target].sort_values(ascending=False)
            sns.heatmap(corr.to_frame(), annot=True, cmap='coolwarm', 
                       center=0, fmt='.2f')
        else:
            # Full correlation matrix
            corr = df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', 
                       center=0, fmt='.2f')
            
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        
    def feature_importance_plot(self, importance_scores: Dict[str, float]) -> None:
        """
        Create a feature importance plot.
        
        Args:
            importance_scores: Dictionary of feature names and their importance scores
        """
        plt.figure(figsize=self.figsize)
        
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_features)
        
        # Create bar plot
        sns.barplot(x=list(scores), y=list(features))
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
    def residual_plot(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Create a residual plot.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
        """
        plt.figure(figsize=self.figsize)
        
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.tight_layout()
        plt.show()
        
    def qq_plot(self, data: np.ndarray) -> None:
        """
        Create a Q-Q plot.
        
        Args:
            data: Data to plot
        """
        plt.figure(figsize=self.figsize)
        stats.probplot(data, dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        plt.tight_layout()
        plt.show()
        
    def create_summary_table(self, df: pd.DataFrame, 
                           target: Optional[str] = None) -> pd.DataFrame:
        """
        Create a comprehensive summary table.
        
        Args:
            df: Input DataFrame
            target: Optional target variable for conditional statistics
            
        Returns:
            Summary DataFrame
        """
        summary = pd.DataFrame()
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Numerical statistics
                stats = {
                    'count': df[col].count(),
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    '25%': df[col].quantile(0.25),
                    '50%': df[col].quantile(0.5),
                    '75%': df[col].quantile(0.75),
                    'max': df[col].max(),
                    'skew': df[col].skew(),
                    'kurtosis': df[col].kurtosis()
                }
                
                if target is not None:
                    # Add conditional statistics
                    for val in df[target].unique():
                        subset = df[df[target] == val][col]
                        stats.update({
                            f'mean_{val}': subset.mean(),
                            f'std_{val}': subset.std()
                        })
                        
            else:
                # Categorical statistics
                stats = {
                    'count': df[col].count(),
                    'unique': df[col].nunique(),
                    'top': df[col].mode().iloc[0],
                    'freq': df[col].value_counts().iloc[0]
                }
                
                if target is not None:
                    # Add conditional statistics
                    for val in df[target].unique():
                        subset = df[df[target] == val][col]
                        stats.update({
                            f'top_{val}': subset.mode().iloc[0],
                            f'freq_{val}': subset.value_counts().iloc[0]
                        })
                        
            summary[col] = pd.Series(stats)
            
        return summary.T
        
    def plot_model_performance(self, metrics: Dict[str, float]) -> None:
        """
        Create a bar plot of model performance metrics.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        plt.figure(figsize=self.figsize)
        
        metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
        sns.barplot(data=metrics_df, x='Metric', y='Value')
        plt.title('Model Performance Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def plot_learning_curve(self, train_sizes: np.ndarray, 
                          train_scores: np.ndarray, 
                          val_scores: np.ndarray) -> None:
        """
        Create a learning curve plot.
        
        Args:
            train_sizes: Array of training set sizes
            train_scores: Array of training scores
            val_scores: Array of validation scores
        """
        plt.figure(figsize=self.figsize)
        
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Score')
        plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation Score')
        
        plt.fill_between(train_sizes, 
                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1),
                        alpha=0.1)
        plt.fill_between(train_sizes,
                        val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1),
                        alpha=0.1)
        
        plt.title('Learning Curve')
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.legend()
        plt.tight_layout()
        plt.show() 