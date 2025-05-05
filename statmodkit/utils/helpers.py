import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
import joblib
import os
from pathlib import Path
import logging
import warnings

def setup_logging(log_file: Optional[str] = None, 
                 level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    
def suppress_warnings() -> None:
    """Suppress common warnings."""
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check data quality metrics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary of data quality metrics
    """
    quality_metrics = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
        'column_types': df.dtypes.value_counts().to_dict()
    }
    
    return quality_metrics

def validate_model(model: BaseEstimator, 
                  X: pd.DataFrame, 
                  y: pd.Series,
                  cv: int = 5) -> Dict[str, float]:
    """
    Validate model performance using cross-validation.
    
    Args:
        model: Scikit-learn compatible model
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary of validation metrics
    """
    scores = cross_val_score(model, X, y, cv=cv)
    
    return {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'min_score': scores.min(),
        'max_score': scores.max()
    }

def save_object(obj: Any, path: str) -> None:
    """
    Save an object to disk.
    
    Args:
        obj: Object to save
        path: Path to save object
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    
def load_object(path: str) -> Any:
    """
    Load an object from disk.
    
    Args:
        path: Path to load object from
        
    Returns:
        Loaded object
    """
    return joblib.load(path)

def create_directory(path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)
    
def get_file_extension(path: str) -> str:
    """
    Get file extension.
    
    Args:
        path: File path
        
    Returns:
        File extension
    """
    return Path(path).suffix.lower()

def is_numeric_column(series: pd.Series) -> bool:
    """
    Check if a column is numeric.
    
    Args:
        series: Pandas Series
        
    Returns:
        True if column is numeric
    """
    return pd.api.types.is_numeric_dtype(series)

def is_categorical_column(series: pd.Series) -> bool:
    """
    Check if a column is categorical.
    
    Args:
        series: Pandas Series
        
    Returns:
        True if column is categorical
    """
    return pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series)

def get_unique_values(series: pd.Series) -> List[Any]:
    """
    Get unique values in a column.
    
    Args:
        series: Pandas Series
        
    Returns:
        List of unique values
    """
    return series.unique().tolist()

def get_missing_values(series: pd.Series) -> int:
    """
    Get number of missing values in a column.
    
    Args:
        series: Pandas Series
        
    Returns:
        Number of missing values
    """
    return series.isnull().sum()

def get_column_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Get basic statistics for a column.
    
    Args:
        series: Pandas Series
        
    Returns:
        Dictionary of column statistics
    """
    stats = {
        'n_unique': series.nunique(),
        'n_missing': series.isnull().sum(),
        'missing_percentage': (series.isnull().sum() / len(series)) * 100
    }
    
    if is_numeric_column(series):
        stats.update({
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            '25%': series.quantile(0.25),
            '50%': series.quantile(0.5),
            '75%': series.quantile(0.75),
            'max': series.max(),
            'skew': series.skew(),
            'kurtosis': series.kurtosis()
        })
    else:
        stats.update({
            'mode': series.mode().iloc[0],
            'mode_frequency': series.value_counts().iloc[0]
        })
        
    return stats

def format_number(number: float, decimals: int = 2) -> str:
    """
    Format a number with specified decimal places.
    
    Args:
        number: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    return f"{number:.{decimals}f}"

def format_percentage(number: float, decimals: int = 1) -> str:
    """
    Format a number as a percentage.
    
    Args:
        number: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{number:.{decimals}f}%"

def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get memory usage information for a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary of memory usage metrics
    """
    memory_usage = df.memory_usage(deep=True)
    
    return {
        'total_memory': memory_usage.sum(),
        'memory_per_column': memory_usage.to_dict(),
        'memory_per_row': memory_usage.sum() / len(df)
    }

def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Optimized DataFrame
    """
    df_optimized = df.copy()
    
    # Optimize numeric columns
    for col in df_optimized.select_dtypes(include=['int']).columns:
        c_min = df_optimized[col].min()
        c_max = df_optimized[col].max()
        
        if c_min >= 0:
            if c_max < 255:
                df_optimized[col] = df_optimized[col].astype(np.uint8)
            elif c_max < 65535:
                df_optimized[col] = df_optimized[col].astype(np.uint16)
            elif c_max < 4294967295:
                df_optimized[col] = df_optimized[col].astype(np.uint32)
            else:
                df_optimized[col] = df_optimized[col].astype(np.uint64)
        else:
            if c_min > -128 and c_max < 127:
                df_optimized[col] = df_optimized[col].astype(np.int8)
            elif c_min > -32768 and c_max < 32767:
                df_optimized[col] = df_optimized[col].astype(np.int16)
            elif c_min > -2147483648 and c_max < 2147483647:
                df_optimized[col] = df_optimized[col].astype(np.int32)
            else:
                df_optimized[col] = df_optimized[col].astype(np.int64)
                
    # Optimize float columns
    for col in df_optimized.select_dtypes(include=['float']).columns:
        df_optimized[col] = df_optimized[col].astype(np.float32)
        
    # Convert object columns to category if they have low cardinality
    for col in df_optimized.select_dtypes(include=['object']).columns:
        if df_optimized[col].nunique() / len(df_optimized) < 0.5:
            df_optimized[col] = df_optimized[col].astype('category')
            
    return df_optimized 