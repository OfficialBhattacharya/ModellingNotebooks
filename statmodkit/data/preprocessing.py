import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import Levenshtein
from typing import List, Tuple, Dict, Union
from collections import defaultdict

class DataPreprocessor:
    """A class for handling data preprocessing and feature engineering."""
    
    def __init__(self, high_unique_threshold: float = 10.0):
        """
        Initialize the DataPreprocessor.
        
        Args:
            high_unique_threshold: Percentage threshold for considering a column as having high unique values
        """
        self.high_unique_threshold = high_unique_threshold
        self.mappings = {}
        self.removed_columns = []
        
    def _intersection_of_lists(self, list1: List, list2: List) -> List:
        """Return the intersection of two lists."""
        return list(set(list1) & set(list2))
    
    def _difference_of_lists(self, list1: List, list2: List) -> List:
        """Return the difference between two lists."""
        return [item for item in list1 if item not in list2]
    
    def get_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Get numeric and non-numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        return numeric_cols, non_numeric_cols
    
    def remove_single_unique_or_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns with single unique value or all NaNs."""
        for column in df.columns:
            if df[column].nunique() <= 1 or df[column].isna().all():
                self.removed_columns.append(column)
                df = df.drop(columns=[column])
        return df
    
    def fill_missing_numeric(self, df: pd.DataFrame, missing_cols: List[str], 
                           numeric_cols: List[str]) -> pd.DataFrame:
        """Fill missing numeric values with median."""
        for col in self._intersection_of_lists(missing_cols, numeric_cols):
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
        return df
    
    def get_high_unique_columns(self, df: pd.DataFrame, categoric_cols: List[str]) -> List[str]:
        """Identify columns with high number of unique values."""
        total_rows = len(df)
        threshold = total_rows * 0.01 * self.high_unique_threshold
        return [col for col in categoric_cols if df[col].nunique() > threshold]
    
    def convert_to_integer(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Convert categorical column to integer encoding."""
        df[col_name] = df[col_name].astype('object')
        df[col_name], unique_values = pd.factorize(df[col_name])
        self.mappings[col_name] = {value: i for i, value in enumerate(unique_values)}
        return df
    
    def predict_missing_values(self, df: pd.DataFrame, col_name: str, 
                             usable_cols: List[str]) -> pd.DataFrame:
        """Predict missing values using logistic regression."""
        non_missing_idx = df[col_name] != -1
        missing_idx = df[col_name] == -1
        
        if missing_idx.sum() > 0:
            X_train = df.loc[non_missing_idx, usable_cols]
            y_train = df.loc[non_missing_idx, col_name]
            X_test = df.loc[missing_idx, usable_cols]
            
            model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
            model.fit(X_train, y_train)
            predicted = model.predict(X_test)
            df.loc[missing_idx, col_name] = predicted
            
        return df
    
    def _get_bigrams(self, string: str) -> List[str]:
        """Generate bigrams from a string."""
        return [string[i:i+2] for i in range(len(string)-1)]
    
    def _sorensen_dice(self, a: str, b: str) -> float:
        """Calculate Sørensen-Dice coefficient."""
        a_bigrams = set(self._get_bigrams(a))
        b_bigrams = set(self._get_bigrams(b))
        overlap = len(a_bigrams & b_bigrams)
        total = len(a_bigrams) + len(b_bigrams)
        return 1.0 if a == b else 0.0 if total == 0 else 2 * overlap / total
    
    def _calculate_mean_distance(self, input_string: str, string_list: List[str]) -> Tuple[float, float]:
        """Calculate mean Levenshtein and Sørensen-Dice distances."""
        sum_lev = sum(Levenshtein.distance(input_string, string) for string in string_list)
        sum_sd = sum(self._sorensen_dice(input_string, string) for string in string_list)
        return float(sum_lev/len(string_list)), float(sum_sd/len(string_list))
    
    def create_distance_metrics(self, df: pd.DataFrame, col_name: str, 
                              target: str, orig_data: pd.DataFrame) -> pd.DataFrame:
        """Create distance-based features for text columns."""
        df[col_name] = df[col_name].astype('str')
        true_list = list(orig_data[orig_data[target]==1][col_name].unique())
        false_list = list(orig_data[orig_data[target]==0][col_name].unique())
        
        # Calculate distances for true class
        df[[f"{col_name}_true_lev", f"{col_name}_true_reg"]] = df[col_name].apply(
            lambda x: pd.Series(self._calculate_mean_distance(x, true_list))
        )
        
        # Calculate distances for false class
        df[[f"{col_name}_false_lev", f"{col_name}_false_reg"]] = df[col_name].apply(
            lambda x: pd.Series(self._calculate_mean_distance(x, false_list))
        )
        
        return df
    
    def preprocess(self, df: pd.DataFrame, target: str, 
                  orig_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Main preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target: Target column name
            orig_data: Original data for reference (needed for test data)
            
        Returns:
            Preprocessed DataFrame
        """
        if orig_data is None:
            orig_data = df.copy()
            
        # Initial cleaning
        df = self.remove_single_unique_or_nans(df)
        numeric_cols, non_numeric_cols = self.get_column_types(df)
        missing_cols = [col for col in df.columns if df[col].isna().any()]
        
        # Fill missing numeric values
        df = self.fill_missing_numeric(df, missing_cols, numeric_cols)
        missing_cols = [col for col in df.columns if df[col].isna().any()]
        
        # Handle categorical columns
        high_uniques = self.get_high_unique_columns(df, non_numeric_cols)
        usable_cols = numeric_cols.copy()
        
        # Process different types of categorical columns
        for col in self._difference_of_lists(non_numeric_cols, missing_cols + high_uniques):
            df = self.convert_to_integer(df, col)
            usable_cols.append(col)
            
        for col in self._difference_of_lists(missing_cols, high_uniques):
            df = self.convert_to_integer(df, col)
            df = self.predict_missing_values(df, col, usable_cols)
            usable_cols.append(col)
            
        for col in high_uniques:
            df = self.create_distance_metrics(df, col, target, orig_data)
            df = df.drop(columns=[col])
            
        return df 