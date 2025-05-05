import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.base import BaseEstimator
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm
import optuna
from sklearn.model_selection import cross_val_score
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib

class FeatureSelector:
    """A class for feature selection and model optimization."""
    
    def __init__(self, n_features_to_select: int = 7, n_jobs: int = 1):
        """
        Initialize the FeatureSelector.
        
        Args:
            n_features_to_select: Number of features to select
            n_jobs: Number of parallel jobs to run
        """
        self.n_features_to_select = n_features_to_select
        self.n_jobs = n_jobs
        self.selected_features = {}
        self.model_summaries = {}
        
    def _valid_subset(self, subset: Tuple[str, ...]) -> bool:
        """Check if a subset of features is valid."""
        original_cols = [col.split('_')[0] for col in subset]
        return all(original_cols.count(col) <= 1 for col in original_cols)
    
    def _calculate_subset_count(self, df: pd.DataFrame, subset: Tuple[str, ...]) -> Tuple[Tuple[str, ...], float]:
        """Calculate the count of a feature subset."""
        subset_df = df[list(subset)]
        count = (subset_df.sum(axis=1) == len(subset)).sum()
        return subset, count / len(df)
    
    def subset_counts_one_hot(self, df: pd.DataFrame, target: str, id_col: str) -> Dict[Tuple[str, ...], float]:
        """Calculate counts for all valid feature subsets."""
        df_target_1 = df[df[target] == 1]
        one_hot_columns = df_target_1.drop(columns=[target, id_col]).columns
        
        subset_counts = {}
        all_combinations = []
        
        # Generate all valid combinations
        for r in range(1, len(one_hot_columns) + 1):
            for subset in combinations(one_hot_columns, r):
                if self._valid_subset(subset):
                    all_combinations.append(subset)
        
        # Calculate counts in parallel
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(self._calculate_subset_count, df_target_1, subset) 
                      for subset in all_combinations]
            for future in as_completed(futures):
                subset, count = future.result()
                subset_counts[subset] = count
                
        return subset_counts
    
    def filter_and_sort_subsets(self, subset_counts: Dict[Tuple[str, ...], float], 
                              threshold: float) -> List[Tuple[Tuple[str, ...], float]]:
        """Filter and sort feature subsets based on threshold."""
        filtered_subsets = {subset: count for subset, count in subset_counts.items() 
                          if count > threshold}
        return sorted(filtered_subsets.items(), key=lambda item: item[1], reverse=True)
    
    def create_fractional_features(self, df: pd.DataFrame, id_col: str, target_col: str,
                                 filtered_subsets: List[Tuple[Tuple[str, ...], float]]) -> pd.DataFrame:
        """Create fractional features from selected subsets."""
        df = df.copy()
        
        for subset, _ in filtered_subsets:
            new_col_name = "_".join(subset) + "_fraction"
            df[new_col_name] = df[list(subset)].mean(axis=1)
            
        new_fraction_columns = [("_".join(subset) + "_fraction") 
                              for subset, _ in filtered_subsets]
        selected_columns = [id_col, target_col] + new_fraction_columns
        
        return df[selected_columns]
    
    def backward_feature_selection(self, df: pd.DataFrame, target: str,
                                 models_dict: Dict[str, BaseEstimator]) -> Dict[str, Any]:
        """Perform backward feature selection using multiple models."""
        X = df.drop(columns=[target])
        y = df[target]
        
        results = {}
        
        for model_name, model in models_dict.items():
            try:
                # Initialize SFS
                sfs = SFS(model,
                         k_features=self.n_features_to_select,
                         forward=False,
                         floating=False,
                         scoring='accuracy',
                         cv=2,
                         n_jobs=self.n_jobs)
                
                # Fit SFS
                sfs = sfs.fit(X, y)
                
                # Get selected features
                selected_features = list(sfs.k_feature_names_)
                self.selected_features[model_name] = selected_features
                
                # Fit statsmodels for detailed analysis
                X_selected = sm.add_constant(X[selected_features])
                sm_model = sm.OLS(y, X_selected).fit()
                
                # Store results
                self.model_summaries[model_name] = sm_model.summary2().tables[1]
                results[model_name] = {
                    'selected_features': selected_features,
                    'model_summary': sm_model.summary2().tables[1]
                }
                
            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
                
        return results
    
    def optimize_model(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                      n_trials: int = 20) -> Tuple[Dict[str, Any], float]:
        """Optimize model hyperparameters using Optuna."""
        def objective(trial):
            if isinstance(model, RandomForestClassifier):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 1, 10),
                    'max_depth': trial.suggest_int('max_depth', 1, 4),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 4),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
                }
            elif isinstance(model, GradientBoostingClassifier):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 1, 10),
                    'max_depth': trial.suggest_int('max_depth', 1, 4)
                }
            elif isinstance(model, XGBClassifier):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 1, 10),
                    'max_depth': trial.suggest_int('max_depth', 1, 4)
                }
            elif isinstance(model, LGBMClassifier):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 1, 10),
                    'max_depth': trial.suggest_int('max_depth', 1, 4)
                }
            elif isinstance(model, KNeighborsClassifier):
                params = {
                    'n_neighbors': trial.suggest_int('n_neighbors', 1, 10),
                    'leaf_size': trial.suggest_int('leaf_size', 10, 30),
                    'p': trial.suggest_int('p', 1, 2)
                }
            elif isinstance(model, SVC):
                params = {
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
                }
            elif isinstance(model, DecisionTreeClassifier):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 1, 3),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 4),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
                }
            else:
                params = {}
                
            model.set_params(**params)
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            return scores.mean()
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, n_jobs=self.n_jobs)
        
        return study.best_params, study.best_value
    
    def save_model(self, model: BaseEstimator, path: str, model_name: str) -> None:
        """Save a trained model to disk."""
        joblib.dump(model, f"{path}/{model_name}.joblib")
        
    def load_model(self, path: str, model_name: str) -> BaseEstimator:
        """Load a trained model from disk."""
        return joblib.load(f"{path}/{model_name}.joblib") 