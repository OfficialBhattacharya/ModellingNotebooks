# StatModKit

A comprehensive statistical modeling toolkit that simplifies the lives of statistical modelers by providing a user-friendly interface for common statistical modeling tasks, enhanced with custom statistical plots and summary tables.

## Features

- **Data Preprocessing**
  - Automatic handling of missing values
  - Categorical variable encoding
  - Text similarity metrics
  - Data quality checks
  - Memory optimization

- **Feature Selection**
  - Backward feature elimination
  - Model-based feature importance
  - Feature subset validation
  - Hyperparameter optimization

- **Model Training & Evaluation**
  - Support for scikit-learn compatible models
  - Cross-validation
  - Performance metrics
  - Model persistence

- **Visualization**
  - Feature distribution plots
  - Correlation heatmaps
  - Feature importance plots
  - Residual plots
  - Q-Q plots
  - Learning curves
  - Model performance metrics

- **Configuration Management**
  - YAML-based configuration
  - Default settings
  - Customizable parameters
  - Configuration persistence

## Installation

```bash
pip install statmodkit
```

## Quick Start

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from statmodkit import StatisticalModel

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize the model
model = StatisticalModel(
    model=RandomForestClassifier(),
    target_column='target',
    problem_type='classification'
)

# Fit the model
model.fit(df)

# Evaluate performance
metrics = model.evaluate()
print(metrics)

# Create visualizations
model.visualize()

# Get data summary
summary = model.get_summary()
print(summary)

# Save the model
model.save('model_directory')
```

## Documentation

### Data Preprocessing

The `DataPreprocessor` class handles data cleaning and transformation:

```python
from statmodkit.data.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
df_processed = preprocessor.preprocess(df)
```

### Feature Selection

The `FeatureSelector` class manages feature selection and model optimization:

```python
from statmodkit.models.feature_selection import FeatureSelector

selector = FeatureSelector()
selector.fit(X, y)
X_selected = selector.transform(X)
```

### Visualization

The `StatisticalVisualizer` class creates statistical plots and summaries:

```python
from statmodkit.visualization.plots import StatisticalVisualizer

visualizer = StatisticalVisualizer()
visualizer.feature_distribution_plot(df, 'feature', 'target')
visualizer.correlation_heatmap(df, 'target')
```

### Configuration

The `ModelConfig` class manages model settings:

```python
from statmodkit.config.config import ModelConfig

config = ModelConfig('config.yaml')
preprocessing_config = config.get_preprocessing_config()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 