"""
Utility Functions for Credit Risk Scoring Model

This module contains helper functions used across the project for:
- Configuration loading
- Logging setup
- File I/O operations
- Common data transformations
"""

import os
import yaml
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def load_config(config_path: str = "config/model_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        raise


def setup_logging(
    log_file: str = "outputs/logs/training.log",
    level: str = "INFO"
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to the log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized")


def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk using joblib.
    
    Args:
        model: Trained model object
        filepath: Path where to save the model
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        logging.info(f"Model saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object
    """
    try:
        model = joblib.load(filepath)
        logging.info(f"Model loaded from {filepath}")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """
    Save a pandas DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        filepath: Path where to save the CSV
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logging.info(f"DataFrame saved to {filepath} ({len(df)} rows)")
    except Exception as e:
        logging.error(f"Error saving DataFrame: {e}")
        raise


def load_dataframe(filepath: str) -> pd.DataFrame:
    """
    Load a pandas DataFrame from CSV.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Loaded DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"DataFrame loaded from {filepath} ({len(df)} rows)")
        return df
    except FileNotFoundError:
        logging.error(f"Data file not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading DataFrame: {e}")
        raise


def create_directories(directories: List[str]) -> None:
    """
    Create multiple directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logging.info(f"Created {len(directories)} directories")


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Set seed for scikit-learn models
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logging.info(f"Random seed set to {seed}")


def get_feature_names(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """
    Get list of feature column names, excluding specified columns.
    
    Args:
        df: Input DataFrame
        exclude_cols: List of columns to exclude (e.g., target, ID columns)
        
    Returns:
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = []
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of missing values in a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing value statistics
    """
    missing_stats = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    
    missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values(
        by='Missing_Percentage', ascending=False
    )
    
    if len(missing_stats) == 0:
        logging.info("No missing values found in the dataset")
    else:
        logging.warning(f"Found missing values in {len(missing_stats)} columns")
    
    return missing_stats


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float = 1.5
) -> Dict[str, int]:
    """
    Detect outliers using the IQR method.
    
    Args:
        df: Input DataFrame
        columns: List of columns to check for outliers
        threshold: IQR multiplier (default: 1.5)
        
    Returns:
        Dictionary with column names and outlier counts
    """
    outlier_counts = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_counts[col] = outliers
    
    total_outliers = sum(outlier_counts.values())
    logging.info(f"Detected {total_outliers} outliers across {len(columns)} columns")
    
    return outlier_counts


def calculate_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for multicollinearity detection.
    
    Args:
        df: Input DataFrame
        features: List of feature columns
        
    Returns:
        DataFrame with VIF values
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [
        variance_inflation_factor(df[features].values, i) 
        for i in range(len(features))
    ]
    
    vif_data = vif_data.sort_values(by="VIF", ascending=False)
    logging.info(f"VIF calculated for {len(features)} features")
    
    return vif_data


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """
    Pretty print model evaluation metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        model_name: Name of the model
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*50}")
    
    for metric_name, value in metrics.items():
        print(f"{metric_name:20s}: {value:.4f}")
    
    print(f"{'='*50}\n")


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path object pointing to project root
    """
    return Path(__file__).parent.parent


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test config loading
    try:
        config = load_config()
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
    
    # Test logging setup
    try:
        setup_logging()
        print("✓ Logging setup successfully")
    except Exception as e:
        print(f"✗ Logging setup failed: {e}")
    
    # Test random seed
    set_random_seed(42)
    print("✓ Random seed set successfully")
    
    print("\nAll utility functions tested!")
