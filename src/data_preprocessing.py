"""
Data Preprocessing Module for Credit Risk Scoring Model

This module handles:
- Data loading and validation
- Missing value imputation
- Outlier detection and treatment
- Data type conversions
- Initial data cleaning
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Import utility functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    load_config, 
    setup_logging, 
    save_dataframe, 
    load_dataframe,
    set_random_seed,
    check_missing_values,
    detect_outliers_iqr
)


class DataPreprocessor:
    """
    Handles all data preprocessing steps for the credit risk model.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.random_state = self.config['random_state']
        set_random_seed(self.random_state)
        
        # Setup logging
        setup_logging(
            log_file=self.config['logging']['file'],
            level=self.config['logging']['level']
        )
        
        logging.info("DataPreprocessor initialized")
        
    
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load raw data from CSV file.
        
        Args:
            filepath: Path to the data file (uses config if None)
            
        Returns:
            Loaded DataFrame
        """
        if filepath is None:
            filepath = self.config['paths']['raw_data']
        
        try:
            df = pd.read_csv(filepath)
            logging.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            logging.error(f"Data file not found: {filepath}")
            raise
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise
    
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame has required columns and structure.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if validation passes
        """
        # Expected columns (flexible - adjust based on actual dataset)
        expected_numeric = ['age', 'person_income', 'loan_amnt', 'loan_int_rate']
        expected_categorical = ['person_home_ownership', 'loan_intent']
        
        # Check if DataFrame is not empty
        if df.empty:
            logging.error("DataFrame is empty")
            return False
        
        # Log data types
        logging.info(f"Data types:\n{df.dtypes}")
        
        # Check for completely null columns
        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            logging.warning(f"Columns with all null values: {null_cols}")
        
        logging.info("Data validation completed")
        return True
    
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame,
        strategy: str = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        if strategy is None:
            strategy = self.config['preprocessing']['handle_missing']['strategy']
        
        df = df.copy()
        
        # Log missing values before handling
        missing_summary = check_missing_values(df)
        if not missing_summary.empty:
            logging.info(f"Missing values summary:\n{missing_summary}")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle numeric columns
        for col in numeric_cols:
            if df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df[col].mean()
                elif strategy == 'median':
                    fill_value = df[col].median()
                elif strategy == 'drop':
                    df = df.dropna(subset=[col])
                    continue
                else:
                    fill_value = df[col].median()
                
                df[col].fillna(fill_value, inplace=True)
                logging.info(f"Filled {col} with {strategy}: {fill_value:.2f}")
        
        # Handle categorical columns
        for col in categorical_cols:
            if df[col].isnull().any():
                if strategy == 'drop':
                    df = df.dropna(subset=[col])
                else:
                    # Fill with mode
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(fill_value, inplace=True)
                    logging.info(f"Filled {col} with mode: {fill_value}")
        
        logging.info(f"Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")
        return df
    
    
    def detect_and_treat_outliers(
        self,
        df: pd.DataFrame,
        method: str = None,
        threshold: float = None,
        treat: bool = True
    ) -> pd.DataFrame:
        """
        Detect and optionally treat outliers in numeric columns.
        
        Args:
            df: Input DataFrame
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold value for outlier detection
            treat: Whether to treat outliers (cap at boundaries)
            
        Returns:
            DataFrame with outliers treated (if treat=True)
        """
        if method is None:
            method = self.config['preprocessing']['outlier_detection']['method']
        if threshold is None:
            threshold = self.config['preprocessing']['outlier_detection']['threshold']
        
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if present
        if 'loan_status' in numeric_cols:
            numeric_cols.remove('loan_status')
        
        if method == 'iqr':
            outlier_counts = detect_outliers_iqr(df, numeric_cols, threshold)
            
            if treat:
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    # Cap outliers
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                logging.info(f"Outliers capped using IQR method (threshold={threshold})")
        
        elif method == 'zscore':
            from scipy import stats
            
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = (z_scores > threshold).sum()
                
                if treat and outliers > 0:
                    # Cap at mean ± threshold*std
                    mean = df[col].mean()
                    std = df[col].std()
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                    
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                logging.info(f"{col}: {outliers} outliers detected")
            
            if treat:
                logging.info(f"Outliers capped using Z-score method (threshold={threshold})")
        
        return df
    
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        df = df.copy()
        initial_rows = len(df)
        df = df.drop_duplicates()
        final_rows = len(df)
        
        removed = initial_rows - final_rows
        if removed > 0:
            logging.info(f"Removed {removed} duplicate rows")
        else:
            logging.info("No duplicate rows found")
        
        return df
    
    
    def correct_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Correct data types for specific columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with corrected data types
        """
        df = df.copy()
        
        # Convert categorical columns if they're object type
        categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 
                          'cb_person_default_on_file']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
                logging.info(f"Converted {col} to category type")
        
        # Ensure numeric columns are numeric
        numeric_cols = ['age', 'person_income', 'person_emp_length', 'loan_amnt',
                       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logging.info("Data types corrected")
        return df
    
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        target_col: str = 'loan_status',
        test_size: float = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            test_size: Proportion of test set
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if test_size is None:
            test_size = self.config['preprocessing']['test_size']
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logging.info(f"Train set: {len(X_train)} samples")
        logging.info(f"Test set: {len(X_test)} samples")
        logging.info(f"Class distribution in train: {y_train.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    
    def preprocess_pipeline(
        self,
        input_path: str = None,
        output_dir: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            input_path: Path to raw data file
            output_dir: Directory to save processed data
            
        Returns:
            Dictionary containing processed dataframes
        """
        logging.info("Starting preprocessing pipeline")
        
        # Load data
        df = self.load_data(input_path)
        
        # Validate data
        self.validate_data(df)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Correct data types
        df = self.correct_data_types(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Detect and treat outliers
        df = self.detect_and_treat_outliers(df, treat=True)
        
        # Save processed data
        if output_dir is None:
            output_dir = self.config['paths']['processed_data']
        
        output_path = os.path.join(output_dir, 'cleaned_data.csv')
        save_dataframe(df, output_path)
        
        logging.info("Preprocessing pipeline completed successfully")
        
        return {
            'processed_data': df,
            'output_path': output_path
        }


def main():
    """
    Main function to run the preprocessing pipeline.
    """
    print("="*60)
    print("Credit Risk Model - Data Preprocessing")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run preprocessing pipeline
    try:
        result = preprocessor.preprocess_pipeline()
        
        print("\n✓ Preprocessing completed successfully!")
        print(f"✓ Processed data saved to: {result['output_path']}")
        print(f"✓ Total rows: {len(result['processed_data'])}")
        print(f"✓ Total columns: {len(result['processed_data'].columns)}")
        
    except Exception as e:
        print(f"\n✗ Preprocessing failed: {e}")
        logging.error(f"Preprocessing failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
