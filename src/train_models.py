"""
Model Training Module for Credit Risk Scoring Model

This module handles:
- Training Logistic Regression, Random Forest, and XGBoost models
- Hyperparameter tuning using GridSearchCV
- Handling class imbalance with SMOTE
- Model persistence
- Training metrics and logging
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any, List
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import time

# Import utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    load_config,
    setup_logging,
    save_model,
    load_model,
    load_dataframe,
    set_random_seed,
    print_metrics
)


class ModelTrainer:
    """
    Handles training and tuning of machine learning models.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize model trainer with configuration.
        
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
        
        # Initialize models dictionary
        self.models = {}
        self.best_models = {}
        self.training_history = {}
        
        logging.info("ModelTrainer initialized")
    
    
    def load_data(self, filepath: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load engineered features and prepare for training.
        
        Args:
            filepath: Path to engineered features file
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if filepath is None:
            filepath = os.path.join(
                self.config['paths']['processed_data'],
                'engineered_features.csv'
            )
        
        df = load_dataframe(filepath)
        
        # Determine target column
        if 'risk_tier_encoded' in df.columns:
            target_col = 'risk_tier_encoded'
            logging.info("Using risk_tier_encoded as target (multi-class)")
        elif 'loan_status' in df.columns:
            target_col = 'loan_status'
            logging.info("Using loan_status as target (binary)")
        else:
            raise ValueError("No target column found (risk_tier_encoded or loan_status)")
        
        # Separate features and target
        X = df.drop(columns=[col for col in ['risk_tier', 'loan_status', 'risk_tier_encoded'] 
                            if col in df.columns])
        y = df[target_col]
        
        logging.info(f"Data loaded: {len(X)} samples, {len(X.columns)} features")
        logging.info(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = None,
        apply_smote: bool = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets, optionally apply SMOTE.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion for test set
            apply_smote: Whether to apply SMOTE for class imbalance
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if test_size is None:
            test_size = self.config['preprocessing']['test_size']
        
        if apply_smote is None:
            apply_smote = self.config['preprocessing']['class_imbalance']['apply_smote']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logging.info(f"Train set: {len(X_train)} samples")
        logging.info(f"Test set: {len(X_test)} samples")
        
        # Apply SMOTE if requested
        if apply_smote:
            smote_strategy = self.config['preprocessing']['class_imbalance']['sampling_strategy']
            smote = SMOTE(sampling_strategy=smote_strategy, random_state=self.random_state)
            
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            logging.info(f"SMOTE applied. New train set: {len(X_train_resampled)} samples")
            logging.info(f"Class distribution after SMOTE:\n{pd.Series(y_train_resampled).value_counts()}")
            
            return X_train_resampled, X_test, y_train_resampled, y_test
        
        return X_train, X_test, y_train, y_test
    
    
    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune_hyperparameters: bool = False
    ) -> LogisticRegression:
        """
        Train Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Trained Logistic Regression model
        """
        logging.info("Training Logistic Regression...")
        start_time = time.time()
        
        if tune_hyperparameters and self.config['hyperparameter_tuning']['enabled']:
            # Hyperparameter tuning
            param_grid = self.config['hyperparameter_tuning']['logistic_regression_grid']
            
            lr = LogisticRegression(
                max_iter=self.config['models']['logistic_regression']['max_iter'],
                random_state=self.random_state
            )
            
            grid_search = GridSearchCV(
                lr,
                param_grid,
                cv=self.config['hyperparameter_tuning']['cv_folds'],
                scoring=self.config['hyperparameter_tuning']['scoring'],
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            
            logging.info(f"Best parameters: {grid_search.best_params_}")
            logging.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters from config
            model = LogisticRegression(
                penalty=self.config['models']['logistic_regression']['penalty'],
                C=self.config['models']['logistic_regression']['C'],
                max_iter=self.config['models']['logistic_regression']['max_iter'],
                solver=self.config['models']['logistic_regression']['solver'],
                class_weight=self.config['models']['logistic_regression']['class_weight'],
                random_state=self.random_state
            )
            
            model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        logging.info(f"Logistic Regression training completed in {training_time:.2f} seconds")
        
        self.models['logistic_regression'] = model
        self.training_history['logistic_regression'] = {'training_time': training_time}
        
        return model
    
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune_hyperparameters: bool = False
    ) -> RandomForestClassifier:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Trained Random Forest model
        """
        logging.info("Training Random Forest...")
        start_time = time.time()
        
        if tune_hyperparameters and self.config['hyperparameter_tuning']['enabled']:
            # Hyperparameter tuning
            param_grid = self.config['hyperparameter_tuning']['random_forest_grid']
            
            rf = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            )
            
            grid_search = GridSearchCV(
                rf,
                param_grid,
                cv=self.config['hyperparameter_tuning']['cv_folds'],
                scoring=self.config['hyperparameter_tuning']['scoring'],
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            
            logging.info(f"Best parameters: {grid_search.best_params_}")
            logging.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters from config
            model = RandomForestClassifier(
                n_estimators=self.config['models']['random_forest']['n_estimators'],
                max_depth=self.config['models']['random_forest']['max_depth'],
                min_samples_split=self.config['models']['random_forest']['min_samples_split'],
                min_samples_leaf=self.config['models']['random_forest']['min_samples_leaf'],
                max_features=self.config['models']['random_forest']['max_features'],
                class_weight=self.config['models']['random_forest']['class_weight'],
                bootstrap=self.config['models']['random_forest']['bootstrap'],
                random_state=self.random_state,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        logging.info(f"Random Forest training completed in {training_time:.2f} seconds")
        
        self.models['random_forest'] = model
        self.training_history['random_forest'] = {'training_time': training_time}
        
        return model
    
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune_hyperparameters: bool = False
    ) -> XGBClassifier:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Trained XGBoost model
        """
        logging.info("Training XGBoost...")
        start_time = time.time()
        
        if tune_hyperparameters and self.config['hyperparameter_tuning']['enabled']:
            # Hyperparameter tuning
            param_grid = self.config['hyperparameter_tuning']['xgboost_grid']
            
            xgb = XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
            
            grid_search = GridSearchCV(
                xgb,
                param_grid,
                cv=self.config['hyperparameter_tuning']['cv_folds'],
                scoring=self.config['hyperparameter_tuning']['scoring'],
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            
            logging.info(f"Best parameters: {grid_search.best_params_}")
            logging.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters from config
            model = XGBClassifier(
                n_estimators=self.config['models']['xgboost']['n_estimators'],
                max_depth=self.config['models']['xgboost']['max_depth'],
                learning_rate=self.config['models']['xgboost']['learning_rate'],
                subsample=self.config['models']['xgboost']['subsample'],
                colsample_bytree=self.config['models']['xgboost']['colsample_bytree'],
                gamma=self.config['models']['xgboost']['gamma'],
                reg_alpha=self.config['models']['xgboost']['reg_alpha'],
                reg_lambda=self.config['models']['xgboost']['reg_lambda'],
                scale_pos_weight=self.config['models']['xgboost']['scale_pos_weight'],
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
            
            model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        logging.info(f"XGBoost training completed in {training_time:.2f} seconds")
        
        self.models['xgboost'] = model
        self.training_history['xgboost'] = {'training_time': training_time}
        
        return model
    
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logging.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # ROC-AUC (handle multi-class)
        try:
            if len(np.unique(y_test)) == 2:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                # Multi-class classification
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            logging.warning(f"Could not calculate ROC-AUC: {e}")
            metrics['roc_auc'] = 0.0
        
        logging.info(f"{model_name} evaluation completed")
        print_metrics(metrics, model_name)
        
        return metrics
    
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        tune_hyperparameters: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all three models and evaluate them.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            tune_hyperparameters: Whether to tune hyperparameters
            
        Returns:
            Dictionary of model names to metrics
        """
        logging.info("Training all models...")
        all_metrics = {}
        
        # Train Logistic Regression
        lr_model = self.train_logistic_regression(X_train, y_train, tune_hyperparameters)
        lr_metrics = self.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        all_metrics['logistic_regression'] = lr_metrics
        
        # Train Random Forest
        rf_model = self.train_random_forest(X_train, y_train, tune_hyperparameters)
        rf_metrics = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
        all_metrics['random_forest'] = rf_metrics
        
        # Train XGBoost
        xgb_model = self.train_xgboost(X_train, y_train, tune_hyperparameters)
        xgb_metrics = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        all_metrics['xgboost'] = xgb_metrics
        
        logging.info("All models trained and evaluated")
        
        return all_metrics
    
    
    def save_models(self, models_dir: str = None) -> None:
        """
        Save all trained models to disk.
        
        Args:
            models_dir: Directory to save models
        """
        if models_dir is None:
            models_dir = self.config['paths']['models']
        
        for model_name, model in self.models.items():
            filepath = os.path.join(models_dir, f"{model_name}.pkl")
            save_model(model, filepath)
        
        logging.info(f"All models saved to {models_dir}")
    
    
    def create_comparison_table(
        self,
        all_metrics: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Create comparison table of model performance.
        
        Args:
            all_metrics: Dictionary of model metrics
            
        Returns:
            DataFrame with comparison table
        """
        comparison_df = pd.DataFrame(all_metrics).T
        comparison_df = comparison_df.round(4)
        
        # Add training times if available
        if self.training_history:
            comparison_df['training_time'] = [
                self.training_history.get(model, {}).get('training_time', 0.0)
                for model in comparison_df.index
            ]
        
        # Sort by F1 score (descending)
        comparison_df = comparison_df.sort_values(by='f1', ascending=False)
        
        logging.info("\nModel Comparison Table:")
        logging.info(f"\n{comparison_df}")
        
        return comparison_df
    
    
    def training_pipeline(
        self,
        tune_hyperparameters: bool = False,
        apply_smote: bool = None
    ) -> Dict[str, Any]:
        """
        Execute complete model training pipeline.
        
        Args:
            tune_hyperparameters: Whether to tune hyperparameters
            apply_smote: Whether to apply SMOTE
            
        Returns:
            Dictionary with training results
        """
        logging.info("Starting model training pipeline")
        
        # Load data
        X, y = self.load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, apply_smote=apply_smote)
        
        # Train all models
        all_metrics = self.train_all_models(
            X_train, X_test, y_train, y_test,
            tune_hyperparameters=tune_hyperparameters
        )
        
        # Create comparison table
        comparison_df = self.create_comparison_table(all_metrics)
        
        # Save comparison table
        output_dir = self.config['paths']['outputs']
        comparison_path = os.path.join(output_dir, 'reports', 'model_comparison.csv')
        os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
        comparison_df.to_csv(comparison_path)
        logging.info(f"Comparison table saved to: {comparison_path}")
        
        # Save models
        self.save_models()
        
        logging.info("Model training pipeline completed successfully")
        
        return {
            'models': self.models,
            'metrics': all_metrics,
            'comparison': comparison_df,
            'X_test': X_test,
            'y_test': y_test
        }


def main():
    """
    Main function to run model training pipeline.
    """
    print("="*60)
    print("Credit Risk Model - Model Training")
    print("="*60)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Run training pipeline
    try:
        print("\nStarting training pipeline...")
        print("This may take a few minutes...\n")
        
        result = trainer.training_pipeline(
            tune_hyperparameters=False,  # Set to True for hyperparameter tuning
            apply_smote=True
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nModel Comparison:")
        print(result['comparison'])
        
        print(f"\n✓ Models saved to: models/")
        print(f"✓ Comparison table saved to: outputs/reports/model_comparison.csv")
        
        # Identify best model
        best_model = result['comparison'].index[0]
        best_f1 = result['comparison'].loc[best_model, 'f1']
        
        print(f"\n🏆 Best Model: {best_model.upper()}")
        print(f"   F1 Score: {best_f1:.4f}")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        logging.error(f"Training failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
