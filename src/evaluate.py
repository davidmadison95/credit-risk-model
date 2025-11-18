"""
Model Evaluation Module for Credit Risk Scoring Model

This module handles:
- Comprehensive model evaluation metrics
- Confusion matrix visualization
- ROC curve plotting
- Precision-Recall curves
- Classification reports
- Model comparison visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import os

# Import utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logging, load_model, load_dataframe


class ModelEvaluator:
    """
    Handles model evaluation and visualization.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize model evaluator with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        
        # Setup logging
        setup_logging(
            log_file=self.config['logging']['file'],
            level=self.config['logging']['level']
        )
        
        logging.info("ModelEvaluator initialized")
    
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save_path: str = None,
        labels: List[str] = None
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save figure
            labels: Class labels
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, square=True)
        
        if labels:
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        else:
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('Actual', fontsize=12)
            plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str,
        save_path: str = None,
        n_classes: int = None
    ) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save figure
            n_classes: Number of classes (for multi-class)
        """
        plt.figure(figsize=(10, 8))
        
        # Determine if binary or multi-class
        unique_classes = np.unique(y_true)
        
        if len(unique_classes) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            
        else:
            # Multi-class classification
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=unique_classes)
            
            # Compute ROC curve and AUC for each class
            for i in range(len(unique_classes)):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'Class {unique_classes[i]} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"ROC curve saved to: {save_path}")
        
        plt.show()
    
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str,
        save_path: str = None
    ) -> None:
        """
        Plot Precision-Recall curve (for binary classification).
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save figure
        """
        # Only for binary classification
        if len(np.unique(y_true)) != 2:
            logging.warning("Precision-Recall curve only for binary classification")
            return
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkgreen', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Precision-Recall curve saved to: {save_path}")
        
        plt.show()
    
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save_path: str = None
    ) -> str:
        """
        Generate and save classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save report
            
        Returns:
            Classification report as string
        """
        report = classification_report(y_true, y_pred)
        
        print(f"\n{'='*60}")
        print(f"Classification Report - {model_name}")
        print(f"{'='*60}")
        print(report)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(f"Classification Report - {model_name}\n")
                f.write("="*60 + "\n")
                f.write(report)
            logging.info(f"Classification report saved to: {save_path}")
        
        return report
    
    
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str,
        save_path: str = None,
        top_n: int = 20
    ) -> None:
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name of the model
            save_path: Path to save figure
            top_n: Number of top features to display
        """
        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            logging.warning(f"{model_name} does not have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices], color='steelblue')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
    
    
    def compare_models_visual(
        self,
        comparison_df: pd.DataFrame,
        save_path: str = None
    ) -> None:
        """
        Create visual comparison of model performance.
        
        Args:
            comparison_df: DataFrame with model comparison metrics
            save_path: Path to save figure
        """
        # Select key metrics for comparison
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.15
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, metric in enumerate(available_metrics):
            offset = width * (i - len(available_metrics)/2)
            ax.bar(x + offset, comparison_df[metric], width, 
                  label=metric.upper(), color=colors[i % len(colors)])
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df.index, rotation=15, ha='right')
        ax.legend(loc='upper left', fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Model comparison plot saved to: {save_path}")
        
        plt.show()
    
    
    def evaluate_all_models(
        self,
        models_dict: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        output_dir: str = None
    ) -> None:
        """
        Evaluate all models and generate visualizations.
        
        Args:
            models_dict: Dictionary of model names to trained models
            X_test: Test features
            y_test: Test target
            output_dir: Directory to save outputs
        """
        if output_dir is None:
            output_dir = self.config['paths']['outputs']
        
        figures_dir = os.path.join(output_dir, 'figures')
        reports_dir = os.path.join(output_dir, 'reports')
        
        for model_name, model in models_dict.items():
            logging.info(f"Evaluating {model_name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Confusion Matrix
            self.plot_confusion_matrix(
                y_test, y_pred, model_name.upper(),
                save_path=os.path.join(figures_dir, f'{model_name}_confusion_matrix.png')
            )
            
            # ROC Curve
            self.plot_roc_curve(
                y_test, y_pred_proba, model_name.upper(),
                save_path=os.path.join(figures_dir, f'{model_name}_roc_curve.png')
            )
            
            # Precision-Recall Curve (binary only)
            if len(np.unique(y_test)) == 2:
                self.plot_precision_recall_curve(
                    y_test, y_pred_proba, model_name.upper(),
                    save_path=os.path.join(figures_dir, f'{model_name}_pr_curve.png')
                )
            
            # Classification Report
            self.generate_classification_report(
                y_test, y_pred, model_name.upper(),
                save_path=os.path.join(reports_dir, f'{model_name}_classification_report.txt')
            )
            
            # Feature Importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                self.plot_feature_importance(
                    model, X_test.columns.tolist(), model_name.upper(),
                    save_path=os.path.join(figures_dir, f'{model_name}_feature_importance.png'),
                    top_n=20
                )
        
        logging.info("All models evaluated successfully")


def main():
    """
    Main function to run model evaluation.
    """
    print("="*60)
    print("Credit Risk Model - Model Evaluation")
    print("="*60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    try:
        # Load models
        models_dir = evaluator.config['paths']['models']
        models = {
            'logistic_regression': load_model(os.path.join(models_dir, 'logistic_regression.pkl')),
            'random_forest': load_model(os.path.join(models_dir, 'random_forest.pkl')),
            'xgboost': load_model(os.path.join(models_dir, 'xgboost.pkl'))
        }
        
        print("\n✓ Models loaded successfully")
        
        # Load test data (you'll need to save this during training)
        # For now, we'll reload and split the data
        from src.train_models import ModelTrainer
        trainer = ModelTrainer()
        X, y = trainer.load_data()
        _, X_test, _, y_test = trainer.split_data(X, y, apply_smote=False)
        
        print("✓ Test data loaded")
        
        # Evaluate all models
        print("\nGenerating evaluation visualizations...")
        evaluator.evaluate_all_models(models, X_test, y_test)
        
        # Load and visualize comparison
        comparison_df = pd.read_csv('outputs/reports/model_comparison.csv', index_col=0)
        evaluator.compare_models_visual(
            comparison_df,
            save_path='outputs/figures/model_comparison.png'
        )
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n✓ All visualizations saved to: outputs/figures/")
        print("✓ All reports saved to: outputs/reports/")
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        logging.error(f"Evaluation failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
