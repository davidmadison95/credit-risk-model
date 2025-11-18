"""
SHAP Explainability Module for Credit Risk Scoring Model

This module handles:
- SHAP value calculation for all models
- Summary plots
- Feature importance visualization
- Force plots for individual predictions
- Waterfall plots
- Model interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Any, Dict, List, Tuple, Optional
import shap
import os

# Import utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logging, load_model, load_dataframe


class SHAPExplainer:
    """
    Handles SHAP-based model explainability.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize SHAP explainer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        
        # Setup logging
        setup_logging(
            log_file=self.config['logging']['file'],
            level=self.config['logging']['level']
        )
        
        self.explainers = {}
        self.shap_values = {}
        
        logging.info("SHAPExplainer initialized")
    
    
    def create_explainer(
        self,
        model: Any,
        X_background: pd.DataFrame,
        model_type: str = 'tree'
    ) -> shap.Explainer:
        """
        Create SHAP explainer for a model.
        
        Args:
            model: Trained model
            X_background: Background dataset for SHAP
            model_type: Type of model ('tree', 'linear', 'kernel')
            
        Returns:
            SHAP explainer object
        """
        logging.info(f"Creating SHAP explainer (type: {model_type})...")
        
        if model_type == 'tree':
            # For tree-based models (Random Forest, XGBoost)
            explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            # For linear models (Logistic Regression)
            explainer = shap.LinearExplainer(model, X_background)
        elif model_type == 'kernel':
            # For any model (slower but universal)
            explainer = shap.KernelExplainer(model.predict_proba, X_background)
        else:
            # Default to TreeExplainer
            explainer = shap.TreeExplainer(model)
        
        logging.info("SHAP explainer created successfully")
        return explainer
    
    
    def calculate_shap_values(
        self,
        explainer: shap.Explainer,
        X: pd.DataFrame,
        sample_size: int = None
    ) -> np.ndarray:
        """
        Calculate SHAP values for a dataset.
        
        Args:
            explainer: SHAP explainer object
            X: Feature dataset
            sample_size: Number of samples to explain (for speed)
            
        Returns:
            SHAP values array
        """
        logging.info("Calculating SHAP values...")
        
        # Sample data if needed for speed
        if sample_size and len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        logging.info(f"SHAP values calculated for {len(X_sample)} samples")
        return shap_values
    
    
    def plot_summary(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        model_name: str,
        save_path: str = None,
        max_display: int = 20
    ) -> None:
        """
        Create SHAP summary plot.
        
        Args:
            shap_values: SHAP values array
            X: Feature dataset
            model_name: Name of the model
            save_path: Path to save figure
            max_display: Maximum features to display
        """
        plt.figure(figsize=(10, 8))
        
        # For multi-class, use the values for all classes
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary
        
        shap.summary_plot(
            shap_values, X,
            max_display=max_display,
            show=False
        )
        
        plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"SHAP summary plot saved to: {save_path}")
        
        plt.show()
    
    
    def plot_bar(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        model_name: str,
        save_path: str = None,
        max_display: int = 20
    ) -> None:
        """
        Create SHAP bar plot (feature importance).
        
        Args:
            shap_values: SHAP values array
            X: Feature dataset
            model_name: Name of the model
            save_path: Path to save figure
            max_display: Maximum features to display
        """
        plt.figure(figsize=(10, 8))
        
        # For multi-class, use the values for all classes
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary
        
        shap.summary_plot(
            shap_values, X,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        
        plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"SHAP bar plot saved to: {save_path}")
        
        plt.show()
    
    
    def plot_waterfall(
        self,
        explainer: shap.Explainer,
        X: pd.DataFrame,
        instance_index: int,
        model_name: str,
        save_path: str = None
    ) -> None:
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            explainer: SHAP explainer object
            X: Feature dataset
            instance_index: Index of instance to explain
            model_name: Name of the model
            save_path: Path to save figure
        """
        plt.figure(figsize=(10, 8))
        
        # Get SHAP values for the instance
        shap_values = explainer.shap_values(X.iloc[instance_index:instance_index+1])
        
        # For multi-class, use positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create explanation object
        if hasattr(shap, 'Explanation'):
            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[1],
                data=X.iloc[instance_index].values,
                feature_names=X.columns.tolist()
            )
            shap.waterfall_plot(explanation, show=False)
        else:
            # Fallback for older SHAP versions
            logging.warning("Waterfall plot requires SHAP >= 0.40.0")
            return
        
        plt.title(f'SHAP Waterfall Plot - {model_name} (Instance {instance_index})', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"SHAP waterfall plot saved to: {save_path}")
        
        plt.show()
    
    
    def plot_force(
        self,
        explainer: shap.Explainer,
        X: pd.DataFrame,
        instance_index: int,
        model_name: str,
        save_path: str = None
    ) -> None:
        """
        Create SHAP force plot for a single prediction.
        
        Args:
            explainer: SHAP explainer object
            X: Feature dataset
            instance_index: Index of instance to explain
            model_name: Name of the model
            save_path: Path to save figure
        """
        # Get SHAP values for the instance
        shap_values = explainer.shap_values(X.iloc[instance_index:instance_index+1])
        
        # For multi-class, use positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
        else:
            expected_value = explainer.expected_value
        
        # Create force plot
        shap.force_plot(
            expected_value,
            shap_values[0],
            X.iloc[instance_index],
            matplotlib=True,
            show=False
        )
        
        plt.title(f'SHAP Force Plot - {model_name} (Instance {instance_index})', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"SHAP force plot saved to: {save_path}")
        
        plt.show()
    
    
    def get_feature_contributions(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get top feature contributions based on mean absolute SHAP values.
        
        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature contributions
        """
        # For multi-class, use positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame
        contributions_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean_Absolute_SHAP': mean_abs_shap
        }).sort_values(by='Mean_Absolute_SHAP', ascending=False).head(top_n)
        
        return contributions_df
    
    
    def explain_all_models(
        self,
        models_dict: Dict[str, Any],
        X: pd.DataFrame,
        output_dir: str = None,
        sample_size: int = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate SHAP explanations for all models.
        
        Args:
            models_dict: Dictionary of model names to trained models
            X: Feature dataset
            output_dir: Directory to save outputs
            sample_size: Number of samples for SHAP calculation
            
        Returns:
            Dictionary of model names to SHAP values
        """
        if output_dir is None:
            output_dir = self.config['paths']['outputs']
        
        if sample_size is None:
            sample_size = self.config['explainability']['shap']['sample_size']
        
        figures_dir = os.path.join(output_dir, 'figures')
        all_shap_values = {}
        
        # Use a subset for background if dataset is large
        X_background = X.sample(n=min(100, len(X)), random_state=42)
        
        for model_name, model in models_dict.items():
            logging.info(f"Explaining {model_name}...")
            
            # Determine model type
            if 'logistic' in model_name:
                model_type = 'linear'
            elif 'forest' in model_name or 'xgboost' in model_name:
                model_type = 'tree'
            else:
                model_type = 'tree'
            
            # Create explainer
            explainer = self.create_explainer(model, X_background, model_type)
            self.explainers[model_name] = explainer
            
            # Calculate SHAP values
            shap_values = self.calculate_shap_values(explainer, X, sample_size)
            all_shap_values[model_name] = shap_values
            self.shap_values[model_name] = shap_values
            
            # Sample data for plotting (match SHAP values)
            if sample_size and len(X) > sample_size:
                X_plot = X.sample(n=sample_size, random_state=42)
            else:
                X_plot = X
            
            # Generate visualizations
            max_display = self.config['explainability']['shap']['max_display']
            
            # Summary plot
            self.plot_summary(
                shap_values, X_plot, model_name.upper(),
                save_path=os.path.join(figures_dir, f'{model_name}_shap_summary.png'),
                max_display=max_display
            )
            
            # Bar plot
            self.plot_bar(
                shap_values, X_plot, model_name.upper(),
                save_path=os.path.join(figures_dir, f'{model_name}_shap_bar.png'),
                max_display=max_display
            )
            
            # Waterfall plot for first instance
            try:
                self.plot_waterfall(
                    explainer, X_plot, 0, model_name.upper(),
                    save_path=os.path.join(figures_dir, f'{model_name}_shap_waterfall.png')
                )
            except Exception as e:
                logging.warning(f"Could not create waterfall plot: {e}")
            
            # Force plot for first instance
            try:
                self.plot_force(
                    explainer, X_plot, 0, model_name.upper(),
                    save_path=os.path.join(figures_dir, f'{model_name}_shap_force.png')
                )
            except Exception as e:
                logging.warning(f"Could not create force plot: {e}")
            
            # Get top feature contributions
            contributions = self.get_feature_contributions(
                shap_values, X_plot.columns.tolist(), top_n=10
            )
            
            print(f"\n{'='*60}")
            print(f"Top 10 Feature Contributions - {model_name.upper()}")
            print(f"{'='*60}")
            print(contributions.to_string(index=False))
            print(f"{'='*60}\n")
        
        logging.info("SHAP explanations generated for all models")
        return all_shap_values


def main():
    """
    Main function to run SHAP explainability analysis.
    """
    print("="*60)
    print("Credit Risk Model - SHAP Explainability")
    print("="*60)
    
    # Initialize explainer
    explainer = SHAPExplainer()
    
    try:
        # Load models
        models_dir = explainer.config['paths']['models']
        models = {
            'logistic_regression': load_model(os.path.join(models_dir, 'logistic_regression.pkl')),
            'random_forest': load_model(os.path.join(models_dir, 'random_forest.pkl')),
            'xgboost': load_model(os.path.join(models_dir, 'xgboost.pkl'))
        }
        
        print("\n✓ Models loaded successfully")
        
        # Load data
        from src.train_models import ModelTrainer
        trainer = ModelTrainer()
        X, y = trainer.load_data()
        _, X_test, _, _ = trainer.split_data(X, y, apply_smote=False)
        
        print("✓ Test data loaded")
        print(f"  Samples: {len(X_test)}")
        print(f"  Features: {len(X_test.columns)}")
        
        # Generate SHAP explanations
        print("\nGenerating SHAP explanations...")
        print("This may take 2-5 minutes...\n")
        
        shap_values = explainer.explain_all_models(models, X_test, sample_size=100)
        
        print("\n" + "="*60)
        print("SHAP ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n✓ All SHAP visualizations saved to: outputs/figures/")
        print("✓ Generated for all 3 models:")
        print("  - Summary plots")
        print("  - Feature importance bar plots")
        print("  - Waterfall plots")
        print("  - Force plots")
        
    except Exception as e:
        print(f"\n✗ SHAP analysis failed: {e}")
        logging.error(f"SHAP analysis failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
