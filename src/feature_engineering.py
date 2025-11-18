"""
Feature Engineering Module for Credit Risk Scoring Model

This module handles:
- Risk tier engineering (Low, Medium, High)
- Feature creation and transformation
- Categorical encoding
- Numerical scaling
- Feature selection (mutual info, correlation, model-based)
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Import utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    load_config,
    setup_logging,
    save_dataframe,
    load_dataframe,
    save_model,
    set_random_seed
)


class FeatureEngineer:
    """
    Handles all feature engineering steps for credit risk model.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize feature engineer with configuration.
        
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
        
        # Initialize scalers and encoders
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        
        logging.info("FeatureEngineer initialized")
    
    
    def engineer_risk_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer risk tiers (Low, Medium, High) from loan_status and other features.
        
        Risk Tier Logic:
        - Low Risk: No defaults, low loan-to-income, low interest rate
        - Medium Risk: Moderate risk factors
        - High Risk: Historical defaults or high-risk indicators
        
        Args:
            df: Input DataFrame with loan_status
            
        Returns:
            DataFrame with 'risk_tier' column
        """
        df = df.copy()
        
        # Initialize risk tier column
        df['risk_tier'] = 'Medium'  # Default
        
        # Get risk tier configuration
        risk_config = self.config['risk_tiers']
        
        # Low Risk Criteria
        low_risk_conditions = (
            (df['loan_status'] == 0) &
            (df.get('loan_percent_income', 1.0) < risk_config['low_risk']['loan_to_income_max']) &
            (df.get('loan_int_rate', 100) < risk_config['low_risk']['interest_rate_max']) &
            (df.get('cb_person_default_on_file', 'Y') == 'N')
        )
        
        # Additional low risk check for loan grade if available
        if 'loan_grade' in df.columns:
            low_risk_grades = risk_config['low_risk']['grades']
            low_risk_conditions &= df['loan_grade'].isin(low_risk_grades)
        
        df.loc[low_risk_conditions, 'risk_tier'] = 'Low'
        
        # High Risk Criteria
        high_risk_conditions = (
            (df['loan_status'] == 1) |
            (df.get('loan_percent_income', 0) > risk_config['high_risk']['loan_to_income_min']) |
            (df.get('loan_int_rate', 0) > risk_config['high_risk']['interest_rate_min']) |
            (df.get('cb_person_default_on_file', 'N') == 'Y')
        )
        
        # Additional high risk check for loan grade if available
        if 'loan_grade' in df.columns:
            high_risk_grades = risk_config['high_risk']['grades']
            high_risk_conditions |= df['loan_grade'].isin(high_risk_grades)
        
        df.loc[high_risk_conditions, 'risk_tier'] = 'High'
        
        # Log distribution
        tier_dist = df['risk_tier'].value_counts()
        logging.info(f"Risk tier distribution:\n{tier_dist}")
        logging.info(f"Low: {tier_dist.get('Low', 0)/len(df)*100:.1f}%, "
                    f"Medium: {tier_dist.get('Medium', 0)/len(df)*100:.1f}%, "
                    f"High: {tier_dist.get('High', 0)/len(df)*100:.1f}%")
        
        return df
    
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new derived features
        """
        df = df.copy()
        
        # Debt-to-income ratio (if not already present)
        if 'loan_amnt' in df.columns and 'person_income' in df.columns:
            df['debt_to_income_ratio'] = df['loan_amnt'] / df['person_income']
            logging.info("Created: debt_to_income_ratio")
        
        # Income per year employed
        if 'person_income' in df.columns and 'person_emp_length' in df.columns:
            df['income_per_year_employed'] = df['person_income'] / (df['person_emp_length'] + 1)
            logging.info("Created: income_per_year_employed")
        
        # Age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0, 25, 35, 45, 55, 100],
                labels=['18-25', '26-35', '36-45', '46-55', '56+']
            )
            logging.info("Created: age_group")
        
        # High interest rate flag
        if 'loan_int_rate' in df.columns:
            df['high_interest_flag'] = (df['loan_int_rate'] > 15).astype(int)
            logging.info("Created: high_interest_flag")
        
        # Short credit history flag
        if 'cb_person_cred_hist_length' in df.columns:
            df['short_credit_history'] = (df['cb_person_cred_hist_length'] < 3).astype(int)
            logging.info("Created: short_credit_history")
        
        # Employment stability indicator
        if 'person_emp_length' in df.columns:
            df['employment_stable'] = (df['person_emp_length'] >= 2).astype(int)
            logging.info("Created: employment_stable")
        
        # Loan amount category
        if 'loan_amnt' in df.columns:
            df['loan_amount_category'] = pd.cut(
                df['loan_amnt'],
                bins=[0, 5000, 10000, 20000, 100000],
                labels=['Small', 'Medium', 'Large', 'Very Large']
            )
            logging.info("Created: loan_amount_category")
        
        # Income category
        if 'person_income' in df.columns:
            df['income_category'] = pd.cut(
                df['person_income'],
                bins=[0, 30000, 60000, 100000, 1000000],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            logging.info("Created: income_category")
        
        logging.info(f"Total features after creation: {len(df.columns)}")
        return df
    
    
    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        method: str = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            method: Encoding method ('onehot' or 'label')
            fit: Whether to fit encoders (True for training, False for test)
            
        Returns:
            DataFrame with encoded features
        """
        if method is None:
            method = self.config['feature_engineering']['encoding']['categorical_method']
        
        df = df.copy()
        
        # Get categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target if present
        if 'risk_tier' in categorical_cols:
            categorical_cols.remove('risk_tier')
        if 'loan_status' in categorical_cols:
            categorical_cols.remove('loan_status')
        
        if method == 'onehot':
            # One-hot encoding
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
            logging.info(f"One-hot encoded {len(categorical_cols)} categorical features")
            
        elif method == 'label':
            # Label encoding
            for col in categorical_cols:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Handle unseen labels
                        le = self.label_encoders[col]
                        df[col] = df[col].apply(
                            lambda x: le.transform([str(x)])[0] 
                            if str(x) in le.classes_ 
                            else -1
                        )
            logging.info(f"Label encoded {len(categorical_cols)} categorical features")
        
        return df
    
    
    def scale_numerical_features(
        self,
        df: pd.DataFrame,
        method: str = None,
        fit: bool = True,
        exclude_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            method: Scaling method ('standard', 'minmax', 'robust')
            fit: Whether to fit scaler (True for training, False for test)
            exclude_cols: Columns to exclude from scaling
            
        Returns:
            DataFrame with scaled features
        """
        if method is None:
            method = self.config['feature_engineering']['scaling']['method']
        
        if exclude_cols is None:
            exclude_cols = ['risk_tier', 'loan_status']
        
        df = df.copy()
        
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if not numerical_cols:
            logging.warning("No numerical columns to scale")
            return df
        
        # Select scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            logging.warning(f"Unknown scaling method: {method}, using StandardScaler")
            scaler = StandardScaler()
        
        # Fit and transform or just transform
        if fit:
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scaler = scaler
            logging.info(f"Fitted and scaled {len(numerical_cols)} features using {method} scaling")
        else:
            if self.scaler is None:
                logging.error("Scaler not fitted. Call with fit=True first.")
                raise ValueError("Scaler not fitted")
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            logging.info(f"Scaled {len(numerical_cols)} features using fitted scaler")
        
        return df
    
    
    def select_features_mutual_info(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = None
    ) -> Tuple[List[str], np.ndarray]:
        """
        Select features using mutual information.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            k: Number of top features to select
            
        Returns:
            Tuple of (selected feature names, importance scores)
        """
        if k is None:
            k = self.config['feature_engineering']['feature_selection']['n_features']
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        
        # Create feature importance DataFrame
        mi_df = pd.DataFrame({
            'Feature': X.columns,
            'MI_Score': mi_scores
        }).sort_values(by='MI_Score', ascending=False)
        
        # Select top k features
        selected_features = mi_df.head(k)['Feature'].tolist()
        
        logging.info(f"Selected {len(selected_features)} features using mutual information")
        logging.info(f"Top 5 features: {selected_features[:5]}")
        
        return selected_features, mi_scores
    
    
    def select_features_correlation(
        self,
        df: pd.DataFrame,
        threshold: float = None
    ) -> List[str]:
        """
        Remove highly correlated features.
        
        Args:
            df: Input DataFrame
            threshold: Correlation threshold
            
        Returns:
            List of features to keep
        """
        if threshold is None:
            threshold = self.config['feature_engineering']['feature_selection']['correlation_threshold']
        
        # Calculate correlation matrix
        corr_matrix = df.corr().abs()
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        features_to_keep = [col for col in df.columns if col not in to_drop]
        
        logging.info(f"Removed {len(to_drop)} highly correlated features (threshold={threshold})")
        if to_drop:
            logging.info(f"Dropped features: {to_drop}")
        
        return features_to_keep
    
    
    def select_features_model_based(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = None
    ) -> Tuple[List[str], np.ndarray]:
        """
        Select features using Random Forest feature importance.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            k: Number of top features to select
            
        Returns:
            Tuple of (selected feature names, importance scores)
        """
        if k is None:
            k = self.config['feature_engineering']['feature_selection']['n_features']
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Select top k features
        selected_features = importance_df.head(k)['Feature'].tolist()
        
        logging.info(f"Selected {len(selected_features)} features using Random Forest importance")
        logging.info(f"Top 5 features: {selected_features[:5]}")
        
        return selected_features, importances
    
    
    def visualize_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: np.ndarray,
        title: str = "Feature Importance",
        save_path: str = None,
        top_n: int = 20
    ) -> None:
        """
        Visualize feature importance scores.
        
        Args:
            feature_names: List of feature names
            importance_scores: Array of importance scores
            title: Plot title
            save_path: Path to save figure
            top_n: Number of top features to display
        """
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values(by='Importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['Importance'].values, color='steelblue')
        plt.yticks(range(len(importance_df)), importance_df['Feature'].values)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
    
    
    def feature_engineering_pipeline(
        self,
        input_path: str = None,
        output_dir: str = None,
        create_risk_tiers: bool = True,
        apply_feature_selection: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute complete feature engineering pipeline.
        
        Args:
            input_path: Path to preprocessed data
            output_dir: Directory to save engineered features
            create_risk_tiers: Whether to engineer risk tiers
            apply_feature_selection: Whether to apply feature selection
            
        Returns:
            Dictionary containing engineered dataframes
        """
        logging.info("Starting feature engineering pipeline")
        
        # Load preprocessed data
        if input_path is None:
            input_path = os.path.join(
                self.config['paths']['processed_data'],
                'cleaned_data.csv'
            )
        
        df = load_dataframe(input_path)
        
        # Engineer risk tiers if requested
        if create_risk_tiers and 'loan_status' in df.columns:
            df = self.engineer_risk_tiers(df)
            
            # Encode risk tiers to numeric for modeling
            risk_tier_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
            df['risk_tier_encoded'] = df['risk_tier'].map(risk_tier_mapping)
        
        # Create derived features
        df = self.create_derived_features(df)
        
        # Separate features and target
        target_cols = ['loan_status', 'risk_tier', 'risk_tier_encoded']
        available_targets = [col for col in target_cols if col in df.columns]
        
        feature_cols = [col for col in df.columns if col not in available_targets]
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        
        # Update feature columns after encoding
        feature_cols = [col for col in df.columns if col not in available_targets]
        
        # Scale numerical features
        df_scaled = df.copy()
        df_scaled = self.scale_numerical_features(df_scaled, fit=True, exclude_cols=available_targets)
        
        # Feature selection if requested
        selected_features = feature_cols.copy()
        
        if apply_feature_selection and self.config['feature_engineering']['feature_selection']['apply']:
            # Use risk_tier_encoded or loan_status as target for selection
            target_for_selection = 'risk_tier_encoded' if 'risk_tier_encoded' in df.columns else 'loan_status'
            
            X = df_scaled[feature_cols]
            y = df_scaled[target_for_selection]
            
            # Mutual information selection
            mi_features, mi_scores = self.select_features_mutual_info(X, y)
            
            # Correlation-based removal
            corr_features = self.select_features_correlation(df_scaled[feature_cols])
            
            # Model-based selection
            rf_features, rf_importances = self.select_features_model_based(X, y)
            
            # Combine selections (intersection of all methods)
            selected_features = list(set(mi_features) & set(corr_features) & set(rf_features))
            
            logging.info(f"Final selected features: {len(selected_features)}")
            
            # Visualize feature importance
            self.visualize_feature_importance(
                X.columns.tolist(),
                mi_scores,
                title="Mutual Information Feature Importance",
                save_path="outputs/figures/mutual_info_importance.png",
                top_n=20
            )
            
            self.visualize_feature_importance(
                X.columns.tolist(),
                rf_importances,
                title="Random Forest Feature Importance",
                save_path="outputs/figures/rf_importance.png",
                top_n=20
            )
        
        # Create final dataset with selected features
        final_cols = selected_features + available_targets
        df_final = df_scaled[final_cols]
        
        # Save engineered features
        if output_dir is None:
            output_dir = self.config['paths']['processed_data']
        
        output_path = os.path.join(output_dir, 'engineered_features.csv')
        save_dataframe(df_final, output_path)
        
        # Save feature names
        self.feature_names = selected_features
        
        # Save scaler and encoders
        save_model(self.scaler, os.path.join(self.config['paths']['models'], 'scaler.pkl'))
        save_model(self.label_encoders, os.path.join(self.config['paths']['models'], 'label_encoders.pkl'))
        save_model(selected_features, os.path.join(self.config['paths']['models'], 'selected_features.pkl'))
        
        logging.info("Feature engineering pipeline completed successfully")
        
        return {
            'engineered_data': df_final,
            'output_path': output_path,
            'selected_features': selected_features
        }


def main():
    """
    Main function to run feature engineering pipeline.
    """
    print("="*60)
    print("Credit Risk Model - Feature Engineering")
    print("="*60)
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Run feature engineering pipeline
    try:
        result = engineer.feature_engineering_pipeline()
        
        print("\n✓ Feature engineering completed successfully!")
        print(f"✓ Engineered data saved to: {result['output_path']}")
        print(f"✓ Total rows: {len(result['engineered_data'])}")
        print(f"✓ Selected features: {len(result['selected_features'])}")
        print(f"\n✓ Top 10 features:")
        for i, feat in enumerate(result['selected_features'][:10], 1):
            print(f"   {i}. {feat}")
        
    except Exception as e:
        print(f"\n✗ Feature engineering failed: {e}")
        logging.error(f"Feature engineering failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
