"""
Sample Data Generator for Credit Risk Model

This script generates synthetic loan data for testing the credit risk model
when real datasets are not available.

The generated data mimics real loan datasets with:
- Demographic features
- Financial features
- Loan characteristics
- Target variable (loan default status)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)


def generate_sample_data(n_samples: int = 10000, save_path: str = None) -> pd.DataFrame:
    """
    Generate synthetic loan dataset.
    
    Args:
        n_samples: Number of samples to generate
        save_path: Path to save the CSV file (optional)
        
    Returns:
        DataFrame with synthetic loan data
    """
    print(f"Generating {n_samples:,} synthetic loan records...")
    
    # Generate demographic features
    age = np.random.normal(loc=35, scale=12, size=n_samples).clip(18, 80).astype(int)
    person_emp_length = np.random.exponential(scale=5, size=n_samples).clip(0, 40)
    
    # Home ownership distribution
    home_ownership = np.random.choice(
        ['RENT', 'MORTGAGE', 'OWN', 'OTHER'],
        size=n_samples,
        p=[0.35, 0.40, 0.20, 0.05]
    )
    
    # Generate financial features
    person_income = np.random.lognormal(mean=10.8, sigma=0.6, size=n_samples).clip(15000, 500000)
    
    # Loan amount (correlated with income)
    loan_amnt = (person_income * np.random.uniform(0.1, 0.5, n_samples)).clip(1000, 40000)
    
    # Loan percent of income
    loan_percent_income = (loan_amnt / person_income).clip(0.01, 0.80)
    
    # Credit history length
    cb_person_cred_hist_length = np.random.gamma(shape=3, scale=3, size=n_samples).clip(0, 30)
    
    # Loan intent
    loan_intent = np.random.choice(
        ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
        size=n_samples,
        p=[0.25, 0.15, 0.10, 0.10, 0.20, 0.20]
    )
    
    # Loan grade (A-G)
    loan_grade = np.random.choice(
        ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        size=n_samples,
        p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.03, 0.02]
    )
    
    # Interest rate (influenced by loan grade)
    grade_interest_map = {'A': (5, 8), 'B': (7, 10), 'C': (9, 13), 
                         'D': (12, 16), 'E': (15, 20), 'F': (18, 24), 'G': (22, 28)}
    
    loan_int_rate = np.array([
        np.random.uniform(*grade_interest_map[grade]) 
        for grade in loan_grade
    ])
    
    # Historical default
    cb_person_default_on_file = np.random.choice(
        ['Y', 'N'],
        size=n_samples,
        p=[0.15, 0.85]
    )
    
    # Generate target variable (loan_status)
    # 0 = Paid, 1 = Default
    # Default probability based on multiple factors
    default_prob = np.zeros(n_samples)
    
    # Increase default probability based on risk factors
    default_prob += (loan_percent_income > 0.35) * 0.20  # High debt-to-income
    default_prob += (loan_int_rate > 15) * 0.15  # High interest rate
    default_prob += (cb_person_default_on_file == 'Y') * 0.30  # Previous default
    default_prob += (loan_grade >= 'E') * 0.15  # Poor loan grade
    default_prob += (person_emp_length < 1) * 0.10  # Short employment
    default_prob += (cb_person_cred_hist_length < 2) * 0.10  # Short credit history
    
    # Add some randomness
    default_prob += np.random.uniform(0, 0.1, n_samples)
    
    # Cap probability between 0 and 1
    default_prob = default_prob.clip(0, 0.95)
    
    # Generate binary outcome
    loan_status = (np.random.random(n_samples) < default_prob).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'person_income': person_income.round(2),
        'person_home_ownership': home_ownership,
        'person_emp_length': person_emp_length.round(2),
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amnt': loan_amnt.round(2),
        'loan_int_rate': loan_int_rate.round(2),
        'loan_percent_income': loan_percent_income.round(4),
        'cb_person_default_on_file': cb_person_default_on_file,
        'cb_person_cred_hist_length': cb_person_cred_hist_length.round(2),
        'loan_status': loan_status
    })
    
    # Add some missing values randomly (5-10% for realism)
    missing_cols = ['person_emp_length', 'loan_int_rate', 'cb_person_cred_hist_length']
    for col in missing_cols:
        if col in df.columns:
            missing_mask = np.random.random(n_samples) < 0.08
            df.loc[missing_mask, col] = np.nan
    
    # Add a few outliers
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    df.loc[outlier_indices, 'person_income'] *= np.random.uniform(2, 5, len(outlier_indices))
    
    print(f"✓ Generated dataset with {len(df):,} rows and {len(df.columns)} columns")
    print(f"✓ Default rate: {df['loan_status'].mean()*100:.2f}%")
    print(f"✓ Missing values: {df.isnull().sum().sum()} cells")
    
    # Save to CSV if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✓ Data saved to: {save_path}")
    
    return df


def generate_and_describe(n_samples: int = 10000):
    """
    Generate sample data and print descriptive statistics.
    
    Args:
        n_samples: Number of samples to generate
    """
    df = generate_sample_data(n_samples, save_path='data/raw/loan_data.csv')
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    print("\n--- Numeric Features ---")
    print(df.describe())
    
    print("\n--- Categorical Features ---")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
    
    print("\n--- Target Variable ---")
    print(f"Loan Status Distribution:")
    print(df['loan_status'].value_counts())
    print(f"Default Rate: {df['loan_status'].mean()*100:.2f}%")
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(missing)
    else:
        print("No missing values")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("="*60)
    print("Credit Risk Model - Sample Data Generator")
    print("="*60)
    print("\nThis script generates synthetic loan data for testing.")
    print("Real datasets can be downloaded from:")
    print("  - Kaggle: Loan Prediction Dataset")
    print("  - Kaggle: Give Me Some Credit")
    print("\n" + "="*60 + "\n")
    
    # Generate data
    generate_and_describe(n_samples=10000)
    
    print("\n✓ Sample data generation complete!")
    print("✓ You can now run the EDA notebook or preprocessing pipeline.")
