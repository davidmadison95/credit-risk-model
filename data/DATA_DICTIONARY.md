# Credit Risk Scoring Model - Data Dictionary

## Dataset Overview
This project uses credit/loan application data to predict borrower risk levels.
The model classifies applicants into three risk tiers: Low, Medium, and High.

---

## Feature Definitions

### Demographic Features
| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| `age` | Numeric | Age of the applicant in years | 25, 35, 55 |
| `person_home_ownership` | Categorical | Home ownership status | RENT, OWN, MORTGAGE, OTHER |
| `person_emp_length` | Numeric | Years of employment | 0.5, 5, 15 |

### Financial Features
| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| `person_income` | Numeric | Annual income in USD | 30000, 75000, 150000 |
| `loan_amnt` | Numeric | Loan amount requested | 5000, 15000, 35000 |
| `loan_int_rate` | Numeric | Loan interest rate (%) | 5.5, 10.2, 18.7 |
| `loan_percent_income` | Numeric | Loan as % of annual income | 0.15, 0.30, 0.55 |
| `cb_person_cred_hist_length` | Numeric | Credit history length (years) | 2, 10, 20 |

### Loan Characteristics
| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| `loan_intent` | Categorical | Purpose of the loan | PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION |
| `loan_grade` | Categorical | Loan grade assigned | A, B, C, D, E, F, G |
| `cb_person_default_on_file` | Categorical | Historical default flag | Y, N |

### Target Variable
| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `loan_status` | Binary → Multi-class | Loan default status (raw: 0=paid, 1=default) | **Engineered to:** Low Risk, Medium Risk, High Risk |

---

## Risk Tier Engineering Logic

The target variable `loan_status` is transformed into three risk tiers based on:

1. **Low Risk**: 
   - No historical defaults
   - Loan-to-income ratio < 0.25
   - Interest rate < 10%
   - Loan grade A or B

2. **Medium Risk**:
   - Minor risk factors present
   - Loan-to-income ratio 0.25-0.40
   - Interest rate 10-15%
   - Loan grade C, D, or E

3. **High Risk**:
   - Historical defaults OR
   - Loan-to-income ratio > 0.40 OR
   - Interest rate > 15% OR
   - Loan grade F or G

---

## Data Quality Notes

### Missing Values
Expected missing value patterns:
- `person_emp_length`: ~10% (unemployed/self-employed individuals)
- `loan_int_rate`: <5% (pending applications)

### Outliers
Features prone to outliers:
- `person_income`: Very high earners (>$500k)
- `loan_amnt`: Large business loans
- `cb_person_cred_hist_length`: Very long credit histories (>30 years)

### Class Imbalance
- Loan default is typically an imbalanced problem (5-20% default rate)
- SMOTE or class weights will be applied during training

---

## Feature Engineering Derived Features

| Feature | Formula | Purpose |
|---------|---------|---------|
| `debt_to_income_ratio` | `loan_amnt / person_income` | Debt burden assessment |
| `income_per_year_employed` | `person_income / max(person_emp_length, 1)` | Income stability |
| `age_group` | Binned age categories | Non-linear age effects |
| `high_interest_flag` | 1 if `loan_int_rate > 15%` | Risk indicator |

---

## Source Dataset
Recommended public datasets:
- **Kaggle**: "Loan Prediction Dataset" 
- **Kaggle**: "Give Me Some Credit"
- **UCI Machine Learning Repository**: Credit Approval datasets

## Usage
Place your downloaded CSV file in the `data/` folder as `loan_data.csv`
