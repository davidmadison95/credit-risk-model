# Credit Risk Scoring Model 💳

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)]()

A complete, production-quality machine learning project for predicting credit risk and classifying loan applicants into Low, Medium, or High risk tiers. Features three ML models, SHAP explainability, and an interactive Streamlit dashboard.

## 🎯 Project Overview

### Business Problem
Financial institutions need to assess credit risk accurately to:
- Minimize loan default rates
- Optimize lending decisions
- Maintain healthy loan portfolios
- Comply with fair lending regulations

### Solution
This project implements a complete ML pipeline that:
1. **Predicts risk tiers** for loan applicants (Low/Medium/High)
2. **Explains predictions** using SHAP values for transparency
3. **Compares multiple models** to find the best performer
4. **Provides an interactive dashboard** for real-time predictions

### Key Features
✅ **Three ML Models**: Logistic Regression, Random Forest, XGBoost  
✅ **Risk Tier Engineering**: Logic-based 3-tier classification  
✅ **SHAP Explainability**: Transparent, interpretable predictions  
✅ **Interactive Dashboard**: Streamlit web app with real-time predictions  
✅ **Production-Ready**: Modular code, comprehensive testing, full documentation  
✅ **End-to-End Pipeline**: From raw data to deployed model  

---

## 📊 Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** 🏆 | 0.8723 | 0.8645 | 0.8723 | **0.8678** | 0.9156 |
| Random Forest | 0.8567 | 0.8489 | 0.8567 | 0.8521 | 0.9012 |
| Logistic Regression | 0.8234 | 0.8156 | 0.8234 | 0.8189 | 0.8567 |

**🏆 Best Model**: XGBoost achieves the highest F1 score (0.8678) and ROC-AUC (0.9156)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager
- 4GB RAM minimum
- 2GB free disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/davidmadison95/credit-risk-model.git
cd credit-risk-model
```

2. **Create virtual environment**
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Generate sample data** (or use your own dataset)
```bash
python data/generate_sample_data.py
```

5. **Run the complete pipeline**
```bash
# Step 1: Preprocess data
python src/data_preprocessing.py

# Step 2: Engineer features
python src/feature_engineering.py

# Step 3: Train models
python src/train_models.py

# Step 4: Evaluate models
python src/evaluate.py

# Step 5: Generate SHAP explanations
python src/explainability.py
```

6. **Launch the dashboard**
```bash
streamlit run app/streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📁 Project Structure
```
credit-risk-model/
│
├── data/                           # Data storage
│   ├── raw/                        # Original datasets
│   ├── processed/                  # Cleaned and engineered data
│   ├── DATA_DICTIONARY.md          # Feature definitions
│   └── generate_sample_data.py     # Sample data generator
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   └── 02_Feature_Engineering.ipynb
│
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── utils.py                   # Helper functions
│   ├── data_preprocessing.py      # Data cleaning pipeline
│   ├── feature_engineering.py     # Feature creation & selection
│   ├── train_models.py            # Model training pipeline
│   ├── evaluate.py                # Model evaluation & metrics
│   └── explainability.py          # SHAP analysis
│
├── app/                           # Streamlit application
│   └── streamlit_app.py           # Interactive dashboard
│
├── models/                        # Saved trained models
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── feature_selector.pkl
│
├── outputs/                       # Generated outputs
│   ├── figures/                   # Visualizations
│   └── reports/                   # Performance reports
│
├── config/                        # Configuration files
│   └── model_config.yaml          # Model hyperparameters
│
├── tests/                         # Unit tests
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── SETUP_GUIDE.md                # Detailed setup instructions
└── PROJECT_STRUCTURE.md           # Architecture documentation
```

---

## 🔄 ML Pipeline
```
┌─────────────────┐
│  Raw Data       │
│  (CSV/Database) │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Data Preprocessing  │
│ - Clean data        │
│ - Handle missing    │
│ - Treat outliers    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Feature Engineering │
│ - Risk tiers        │
│ - Derived features  │
│ - Encode & scale    │
│ - Feature selection │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Model Training     │
│ - Logistic Reg      │
│ - Random Forest     │
│ - XGBoost           │
│ - Hyperparameter    │
│   tuning (optional) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Model Evaluation   │
│ - Confusion matrix  │
│ - ROC curves        │
│ - Feature importance│
│ - Comparison table  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  SHAP Explainability│
│ - Summary plots     │
│ - Force plots       │
│ - Feature contrib.  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Streamlit App      │
│ - User interface    │
│ - Real-time predict │
│ - SHAP explanation  │
└─────────────────────┘
```

---

## 📚 Data Dictionary

### Target Variable
- **risk_tier**: Engineered risk classification (Low, Medium, High)
- **loan_status**: Original default indicator (0=Paid, 1=Default)

### Input Features

#### Demographics
| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| `age` | Borrower's age in years | Numeric | 18-80 |
| `person_emp_length` | Years of employment | Numeric | 0-40 |
| `person_home_ownership` | Home ownership status | Categorical | RENT, MORTGAGE, OWN, OTHER |

#### Financial
| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| `person_income` | Annual income (USD) | Numeric | 15,000-500,000 |
| `loan_amnt` | Requested loan amount | Numeric | 1,000-40,000 |
| `loan_int_rate` | Loan interest rate (%) | Numeric | 5-30 |
| `loan_percent_income` | Loan as % of income | Numeric | 0.01-0.80 |

#### Credit History
| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| `cb_person_cred_hist_length` | Credit history length (years) | Numeric | 0-30 |
| `cb_person_default_on_file` | Previous default flag | Categorical | Y, N |

#### Loan Characteristics
| Feature | Description | Type | Values |
|---------|-------------|------|--------|
| `loan_intent` | Purpose of loan | Categorical | PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION |
| `loan_grade` | Assigned loan grade | Categorical | A, B, C, D, E, F, G |

### Derived Features
- `debt_to_income_ratio`: Loan amount / Annual income
- `income_per_year_employed`: Income / Employment length
- `high_interest_flag`: Binary (1 if rate > 15%)
- `short_credit_history`: Binary (1 if history < 3 years)
- `employment_stable`: Binary (1 if employed >= 2 years)

---

## 🎯 Risk Tier Logic

### Low Risk
- ✅ No previous defaults
- ✅ Debt-to-income ratio < 25%
- ✅ Interest rate < 10%
- ✅ Loan grades A or B

### Medium Risk
- ⚠️ Moderate risk factors
- ⚠️ Debt-to-income ratio 25-40%
- ⚠️ Interest rate 10-15%
- ⚠️ Loan grades C, D, or E

### High Risk
- 🚨 Previous defaults OR
- 🚨 Debt-to-income ratio > 40% OR
- 🚨 Interest rate > 15% OR
- 🚨 Loan grades F or G

---

## 🔍 Model Explainability

This project uses **SHAP (SHapley Additive exPlanations)** for model interpretation:

### SHAP Features
- **Summary Plots**: Overall feature importance across all predictions
- **Bar Plots**: Average absolute SHAP values per feature
- **Waterfall Plots**: Step-by-step prediction explanation for individual cases
- **Force Plots**: Visual representation of feature contributions

### Example Interpretation
```
Top Contributing Features:
1. loan_int_rate: +0.234 (increases risk)
2. loan_percent_income: +0.189 (increases risk)
3. person_income: -0.156 (decreases risk)
4. cb_person_default_on_file_Y: +0.142 (increases risk)
5. cb_person_cred_hist_length: -0.098 (decreases risk)
```

---

## 🖥️ Streamlit Dashboard

### Features
1. **Interactive Prediction**
   - Input borrower information via sidebar
   - Select model (Logistic Regression, Random Forest, XGBoost)
   - Get instant risk prediction

2. **Risk Visualization**
   - Color-coded risk tiers (Green/Yellow/Red)
   - Probability distribution charts
   - Confidence indicators

3. **SHAP Explanation**
   - Feature contribution analysis
   - Interactive visualizations
   - Transparent decision-making

4. **Model Comparison**
   - Performance metrics table
   - Side-by-side comparison charts
   - Best model identification

---

## 📈 Model Training Details

### Hyperparameters

**Logistic Regression**
```yaml
penalty: l2
C: 1.0
max_iter: 1000
solver: lbfgs
class_weight: balanced
```

**Random Forest**
```yaml
n_estimators: 200
max_depth: 20
min_samples_split: 5
min_samples_leaf: 2
max_features: sqrt
class_weight: balanced
```

**XGBoost**
```yaml
n_estimators: 200
max_depth: 6
learning_rate: 0.1
subsample: 0.8
colsample_bytree: 0.8
scale_pos_weight: 1
```

### Training Process
1. **Data Split**: 80% train, 20% test
2. **Class Imbalance**: SMOTE oversampling applied
3. **Validation**: 5-fold cross-validation
4. **Scoring Metric**: ROC-AUC
5. **Training Time**: 2-5 minutes for all models

---

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/
```

### Run Individual Components
```bash
# Test data preprocessing
python src/data_preprocessing.py

# Test feature engineering
python src/feature_engineering.py

# Test model training
python src/train_models.py

# Test evaluation
python src/evaluate.py

# Test SHAP explainability
python src/explainability.py
```

---

## 🔧 Configuration

All model parameters are configurable via `config/model_config.yaml`:
```yaml
random_state: 42

preprocessing:
  test_size: 0.2
  handle_missing:
    strategy: "median"
  outlier_detection:
    method: "iqr"
    threshold: 3
  class_imbalance:
    apply_smote: true

feature_engineering:
  scaling:
    method: "standard"
  feature_selection:
    n_features: 15

models:
  xgboost:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.1
```

---

## 🚢 Deployment

### Local Deployment
```bash
streamlit run app/streamlit_app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

### Docker Deployment
```bash
# Build image
docker build -t credit-risk-model .

# Run container
docker run -p 8501:8501 credit-risk-model
```

---

## 🔮 Future Improvements

- [ ] Add more ML models (LightGBM, CatBoost, Neural Networks)
- [ ] Implement real-time model monitoring
- [ ] Add A/B testing framework
- [ ] Create REST API for predictions
- [ ] Add automated retraining pipeline
- [ ] Implement fairness and bias testing
- [ ] Add database integration for production data
- [ ] Create batch prediction capability
- [ ] Add model versioning with MLflow
- [ ] Implement automated alerting for model drift

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**David Madison**  
📧 Email: davidmadison95@yahoo.com  
💼 LinkedIn: [linkedin.com/in/davidmadison95](https://www.linkedin.com/in/davidmadison95/)  
🌐 Portfolio: [davidmadison95.github.io/Business-Portfolio](https://davidmadison95.github.io/Business-Portfolio/)  
📂 GitHub: [@davidmadison95](https://github.com/davidmadison95)

**Project Repository**: [github.com/davidmadison95/credit-risk-model](https://github.com/davidmadison95/credit-risk-model)

---

## 🙏 Acknowledgments

- Dataset inspired by Kaggle's "Give Me Some Credit" and "Loan Prediction" datasets
- SHAP library by Scott Lundberg
- Streamlit for the amazing dashboard framework
- scikit-learn and XGBoost communities

---

## 📚 References

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
3. Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ by David Madison  
*Data Analyst | Machine Learning Enthusiast*

</div>