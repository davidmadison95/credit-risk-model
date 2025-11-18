# Credit Risk Scoring Model

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready machine learning system for predicting credit risk and loan default probability. This project demonstrates a complete end-to-end ML pipeline from data preprocessing through model deployment, achieving 87% F1 score and 92% ROC-AUC with XGBoost.

## 🌐 Live Demo

**Try the interactive dashboard:** [https://davidmadison-credit-risk.streamlit.app](https://davidmadison-credit-risk.streamlit.app)

![Credit Risk Dashboard](assets/images/credit-risk-dashboard.png)

> 🚧 **Note:** The deployed application displays demo mode. Follow the [Setup Guide](#installation) below to train models and run locally with full functionality.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Performance](#-performance)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [Model Details](#-model-details)
- [Dashboard Features](#-dashboard-features)
- [Results & Visualizations](#-results--visualizations)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Project Overview

### Business Problem

Financial institutions need accurate credit risk assessment to make informed lending decisions. This project builds a multi-model machine learning system that predicts loan default probability and classifies borrowers into risk tiers (Low, Medium, High).

### Solution

A comprehensive ML pipeline that:
- Processes loan applications and borrower financial data
- Engineers 20+ predictive features from raw data
- Trains and compares three classification models
- Provides transparent predictions using SHAP explanations
- Deploys an interactive dashboard for risk assessment

### Target Audience

- **Financial Analysts**: Risk assessment and portfolio management
- **Data Scientists**: ML pipeline implementation reference
- **Loan Officers**: Decision support tool
- **Hiring Managers**: Portfolio demonstration of end-to-end ML capabilities

---

## ✨ Key Features

### Three ML Models
- **Logistic Regression**: Fast, interpretable baseline model
- **Random Forest**: Robust ensemble with feature importance
- **XGBoost**: Advanced gradient boosting (best performance: 87% F1, 92% ROC-AUC)

### SHAP Explainability
- Transparent, interpretable predictions using SHAP values
- Feature contribution analysis for each prediction
- Waterfall and force plots for model decisions

### Interactive Dashboard
- Real-time risk assessment with Streamlit interface
- Model comparison and performance metrics
- Visual explanations with interactive charts (Plotly)
- User-friendly input forms for borrower information

### End-to-End Pipeline
- Data preprocessing and cleaning
- Advanced feature engineering
- Model training with hyperparameter tuning
- Comprehensive evaluation with multiple metrics
- Cloud deployment (Streamlit Community Cloud)

---

## 📊 Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** 🏆 | 0.872 | 0.865 | 0.872 | **0.868** | **0.916** |
| Random Forest | 0.857 | 0.849 | 0.857 | 0.852 | 0.901 |
| Logistic Regression | 0.823 | 0.816 | 0.823 | 0.819 | 0.857 |

**Key Achievements:**
- ✅ **87% F1 Score** - Excellent balance of precision and recall
- ✅ **92% ROC-AUC** - Strong discrimination between risk classes
- ✅ **<100ms Prediction Time** - Real-time inference capability
- ✅ **Production Deployed** - Live dashboard on Streamlit Cloud

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                          Data Sources                                │
│              (Loan Applications, Borrower Profiles)                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Data Preprocessing                               │
│  • Handle missing values  • Encode categoricals  • Detect outliers  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Feature Engineering                               │
│  • Debt-to-income ratio  • Income per employment year               │
│  • High interest flags   • Credit history indicators                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Model Training                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  Logistic    │  │   Random     │  │   XGBoost    │             │
│  │  Regression  │  │   Forest     │  │   Classifier │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Model Evaluation                                  │
│  • Confusion matrices  • ROC curves  • Feature importance           │
│  • SHAP explanations   • Cross-validation  • Metrics comparison     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Deployment (Streamlit)                            │
│  • Interactive dashboard  • Real-time predictions  • SHAP viz       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/davidmadison95/credit-risk-model.git
cd credit-risk-model
```

2. **Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Generate sample data**
```bash
python data/generate_sample_data.py
```

This creates synthetic loan data with realistic distributions for demonstration purposes.

---

## 💻 Usage

### Complete Pipeline Execution

Run the full ML pipeline from data preprocessing through model training:
```bash
# Step 1: Preprocess raw data
python src/data_preprocessing.py

# Step 2: Engineer features
python src/feature_engineering.py

# Step 3: Train all models
python src/train_models.py

# Step 4: Evaluate models
python src/evaluate.py

# Step 5: Generate SHAP explanations
python src/explainability.py
```

### Launch Interactive Dashboard
```bash
streamlit run app/streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Quick Start (Single Command)
```bash
# Run complete pipeline
python src/data_preprocessing.py && \
python src/feature_engineering.py && \
python src/train_models.py && \
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure
```
credit-risk-model/
│
├── app/                              # Streamlit dashboard application
│   └── streamlit_app.py             # Main dashboard code
│
├── assets/                           # Visual assets and screenshots
│   └── images/
│       └── credit-risk-dashboard.png
│
├── config/                           # Configuration files
│   └── model_config.yaml            # Model hyperparameters and settings
│
├── data/                            # Data directory
│   ├── raw/                         # Raw data files (generated)
│   ├── processed/                   # Processed datasets
│   └── generate_sample_data.py      # Sample data generator
│
├── models/                          # Trained model artifacts
│   ├── logistic_regression.pkl      # Logistic Regression model
│   ├── random_forest.pkl            # Random Forest model
│   ├── xgboost.pkl                  # XGBoost model
│   ├── scaler.pkl                   # Feature scaler
│   └── selected_features.pkl        # Feature list
│
├── notebooks/                       # Jupyter notebooks
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   └── 02_Feature_Engineering.ipynb # Feature engineering experiments
│
├── outputs/                         # Generated outputs
│   ├── figures/                     # Visualization plots
│   └── reports/                     # Performance reports
│       └── model_comparison.csv     # Model metrics comparison
│
├── src/                             # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py        # Data cleaning and preprocessing
│   ├── feature_engineering.py       # Feature creation and selection
│   ├── train_models.py              # Model training pipeline
│   ├── evaluate.py                  # Model evaluation
│   ├── explainability.py            # SHAP analysis
│   └── utils.py                     # Helper functions
│
├── tests/                           # Unit tests
│   └── test_utils.py
│
├── .gitignore                       # Git ignore file
├── LICENSE                          # MIT License
├── PROJECT_COMPLETE.md              # Project completion checklist
├── PROJECT_STRUCTURE.md             # Detailed structure documentation
├── QUICK_REFERENCE.md               # Quick reference guide
├── README.md                        # This file
├── SETUP_GUIDE.md                   # Detailed setup instructions
├── requirements.txt                 # Python dependencies
└── setup_project.py                 # Project setup script
```

---

## 🔄 ML Pipeline

### 1. Data Preprocessing
- **Missing Value Imputation**: Median/mode imputation for numerical/categorical features
- **Outlier Detection**: IQR-based outlier identification and capping
- **Categorical Encoding**: One-hot encoding for nominal features
- **Data Validation**: Schema validation and consistency checks

### 2. Feature Engineering
- **Derived Features**:
  - `debt_to_income_ratio`: Loan amount / Annual income
  - `income_per_year_employed`: Income / (Employment length + 1)
  - `high_interest_flag`: Interest rate > 15% threshold
  - `short_credit_history`: Credit history < 3 years
  - `employment_stable`: Employment length ≥ 2 years

- **Feature Selection**: SelectKBest with chi-squared test (top 20 features)

### 3. Model Training
- **Train/Test Split**: 80/20 with stratification
- **Class Imbalance**: SMOTE oversampling for minority classes
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Model Serialization**: Pickle format for model persistence

### 4. Model Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **Visualizations**: Confusion matrices, ROC curves, precision-recall curves
- **Cross-Validation**: 5-fold stratified CV for robust performance estimates

### 5. Model Explainability
- **SHAP Analysis**: TreeExplainer for tree-based models
- **Feature Importance**: Global and local feature contributions
- **Visualization**: Waterfall plots, force plots, summary plots

---

## 🤖 Model Details

### XGBoost Classifier (Best Performance)

**Hyperparameters:**
```yaml
n_estimators: 200
max_depth: 6
learning_rate: 0.1
subsample: 0.8
colsample_bytree: 0.8
objective: multi:softmax
num_class: 3
```

**Why XGBoost Performs Best:**
- ✅ Handles non-linear relationships effectively
- ✅ Built-in regularization prevents overfitting
- ✅ Robust to outliers and missing values
- ✅ Ensemble of weak learners improves generalization

### Random Forest Classifier

**Hyperparameters:**
```yaml
n_estimators: 100
max_depth: 10
min_samples_split: 5
min_samples_leaf: 2
```

**Advantages:**
- ✅ Feature importance for interpretability
- ✅ Robust to outliers
- ✅ Minimal hyperparameter tuning required

### Logistic Regression

**Hyperparameters:**
```yaml
penalty: l2
C: 1.0
max_iter: 1000
solver: lbfgs
```

**Use Cases:**
- ✅ Fast training and inference
- ✅ Highly interpretable coefficients
- ✅ Good baseline model

---

## 📱 Dashboard Features

### 1. Prediction Tab
- **Input Form**: 11 borrower attributes including:
  - Demographics (age, employment length)
  - Financial (income, loan amount, interest rate)
  - Credit history (length, previous defaults)
  - Loan characteristics (purpose, grade)

- **Model Selection**: Choose between Logistic Regression, Random Forest, or XGBoost

- **Risk Assessment**: Instant classification into Low/Medium/High risk tiers

- **Probability Distribution**: Visual breakdown of risk probabilities

### 2. SHAP Explanations
- **Top Contributing Features**: Bar chart of feature impacts
- **Feature Value Display**: Actual values for transparency
- **Interpretation Guide**: How to read SHAP values

### 3. Model Comparison
- **Performance Metrics Table**: Side-by-side comparison
- **Interactive Charts**: Grouped bar charts for visual comparison
- **Best Model Recommendation**: Automatic selection based on F1 score

---

## 📈 Results & Visualizations

### Model Performance Comparison

The project includes comprehensive evaluation visualizations:

- **ROC Curves**: Compare discrimination ability across models
- **Confusion Matrices**: Detailed classification breakdowns
- **Feature Importance**: Top predictive features for each model
- **SHAP Summary Plots**: Global feature impact analysis
- **Learning Curves**: Training vs validation performance

### Key Findings

**Most Important Features (XGBoost):**
1. `loan_int_rate` - Interest rate (strongest predictor)
2. `loan_percent_income` - Debt-to-income ratio
3. `person_income` - Annual income
4. `loan_grade` - Assigned loan grade
5. `cb_person_default_on_file` - Previous default history

**Risk Classification Distribution:**
- Low Risk: 35% of borrowers
- Medium Risk: 45% of borrowers
- High Risk: 20% of borrowers

---

## 🚀 Future Enhancements

### Technical Improvements
- [ ] Implement MLflow for experiment tracking
- [ ] Add CI/CD pipeline with GitHub Actions
- [ ] Deploy with Docker containerization
- [ ] Set up model monitoring and drift detection
- [ ] Add automated retraining pipeline

### Feature Additions
- [ ] Add ensemble model voting classifier
- [ ] Implement neural network model
- [ ] Add time-series analysis for payment history
- [ ] Include external credit bureau data
- [ ] Add adversarial validation

### Dashboard Enhancements
- [ ] Add user authentication
- [ ] Implement batch prediction upload
- [ ] Add historical prediction tracking
- [ ] Include A/B testing framework
- [ ] Add PDF report generation

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**David Madison**

- **Email**: davidmadison95@yahoo.com
- **LinkedIn**: [linkedin.com/in/davidmadison95](https://www.linkedin.com/in/davidmadison95/)
- **Portfolio**: [davidmadison95.github.io/Business-Portfolio](https://davidmadison95.github.io/Business-Portfolio/)
- **GitHub**: [github.com/davidmadison95](https://github.com/davidmadison95)

---

## 🙏 Acknowledgments

- **scikit-learn**: Machine learning library
- **XGBoost**: Gradient boosting framework
- **SHAP**: Model explainability library
- **Streamlit**: Dashboard framework
- **Plotly**: Interactive visualizations

---

## 📊 Project Stats

- **Lines of Code**: 5,000+
- **Models Trained**: 3
- **Features Engineered**: 20+
- **Data Points**: 10,000+
- **Deployment**: Streamlit Community Cloud
- **Development Time**: 4 weeks

---

<div align="center">

**⭐ If you find this project helpful, please give it a star!**

[![GitHub stars](https://img.shields.io/github/stars/davidmadison95/credit-risk-model?style=social)](https://github.com/davidmadison95/credit-risk-model)

</div>
