# Credit Risk Scoring Model - Project Structure

```
credit-risk-model/
│
├── data/                          # Data storage
│   ├── raw/                       # Original datasets
│   │   └── loan_data.csv
│   ├── processed/                 # Cleaned and processed data
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│   │   └── feature_engineered.csv
│   └── DATA_DICTIONARY.md         # Feature definitions
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Experiments.ipynb
│
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data cleaning and validation
│   ├── feature_engineering.py    # Feature creation and selection
│   ├── train_models.py           # Model training pipeline
│   ├── evaluate.py               # Model evaluation metrics
│   ├── explainability.py         # SHAP analysis
│   └── utils.py                  # Helper functions
│
├── app/                          # Streamlit application
│   ├── streamlit_app.py          # Main dashboard
│   ├── components/               # UI components
│   │   ├── prediction.py
│   │   ├── visualization.py
│   │   └── model_comparison.py
│   └── assets/                   # Images, CSS
│
├── models/                       # Saved trained models
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── feature_selector.pkl
│
├── outputs/                      # Generated outputs
│   ├── figures/                  # EDA and SHAP plots
│   ├── reports/                  # Model performance reports
│   └── logs/                     # Training logs
│
├── tests/                        # Unit tests (optional)
│   ├── test_preprocessing.py
│   └── test_models.py
│
├── config/                       # Configuration files
│   └── model_config.yaml
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── .gitignore                    # Git ignore rules
└── setup.py                      # Package setup (optional)
```

## File Purposes

### Data (`data/`)
- **raw/**: Store original, unmodified datasets
- **processed/**: Store cleaned, transformed datasets ready for modeling
- **DATA_DICTIONARY.md**: Reference for all features and their meanings

### Notebooks (`notebooks/`)
- **01_EDA.ipynb**: Initial data exploration, visualization, and insights
- **02_Feature_Engineering.ipynb**: Feature creation and selection experiments
- **03_Model_Experiments.ipynb**: Model comparison and hyperparameter tuning

### Source Code (`src/`)
- **data_preprocessing.py**: Functions for data cleaning, missing value handling, outlier detection
- **feature_engineering.py**: Feature transformation, encoding, scaling, selection
- **train_models.py**: Training pipeline for all three models
- **evaluate.py**: Performance metrics, confusion matrices, ROC curves
- **explainability.py**: SHAP value generation and visualization
- **utils.py**: Shared utility functions (logging, file I/O, etc.)

### Application (`app/`)
- **streamlit_app.py**: Main Streamlit dashboard with user interface
- **components/**: Modular UI components for clean code organization
- **assets/**: Static files for styling and branding

### Models (`models/`)
- Stores serialized trained models using joblib/pickle
- Includes preprocessing artifacts (scalers, encoders, selectors)

### Outputs (`outputs/`)
- **figures/**: All generated plots and visualizations
- **reports/**: Performance summaries, comparison tables
- **logs/**: Training logs and experiment tracking

## Design Principles

1. **Modularity**: Each script has a single, clear responsibility
2. **Reproducibility**: All random seeds are fixed
3. **Scalability**: Code can handle different datasets with minimal changes
4. **Maintainability**: Clear naming conventions and comprehensive documentation
5. **Production-Ready**: Follows software engineering best practices

## Next Steps
1. Download dataset to `data/raw/`
2. Run notebooks in order (01 → 02 → 03)
3. Execute training scripts in `src/`
4. Launch Streamlit app
