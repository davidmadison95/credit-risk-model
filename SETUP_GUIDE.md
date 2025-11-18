# Credit Risk Scoring Model - Setup Guide

## Quick Start

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Git (for version control)

---

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/davidmadison95/credit-risk-model.git
cd credit-risk-model
```

### 2. Create Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Dataset
Download one of these datasets:
- **Kaggle**: [Loan Prediction Dataset](https://www.kaggle.com/datasets)
- **Kaggle**: [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit)

Save the CSV file as:
```
data/raw/loan_data.csv
```

---

## Running the Project

### Step 1: Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_EDA.ipynb
```
This will:
- Load and inspect the dataset
- Visualize distributions and correlations
- Identify missing values and outliers
- Analyze class imbalance

### Step 2: Data Preprocessing
```bash
python src/data_preprocessing.py
```
This will:
- Clean the raw data
- Handle missing values
- Detect and treat outliers
- Save processed data to `data/processed/`

### Step 3: Feature Engineering
```bash
python src/feature_engineering.py
```
This will:
- Create derived features
- Encode categorical variables
- Scale numerical features
- Select important features
- Engineer risk tiers (Low, Medium, High)

### Step 4: Train Models
```bash
python src/train_models.py
```
This will:
- Train Logistic Regression
- Train Random Forest
- Train XGBoost
- Perform hyperparameter tuning
- Save models to `models/` directory

### Step 5: Evaluate Models
```bash
python src/evaluate.py
```
This will:
- Generate performance metrics
- Create confusion matrices
- Plot ROC curves
- Compare all models
- Save results to `outputs/reports/`

### Step 6: Generate SHAP Explanations
```bash
python src/explainability.py
```
This will:
- Calculate SHAP values
- Create summary plots
- Generate feature importance visualizations
- Create force plots for sample predictions
- Save figures to `outputs/figures/`

### Step 7: Launch Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
This will:
- Start the interactive web application
- Open in your browser (usually http://localhost:8501)
- Allow you to input borrower information
- Display risk predictions with explanations

---

## Project Workflow Diagram
```
┌─────────────────┐
│  Raw Data       │
│  (loan_data.csv)│
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
│ - Create features   │
│ - Encode categoricals│
│ - Scale features    │
│ - Select features   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Model Training     │
│ - Logistic Reg      │
│ - Random Forest     │
│ - XGBoost           │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Model Evaluation   │
│ - Metrics           │
│ - Confusion Matrix  │
│ - ROC Curves        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  SHAP Explainability│
│ - Summary plots     │
│ - Feature importance│
│ - Force plots       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Streamlit App      │
│ - User inputs       │
│ - Predictions       │
│ - Visualizations    │
└─────────────────────┘
```

---

## Troubleshooting

### Issue: Module not found
**Solution**: Make sure you're in the project root directory and virtual environment is activated
```bash
cd credit-risk-model
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: SHAP installation fails
**Solution**: Install with specific versions
```bash
pip install shap==0.42.1 --no-cache-dir
```

### Issue: Streamlit port already in use
**Solution**: Use a different port
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

### Issue: Dataset not found
**Solution**: Ensure the CSV file is in the correct location
```bash
# Check if file exists
ls data/raw/loan_data.csv

# If not, download and place it there
```

---

## Configuration

### Model Hyperparameters
Edit `config/model_config.yaml` to customize:
- Random seeds
- Train/test split ratio
- Model hyperparameters
- Feature selection thresholds

### Streamlit App Settings
Edit `app/streamlit_app.py` to customize:
- Page title and icon
- Color schemes
- Layout options
- Default input values

---

## Testing

Run unit tests (if implemented):
```bash
pytest tests/
```

---

## Deployment Options

### Local Deployment
```bash
streamlit run app/streamlit_app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

### Docker Deployment
```bash
# Create Dockerfile in project root
docker build -t credit-risk-model .
docker run -p 8501:8501 credit-risk-model
```

---

## Support

For issues or questions:
- Check the README.md
- Review the DATA_DICTIONARY.md
- Check GitHub Issues: [github.com/davidmadison95/credit-risk-model/issues](https://github.com/davidmadison95/credit-risk-model/issues)
- Contact: davidmadison95@yahoo.com

---

## Next Steps After Setup

1. ✅ Run EDA notebook to understand the data
2. ✅ Execute preprocessing pipeline
3. ✅ Train all three models
4. ✅ Compare model performance
5. ✅ Analyze SHAP explanations
6. ✅ Launch and test Streamlit app
7. ✅ Customize for your specific use case

---

## License
MIT License - Feel free to use and modify for your portfolio

---

## Author

**David Madison**  
Data Analyst | Machine Learning Enthusiast

📧 davidmadison95@yahoo.com  
💼 [LinkedIn](https://www.linkedin.com/in/davidmadison95/)  
🌐 [Portfolio](https://davidmadison95.github.io/Business-Portfolio/)  
📂 [GitHub](https://github.com/davidmadison95)