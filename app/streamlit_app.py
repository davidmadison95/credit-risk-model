"""
Streamlit Dashboard for Credit Risk Scoring Model

Interactive web application for:
- Credit risk prediction
- Model explanation (SHAP)
- Model comparison
- Feature input
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any
import shap
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_model, load_config

# Page configuration
st.set_page_config(
    page_title="Credit Risk Scoring Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-low {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .risk-medium {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_and_artifacts():
    """Load all trained models and preprocessing artifacts."""
    try:
        models = {
            'Logistic Regression': load_model('models/logistic_regression.pkl'),
            'Random Forest': load_model('models/random_forest.pkl'),
            'XGBoost': load_model('models/xgboost.pkl')
        }
        scaler = load_model('models/scaler.pkl')
        selected_features = load_model('models/selected_features.pkl')
        config = load_config('config/model_config.yaml')
        
        return models, scaler, selected_features, config
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None


@st.cache_resource
def load_comparison_data():
    """Load model comparison results."""
    try:
        comparison_df = pd.read_csv('outputs/reports/model_comparison.csv', index_col=0)
        return comparison_df
    except:
        return None


def create_input_features():
    """Create sidebar inputs for borrower features."""
    st.sidebar.header("📋 Borrower Information")
    
    # Demographic features
    st.sidebar.subheader("Demographics")
    age = st.sidebar.slider("Age", 18, 80, 35)
    person_emp_length = st.sidebar.slider("Years Employed", 0.0, 40.0, 5.0, 0.5)
    person_home_ownership = st.sidebar.selectbox(
        "Home Ownership",
        ["RENT", "MORTGAGE", "OWN", "OTHER"]
    )
    
    # Financial features
    st.sidebar.subheader("Financial Information")
    person_income = st.sidebar.number_input(
        "Annual Income ($)", 
        min_value=15000, max_value=500000, value=60000, step=5000
    )
    
    loan_amnt = st.sidebar.number_input(
        "Loan Amount ($)",
        min_value=1000, max_value=40000, value=10000, step=1000
    )
    
    loan_int_rate = st.sidebar.slider(
        "Interest Rate (%)",
        5.0, 30.0, 12.0, 0.5
    )
    
    cb_person_cred_hist_length = st.sidebar.slider(
        "Credit History Length (years)",
        0.0, 30.0, 5.0, 0.5
    )
    
    # Loan characteristics
    st.sidebar.subheader("Loan Details")
    loan_intent = st.sidebar.selectbox(
        "Loan Purpose",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    )
    
    loan_grade = st.sidebar.selectbox(
        "Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"]
    )
    
    cb_person_default_on_file = st.sidebar.selectbox(
        "Previous Default",
        ["N", "Y"]
    )
    
    # Calculate derived features
    loan_percent_income = loan_amnt / person_income
    
    # Create feature dictionary
    features = {
        'age': age,
        'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'person_emp_length': person_emp_length,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_default_on_file': cb_person_default_on_file,
        'cb_person_cred_hist_length': cb_person_cred_hist_length
    }
    
    return features


def preprocess_features(features: Dict, scaler, selected_features):
    """Preprocess input features for prediction."""
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Add derived features (matching feature engineering)
    df['debt_to_income_ratio'] = df['loan_amnt'] / df['person_income']
    df['income_per_year_employed'] = df['person_income'] / (df['person_emp_length'] + 1)
    df['high_interest_flag'] = (df['loan_int_rate'] > 15).astype(int)
    df['short_credit_history'] = (df['cb_person_cred_hist_length'] < 3).astype(int)
    df['employment_stable'] = (df['person_emp_length'] >= 2).astype(int)
    
    # Encode categorical features
    df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent', 
                                     'loan_grade', 'cb_person_default_on_file'], 
                       drop_first=True, dtype=int)
    
    # Ensure all required columns are present
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the features used during training
    df = df[selected_features]
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    return df_scaled


def predict_risk(features_scaled, models, model_name):
    """Make prediction using selected model."""
    model = models[model_name]
    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[0]
    
    return prediction[0], probabilities


def display_risk_prediction(prediction, probabilities):
    """Display risk prediction with styling."""
    risk_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    risk_colors = {0: "risk-low", 1: "risk-medium", 2: "risk-high"}
    risk_icons = {0: "✅", 1: "⚠️", 2: "🚨"}
    
    risk_tier = risk_mapping.get(prediction, "Unknown")
    risk_class = risk_colors.get(prediction, "")
    risk_icon = risk_icons.get(prediction, "")
    
    st.markdown(f"""
    <div class="{risk_class}">
        <h2>{risk_icon} Predicted Risk: {risk_tier}</h2>
        <p>The model predicts this borrower falls into the <strong>{risk_tier}</strong> category.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display probabilities
    st.subheader("Risk Probabilities")
    
    prob_df = pd.DataFrame({
        'Risk Tier': ['Low Risk', 'Medium Risk', 'High Risk'],
        'Probability': probabilities
    })
    
    fig = go.Figure(data=[
        go.Bar(
            x=prob_df['Risk Tier'],
            y=prob_df['Probability'],
            marker_color=['green', 'orange', 'red'],
            text=[f'{p:.1%}' for p in prob_df['Probability']],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Probability Distribution",
        xaxis_title="Risk Tier",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_shap_explanation(features_scaled, model, feature_names, model_name):
    """Display SHAP explanation for the prediction."""
    st.subheader("🔍 Model Explanation (SHAP)")
    
    try:
        # Create SHAP explainer
        if 'Logistic' in model_name:
            # For linear models, we need a background dataset
            st.info("SHAP explanation for Logistic Regression requires background data. Using TreeExplainer for tree-based models only.")
            return
        else:
            explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(features_scaled)
        
        # For multi-class, use the predicted class
        if isinstance(shap_values, list):
            prediction = model.predict(features_scaled)[0]
            shap_values = shap_values[prediction]
        
        # Create a DataFrame for display
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': features_scaled[0],
            'SHAP Value': shap_values[0]
        }).sort_values(by='SHAP Value', key=abs, ascending=False).head(10)
        
        # Display top contributing features
        st.write("**Top 10 Contributing Features:**")
        
        fig = go.Figure(data=[
            go.Bar(
                y=shap_df['Feature'],
                x=shap_df['SHAP Value'],
                orientation='h',
                marker=dict(
                    color=shap_df['SHAP Value'],
                    colorscale='RdYlGn',
                    reversescale=True
                ),
                text=[f'{v:.3f}' for v in shap_df['SHAP Value']],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Feature Contributions to Prediction",
            xaxis_title="SHAP Value (Impact on Prediction)",
            yaxis_title="Feature",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **How to interpret:**
        - Positive SHAP values (red) push the prediction toward higher risk
        - Negative SHAP values (green) push the prediction toward lower risk
        - Larger absolute values indicate stronger influence
        """)
        
    except Exception as e:
        st.error(f"Could not generate SHAP explanation: {e}")


def display_model_comparison(comparison_df):
    """Display model comparison section."""
    st.header("📊 Model Comparison")
    
    if comparison_df is None:
        st.warning("Model comparison data not available. Train models first.")
        return
    
    # Display metrics table
    st.subheader("Performance Metrics")
    st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
    
    # Visualize comparison
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
    
    fig = go.Figure()
    
    for metric in available_metrics:
        fig.add_trace(go.Bar(
            name=metric.upper(),
            x=comparison_df.index,
            y=comparison_df[metric],
            text=[f'{v:.3f}' for v in comparison_df[metric]],
            textposition='outside'
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        height=500,
        yaxis=dict(range=[0, 1.1])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model
    best_model = comparison_df['f1'].idxmax()
    best_f1 = comparison_df.loc[best_model, 'f1']
    
    st.success(f"🏆 **Best Model:** {best_model} (F1 Score: {best_f1:.4f})")


def main():
    """Main application function."""
    # Header
    st.markdown('<p class="main-header">💳 Credit Risk Scoring Dashboard</p>', unsafe_allow_html=True)
    
    # Load models and artifacts
    models, scaler, selected_features, config = load_models_and_artifacts()
    
    if models is None:
        st.error("Failed to load models. Please ensure models are trained and saved.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Comparison", "ℹ️ About"])
    
    # Tab 1: Prediction
    with tab1:
        # Sidebar inputs
        features = create_input_features()
        
        # Model selection
        st.subheader("Model Selection")
        model_name = st.selectbox(
            "Choose a model for prediction:",
            list(models.keys()),
            index=2  # Default to XGBoost
        )
        
        # Predict button
        if st.button("🚀 Predict Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing borrower profile..."):
                # Preprocess features
                features_scaled = preprocess_features(features, scaler, selected_features)
                
                # Make prediction
                prediction, probabilities = predict_risk(features_scaled, models, model_name)
                
                # Display results
                st.success("Analysis complete!")
                display_risk_prediction(prediction, probabilities)
                
                # Display input summary
                with st.expander("📋 View Input Summary"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Demographics**")
                        st.write(f"- Age: {features['age']}")
                        st.write(f"- Years Employed: {features['person_emp_length']}")
                        st.write(f"- Home Ownership: {features['person_home_ownership']}")
                    
                    with col2:
                        st.write("**Financial**")
                        st.write(f"- Annual Income: ${features['person_income']:,.0f}")
                        st.write(f"- Loan Amount: ${features['loan_amnt']:,.0f}")
                        st.write(f"- Interest Rate: {features['loan_int_rate']:.1f}%")
                        st.write(f"- Debt-to-Income: {features['loan_percent_income']:.1%}")
                
                # SHAP explanation
                if model_name != 'Logistic Regression':
                    display_shap_explanation(
                        features_scaled, models[model_name], 
                        selected_features, model_name
                    )
    
    # Tab 2: Model Comparison
    with tab2:
        comparison_df = load_comparison_data()
        display_model_comparison(comparison_df)
    
    # Tab 3: About
    with tab3:
        st.header("About This Dashboard")
        
        st.markdown("""
        ### 🎯 Purpose
        This dashboard provides an interactive interface for credit risk assessment using machine learning models.
        
        ### 🤖 Models
        Three models are available for prediction:
        - **Logistic Regression**: Fast, interpretable baseline model
        - **Random Forest**: Ensemble model with strong performance
        - **XGBoost**: Advanced gradient boosting (typically best performance)
        
        ### 📊 Risk Tiers
        - **Low Risk**: Borrowers with strong credit profiles
        - **Medium Risk**: Borrowers with moderate risk factors
        - **High Risk**: Borrowers with significant risk indicators
        
        ### 🔍 SHAP Explanations
        SHAP (SHapley Additive exPlanations) values show how each feature contributes to the prediction:
        - Helps understand *why* a prediction was made
        - Identifies most important factors
        - Provides transparency and interpretability
        
        ### 📈 Features Considered
        - Demographics (age, employment)
        - Financial (income, loan amount, interest rate)
        - Credit history (length, previous defaults)
        - Loan characteristics (purpose, grade)
        
        ### 💡 How to Use
        1. Enter borrower information in the sidebar
        2. Select a model
        3. Click "Predict Risk"
        4. Review the prediction and explanation
        
        ### 📚 Learn More
        - View the complete codebase on GitHub
        - Read the project documentation
        - Explore model training notebooks
        """)
        
        st.info("💼 **Portfolio Project** - Built with Python, scikit-learn, XGBoost, SHAP, and Streamlit")


if __name__ == "__main__":
    main()
