#!/usr/bin/env python
"""Streamlit Heart Disease Predictor (cleaned)

This file contains only the Streamlit UI and model-loading logic.
Exploratory prints and training code were removed to avoid raw outputs
appearing in the app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time


def safe_load(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# Load saved objects (models, scaler, columns). If missing, show a friendly message in the UI.
scaler = safe_load("scaler.pkl")
model_columns = safe_load("columns.pkl")
rf_model = safe_load("random_forest_model.pkl")
knn_model = safe_load("knn_model.pkl")
dt_model = safe_load("dt_model.pkl")

# Numerical columns (same as cont_val in training)
NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Base feature names before encoding
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# =========================
# Streamlit Page Config & CSS
# =========================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e8eaed 100%);
            font-family: 'Helvetica Neue', Arial, sans-serif;
            color: #2c3e50;
        }

        h1 {
            color: #2c3e50 !important;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e74c3c;
        }

        .subtitle {
            text-align: center;
            color: #34495e !important;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #2c3e50;
        }

        [data-testid="stSidebar"] h2 {
            color: #ecf0f1 !important;
            font-size: 1.5rem;
            border-bottom: 1px solid #e74c3c;
            padding-bottom: 0.7rem;
            margin-bottom: 1.5rem;
        }

        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stExpander div,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #ecf0f1 !important;
        }

        /* General cards */
        .card {
            background-color: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            border-left: 4px solid #3498db;
        }

        .risk-meter { height: 8px; background-color: #eee; border-radius: 4px; }

        div.stButton > button { background-color: #e74c3c; color: white !important; }
    </style>
""", unsafe_allow_html=True)

# =========================
# Title
# =========================
st.markdown("<h1>Heart Disease Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-powered clinical decision support tool</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

# =========================
# Sidebar Inputs
# =========================
with st.sidebar:
    st.markdown("<h2>Patient Information</h2>", unsafe_allow_html=True)

    with st.expander("Demographics", expanded=True):
        age = st.slider("Age", 20, 100, 50)
        sex = st.radio("Sex", options=["Male", "Female"])
        sex_numeric = 1 if sex == "Male" else 0

    with st.expander("Clinical Measurements", expanded=True):
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 250)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        fbs_numeric = 1 if fbs == "Yes" else 0
        thalach = st.slider("Maximum Heart Rate", 70, 220, 150)

    with st.expander("Cardiac Assessment", expanded=True):
        cp_options = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
        cp = st.selectbox("Chest Pain Type", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])

        restecg_options = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
        restecg = st.selectbox("Resting ECG", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])

        exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
        exang_numeric = 1 if exang == "Yes" else 0

        oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0, 0.1)

        slope_options = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])

    with st.expander("Advanced Indicators", expanded=True):
        ca = st.slider("Number of Major Vessels (0-4)", 0, 4, 1)

        thal_options = {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Unknown"}
        thal = st.selectbox("Thalassemia", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])

# Create raw input DataFrame (before encoding)
user_data = {
    "age": age,
    "sex": sex_numeric,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs_numeric,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang_numeric,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}
user_input = pd.DataFrame(user_data, index=[0])

# Left column: overview
with col1:
    st.markdown("<h3>Patient Overview</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; border-left: 4px solid #3498db;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;"><div><strong>Age:</strong> {age}</div><div><strong>Sex:</strong> {sex}</div></div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;"><div><strong>BP:</strong> {trestbps} mmHg</div><div><strong>Cholesterol:</strong> {chol} mg/dl</div></div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;"><div><strong>Max HR:</strong> {thalach} bpm</div><div><strong>Chest Pain:</strong> {cp_options[cp]}</div></div>
    </div>
    """, unsafe_allow_html=True)

    def risk_level(val, feature):
        if feature == "chol":
            if val < 200:
                return "Low", "2ecc71"
            elif val < 240:
                return "Moderate", "f39c12"
            else:
                return "High", "e74c3c"
        elif feature == "trestbps":
            if val < 120:
                return "Normal", "2ecc71"
            elif val < 130:
                return "Elevated", "3498db"
            elif val < 140:
                return "Stage 1", "f39c12"
            else:
                return "Stage 2", "e74c3c"
        elif feature == "age":
            if val < 45:
                return "Lower", "2ecc71"
            elif val < 65:
                return "Moderate", "f39c12"
            else:
                return "Higher", "e74c3c"

    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        chol_level, chol_color = risk_level(chol, "chol")
        st.markdown(f"""
        <div style="text-align: center;"><p><strong>Cholesterol</strong></p><div class="risk-meter"><div class="risk-meter-fill" style="width: {min(100, max(0, (chol-100)/5))}%; background-color: #{chol_color};"></div></div><p style="color: #{chol_color};">{chol_level} ({chol} mg/dl)</p></div>
        """, unsafe_allow_html=True)

    with col_r2:
        bp_level, bp_color = risk_level(trestbps, "trestbps")
        st.markdown(f"""
        <div style="text-align: center;"><p><strong>Blood Pressure</strong></p><div class="risk-meter"><div class="risk-meter-fill" style="width: {min(100, max(0, (trestbps-80)/1.2))}%; background-color: #{bp_color};"></div></div><p style="color: #{bp_color};">{bp_level} ({trestbps} mmHg)</p></div>
        """, unsafe_allow_html=True)

    with col_r3:
        age_level, age_color = risk_level(age, "age")
        st.markdown(f"""
        <div style="text-align: center;"><p><strong>Age Risk</strong></p><div class="risk-meter"><div class="risk-meter-fill" style="width: {min(100, age)}%; background-color: #{age_color};"></div></div><p style="color: #{age_color};">{age_level} ({age} years)</p></div>
        """, unsafe_allow_html=True)

# Right column: prediction logic
with col2:
    st.markdown("<h3>Heart Disease Risk Assessment</h3>", unsafe_allow_html=True)

    def preprocess_for_model(df_raw: pd.DataFrame) -> pd.DataFrame:
        df_enc = pd.get_dummies(df_raw)
        df_enc = df_enc.reindex(columns=model_columns, fill_value=0)
        df_enc[NUM_COLS] = scaler.transform(df_enc[NUM_COLS])
        return df_enc

    if st.button("Generate Risk Assessment", key="predict_button"):
        with st.spinner("Analyzing patient data..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            # Ensure required models/resources are available
            missing = []
            if model_columns is None:
                missing.append("columns.pkl")
            if scaler is None:
                missing.append("scaler.pkl")
            if rf_model is None or knn_model is None or dt_model is None:
                missing.append("one or more model files")

            if missing:
                st.error(f"Missing required files: {', '.join(missing)}. Please run the training script to create them.")
            else:
                X_input = preprocess_for_model(user_input)

                rf_pred = rf_model.predict(X_input)[0]
                rf_proba = rf_model.predict_proba(X_input)[0][1]

                knn_pred = knn_model.predict(X_input)[0]
                knn_proba = knn_model.predict_proba(X_input)[0][1]

                dt_pred = dt_model.predict(X_input)[0]
                dt_proba = dt_model.predict_proba(X_input)[0][1]

                ensemble_risk = (rf_proba + knn_proba + dt_proba) / 3

                st.markdown("<h3>Overall Risk Assessment</h3>", unsafe_allow_html=True)
                risk_level_txt = "Low" if ensemble_risk < 0.4 else "Moderate" if ensemble_risk < 0.7 else "High"
                risk_color = "#2ecc71" if risk_level_txt == "Low" else "#f39c12" if risk_level_txt == "Moderate" else "#e74c3c"

                st.markdown(f"""
                <div style="text-align: center; margin-bottom: 20px;"><div style="background-color: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h2 style="color: {risk_color}; margin-bottom: 10px;">{risk_level_txt} Risk</h2>
                    <div style="height: 20px; background-color: #eee; border-radius: 10px; margin: 20px 0; position: relative;">
                        <div style="width: {ensemble_risk * 100}%; height: 100%; background: linear-gradient(90deg, #2ecc71, #f39c12, #e74c3c); border-radius: 10px;"></div>
                    </div>
                    <p style="font-size: 1.5rem; font-weight: bold; color: #2c3e50;">{ensemble_risk:.1%}</p>
                    <p style="color: #2c3e50;">Probability of Heart Disease</p>
                </div></div>
                """, unsafe_allow_html=True)

                col_rf, col_knn, col_dt = st.columns(3)

                def show_model_prediction(col, model_name, prediction, probability, icon):
                    result_text = "Positive" if prediction == 1 else "Negative"
                    result_color = "#e74c3c" if prediction == 1 else "#2ecc71"
                    col.markdown(f"""
                    <div style="background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 15px; border-left: 4px solid {result_color};">
                        <div style="text-align: center;"><span style="font-size: 1.5rem;">{icon}</span><h4 style="margin: 5px 0; color: #2c3e50;">{model_name}</h4></div>
                        <div style="text-align: center;"><p style="font-size: 1.2rem; font-weight: bold; color: {result_color};">{result_text}</p><p style="font-size: 0.9rem; color: #2c3e50;">Confidence: {probability:.1%}</p></div>
                    </div>
                    """, unsafe_allow_html=True)

                show_model_prediction(col_rf, "Random Forest", rf_pred, rf_proba, "🌲")
                show_model_prediction(col_knn, "K-Nearest Neighbors", knn_pred, knn_proba, "🔗")
                show_model_prediction(col_dt, "Decision Tree", dt_pred, dt_proba, "🌳")

                st.markdown("<h4>Key Findings & Recommendations</h4>", unsafe_allow_html=True)

                recommendations = []
                if chol > 200:
                    recommendations.append("Monitor cholesterol levels and consider dietary changes.")
                if trestbps >= 130:
                    recommendations.append("Follow up on blood pressure management.")
                if age > 65:
                    recommendations.append("Regular cardiac check-ups recommended for this age group.")
                if exang == "Yes":
                    recommendations.append("Further evaluation of exercise-induced angina is advised.")
                if ca > 0:
                    recommendations.append("Follow up with a cardiologist regarding coronary vessel health.")

                recommendations.append("Maintain a heart-healthy diet rich in fruits and vegetables.")
                recommendations.append("Engage in regular physical activity appropriate for your condition.")

                st.markdown('<div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin: 20px 0;">', unsafe_allow_html=True)
                if ensemble_risk > 0.7:
                    st.markdown('<p><strong>Clinical Assessment:</strong> Multiple risk factors indicate a high probability of coronary artery disease. Immediate follow-up with a cardiologist is strongly recommended.</p>', unsafe_allow_html=True)
                elif ensemble_risk > 0.4:
                    st.markdown('<p><strong>Clinical Assessment:</strong> The patient presents moderate risk factors. Additional diagnostic testing may be warranted.</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p><strong>Clinical Assessment:</strong> The patient currently shows a lower risk profile. Preventive measures and routine monitoring are advised.</p>', unsafe_allow_html=True)

                st.markdown('<p><strong>Recommendations:</strong></p>', unsafe_allow_html=True)
                for rec in recommendations[:5]:
                    st.markdown(f'<p style="margin: 5px 0 5px 15px;">• {rec}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h4>About This Tool</h4>
            <p>This clinical decision support system uses machine learning models trained on heart disease data to estimate the probability of heart disease.</p>
            <p>Enter patient information in the sidebar and click <strong>"Generate Risk Assessment"</strong> to get a detailed analysis.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h4>Model Information</h4>", unsafe_allow_html=True)
        model_info = [
            {"name": "Random Forest", "icon": "🌲", "description": "Ensemble of decision trees for robust, high-accuracy predictions."},
            {"name": "K-Nearest Neighbors", "icon": "🔗", "description": "Compares the patient to similar historical cases."},
            {"name": "Decision Tree", "icon": "🌳", "description": "Transparent, rule-based model similar to clinical decision trees."}
        ]

        for m in model_info:
            st.markdown(f"""
            <div style="background-color: #f9f9f9; border-radius: 8px; padding: 15px; margin-bottom: 15px; border-left: 3px solid #3498db;">
                <div style="display: flex; align-items: center;"><span style="font-size: 1.5rem; margin-right: 10px;">{m['icon']}</span><div><h5 style="margin: 0;">{m['name']}</h5><p style="margin: 5px 0 0 0; font-size: 0.9rem;">{m['description']}</p></div></div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
<div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0; text-align: center;">
    <p style="color: #7f8c8d; font-size: 0.9rem;">© 2025 Heart Disease Risk Assessment Tool</p>
    <p style="color: #7f8c8d; font-size: 0.9rem;">Clinical decision support powered by machine learning</p>
</div>
""", unsafe_allow_html=True)
