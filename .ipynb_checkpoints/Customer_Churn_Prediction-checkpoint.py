import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction App")
st.write("Upload a CSV with customer records to get churn predictions.")

# --- load artifacts ---
rf_model = joblib.load("rf_model.pkl")
model_features = joblib.load("model_features.pkl")

# --- SAME PREPROCESSING AS TRAINING ---
def make_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df['tenure'] = pd.to_numeric(df.get('tenure'), errors='coerce').fillna(0)
    df['MonthlyCharges'] = pd.to_numeric(df.get('MonthlyCharges'), errors='coerce').fillna(0)
    df['TotalCharges'] = pd.to_numeric(df.get('TotalCharges'), errors='coerce').fillna(0)
    df = df.replace({'No internet service': 'No', 'No phone service': 'No'})

    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[-0.1, 12, 24, 48, 1000],
        labels=['0-1 year', '1-2 years', '2-4 years', '4+ years']
    )

    avg = df['TotalCharges'] / df['tenure'].replace(0, np.nan)
    df['avg_monthly_charges'] = avg.replace([np.inf, -np.inf], np.nan).fillna(0)

    base_cols = [
        'Dependents','tenure','PhoneService','MultipleLines','InternetService',
        'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
        'StreamingTV','StreamingMovies','Contract','PaperlessBilling',
        'PaymentMethod','MonthlyCharges','TotalCharges','tenure_group','avg_monthly_charges'
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = np.nan

    X = df[base_cols].copy()
    X = pd.get_dummies(X, drop_first=True)

    # CRITICAL: Align to training columns so feature names match
    X = X.reindex(columns=model_features, fill_value=0)
    return X

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df_new = pd.read_csv(uploaded)
    st.write("Preview:", df_new.head())

    X_new = make_features(df_new)
    preds = rf_model.predict(X_new)
    proba = rf_model.predict_proba(X_new)[:, 1]

    out = df_new.copy()
    out['churn_pred'] = preds
    out['churn_prob'] = np.round(proba, 3)

    st.write("Predictions:", out.head())
    st.download_button(
        "Download predictions as CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions.csv",
        mime="text/csv",
    )
else:
    st.info("Upload a CSV to get predictions.")
