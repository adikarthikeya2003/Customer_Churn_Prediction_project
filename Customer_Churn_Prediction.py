import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction & EDA Dashboard")

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

    # Align with training
    X = X.reindex(columns=model_features, fill_value=0)
    return X

# --- File Upload (once) ---
uploaded = st.file_uploader("📂 Upload customer dataset (CSV)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("📄 Preview of uploaded data:", df.head())

    # ================== PREDICTIONS ==================
    st.header("🔮 Predictions")
    X_new = make_features(df)
    preds = rf_model.predict(X_new)
    proba = rf_model.predict_proba(X_new)[:, 1]

    out = df.copy()
    out['churn_pred'] = preds
    out['churn_prob'] = np.round(proba, 3)

    st.success("✅ Predictions generated!")
    st.write(out.head())

    st.download_button(
        "⬇ Download predictions as CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions.csv",
        mime="text/csv",
    )

    # ================== EDA VISUALIZATIONS ==================
    st.header("📈 Exploratory Data Analysis")

    # --- Correlation Heatmap ---
    st.subheader("🔗 Correlation Heatmap (Numerical Features)")
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found for correlation.")

    # --- Churn Distribution ---
    if "Churn" in df.columns:
        st.subheader("📊 Churn Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Churn", data=df, ax=ax)
        st.pyplot(fig)

    # --- Categorical Distributions ---
    st.subheader("📊 Categorical Feature Distributions")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["customerID", "Churn"]]  # exclude IDs & churn

    for col in cat_cols:
        st.write(f"### {col}")
        fig, ax = plt.subplots()
        if "Churn" in df.columns:
            sns.countplot(x=col, hue="Churn", data=df, ax=ax)
        else:
            sns.countplot(x=col, data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- Feature Importance ---
    st.subheader("⭐ Feature Importance from Model")
    importances = pd.Series(rf_model.feature_importances_, index=model_features)
    top_feats = importances.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x=top_feats.values, y=top_feats.index, ax=ax)
    st.pyplot(fig)

else:
    st.info("Upload a CSV file to generate predictions and explore EDA.")
