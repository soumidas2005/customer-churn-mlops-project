import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Churn Dashboard", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("final_dataset.csv")

df = load_data()

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("finalized_churn_forecaster.pkl")

model = load_model()

# -----------------------------
# SHAP (FINAL STABLE VERSION)
# -----------------------------
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# ❌ NO CACHE HERE (IMPORTANT)
def compute_shap_values(explainer, data):
    return explainer.shap_values(data)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Customer Search",
        "Segment View",
        "What-If Analysis",
        "Churn Risk Dashboard",
        "Segment Visualization"
    ]
)

# PIPELINE BUTTON
if st.sidebar.button("Run Full Pipeline"):
    st.write("🚀 Running Pipeline...")
    os.system("python airflow_pipeline.py")
    st.success("✅ Pipeline Executed")

# -----------------------------
# OVERVIEW
# -----------------------------
if page == "Overview":
    st.title("📊 Customer Churn Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("Avg Churn Risk", round(df["Churn_Risk"].mean(), 3) if "Churn_Risk" in df.columns else "N/A")
    col3.metric("Avg Revenue", round(df["Monetary"].mean(), 2) if "Monetary" in df.columns else "N/A")

    if "Churn" in df.columns:
        st.subheader("Churn Distribution")
        st.bar_chart(df["Churn"].value_counts())

# -----------------------------
# CUSTOMER SEARCH
# -----------------------------
elif page == "Customer Search":
    st.title("🔍 Customer Search")

    idx = st.number_input("Customer Index", 0, len(df)-1, 0)
    customer = df.iloc[[idx]]

    st.write(customer)

    if st.button("Predict Churn"):
        input_data = customer.drop(columns=["Churn"], errors="ignore")

        expected_cols = model.feature_names_in_

        for col in expected_cols:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[expected_cols]

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.success(f"Prediction: {'Churn' if pred==1 else 'No Churn'}")
        st.info(f"Probability: {prob:.2f}")

# -----------------------------
# SEGMENT VIEW
# -----------------------------
elif page == "Segment View":
    st.title("📊 Customer Segmentation")

    if "Segment" not in df.columns:
        st.warning("⚠️ Segment column not found.")
    else:
        st.bar_chart(df["Segment"].value_counts())
        st.write(df.groupby("Segment").mean())

# -----------------------------
# WHAT-IF ANALYSIS
# -----------------------------
elif page == "What-If Analysis":
    st.title("⚙️ What-If Simulation")

    monthly = st.slider("Monthly Charges", 0, 200, 70)
    tenure = st.slider("Tenure", 0, 72, 12)
    support = st.slider("Support Tickets", 0, 10, 2)

    if st.button("Simulate"):
        sample = df.drop(columns=["Churn"], errors="ignore").iloc[[0]].copy()

        if "MonthlyCharges" in sample.columns:
            sample["MonthlyCharges"] = monthly
        if "tenure" in sample.columns:
            sample["tenure"] = tenure
        if "support_tickets" in sample.columns:
            sample["support_tickets"] = support

        expected_cols = model.feature_names_in_

        for col in expected_cols:
            if col not in sample.columns:
                sample[col] = 0

        sample = sample[expected_cols]

        prob = model.predict_proba(sample)[0][1]
        st.success(f"Predicted Churn Risk: {prob:.2f}")

    # EXPORT
    st.subheader("📤 Export High-Risk Customers")

    if "Churn_Risk" in df.columns:
        high_risk = df[df["Churn_Risk"] > 0.7]

        st.download_button(
            label="Download CSV",
            data=high_risk.to_csv(index=False),
            file_name="high_risk_customers.csv",
            mime="text/csv"
        )

# -----------------------------
# CHURN DASHBOARD
# -----------------------------
elif page == "Churn Risk Dashboard":
    st.title("🔥 Churn Risk Dashboard")

    input_df = df.drop(columns=["Churn"], errors="ignore")

    expected_cols = model.feature_names_in_

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_cols]

    probs = model.predict_proba(input_df)[:, 1]

    df_dash = df.copy()
    df_dash["Churn_Probability"] = probs

    st.subheader("📊 Customer Risk Table")

    st.dataframe(df_dash.sort_values(by="Churn_Probability", ascending=False).head(50))

    # SHAP
    st.subheader("🧠 SHAP Explainability")

    if st.button("Load SHAP Analysis"):
        explainer = get_explainer(model)

        sample_df = input_df.sample(min(200, len(input_df)), random_state=42)

        shap_values = compute_shap_values(explainer, sample_df)

        shap.summary_plot(shap_values, sample_df, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

# -----------------------------
# SEGMENT VISUALIZATION
# -----------------------------
elif page == "Segment Visualization":
    st.title("📈 Segment Visualization")

    if "Segment" not in df.columns or "Churn_Risk" not in df.columns:
        st.warning("⚠️ Missing columns")
    else:
        fig, ax = plt.subplots()
        ax.scatter(df["Monetary"], df["Churn_Risk"])
        ax.set_xlabel("Revenue")
        ax.set_ylabel("Churn Risk")

        st.pyplot(fig)

        st.subheader("Average Metrics per Segment")
        st.bar_chart(df.groupby("Segment")["Churn_Risk"].mean())