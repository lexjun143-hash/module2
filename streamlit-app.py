import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from snowflake.snowpark.context import get_active_session

# Get Snowflake session
session = get_active_session()

# Load health monitoring data from Snowflake
@st.cache_data
def load_data(_session):
    query = """
    SELECT
        PATIENT_ID,
        PATIENT_NAME,
        GENDER,
        AGE,
        RECORD_DATE,
        HEART_RATE,
        BODY_TEMPERATURE,
        OXYGEN_LEVEL,
        BLOOD_PRESSURE,
        RISK_LEVEL,
        ALERT_STATUS,
        LOCATION
    FROM HEALTH_DB.PUBLIC.PATIENT_VITALS
    """
    df = _session.sql(query).to_pandas()
    return df

# Load dataset
df = load_data(session)

# -----------------------------
# Dashboard Title
# -----------------------------
st.title("🏥 Health Monitoring Dashboard")
st.markdown("### Real-Time Patient Vitals, Risk Analysis, and Alerts")

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🔍 Filters")

patients = ["All"] + sorted(df["PATIENT_NAME"].dropna().unique().tolist())
selected_patients = st.sidebar.multiselect(
    "Select Patient(s)", patients, default=["All"]
)

genders = ["All"] + sorted(df["GENDER"].dropna().unique().tolist())
selected_gender = st.sidebar.multiselect(
    "Select Gender", genders, default=["All"]
)

risk_levels = ["All"] + sorted(df["RISK_LEVEL"].dropna().unique().tolist())
selected_risk = st.sidebar.multiselect(
    "Select Risk Level", risk_levels, default=["All"]
)

# Apply filters
filtered_df = df.copy()

if "All" not in selected_patients:
    filtered_df = filtered_df[filtered_df["PATIENT_NAME"].isin(selected_patients)]

if "All" not in selected_gender:
    filtered_df = filtered_df[filtered_df["GENDER"].isin(selected_gender)]

if "All" not in selected_risk:
    filtered_df = filtered_df[filtered_df["RISK_LEVEL"].isin(selected_risk)]

# -----------------------------
# KPI Metrics
# -----------------------------
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("👥 Patients", filtered_df["PATIENT_ID"].nunique())

with col2:
    st.metric("❤️ Avg Heart Rate", f"{filtered_df['HEART_RATE'].mean():.0f} bpm")

with col3:
    st.metric("🌡️ Avg Temperature", f"{filtered_df['BODY_TEMPERATURE'].mean():.1f} °C")

with col4:
    alert_rate = (filtered_df["ALERT_STATUS"].sum() / len(filtered_df)) * 100
    st.metric("🚨 Alert Rate", f"{alert_rate:.1f}%")

# -----------------------------
# Data Preview
# -----------------------------
st.markdown("---")
st.subheader("📋 Patient Vitals Preview")
st.dataframe(filtered_df.head(10), use_container_width=True)

# -----------------------------
# Average Vitals by Risk Level
# -----------------------------
st.markdown("---")
st.subheader("📊 Average Vitals by Risk Level")

risk_summary = filtered_df.groupby("RISK_LEVEL").agg({
    "HEART_RATE": "mean",
    "BODY_TEMPERATURE": "mean",
    "OXYGEN_LEVEL": "mean"
})

fig, ax = plt.subplots(figsize=(10, 6))
risk_summary.plot(kind="bar", ax=ax)
ax.set_ylabel("Average Value")
ax.set_title("Patient Vitals by Risk Level")
ax.grid(axis="y", alpha=0.3)

st.pyplot(fig)

# -----------------------------
# Health Alerts
# -----------------------------
st.markdown("---")
st.subheader("⚠️ Patients with Active Alerts")

alerts_df = filtered_df[filtered_df["ALERT_STATUS"] == True]

alerts_summary = alerts_df.groupby("PATIENT_NAME").agg({
    "HEART_RATE": "mean",
    "BODY_TEMPERATURE": "mean",
    "OXYGEN_LEVEL": "mean",
    "ALERT_STATUS": "count"
}).reset_index()

alerts_summary.columns = [
    "Patient Name",
    "Avg Heart Rate",
    "Avg Temperature",
    "Avg Oxygen Level",
    "Alert Count"
]

st.dataframe(alerts_summary.sort_values("Alert Count", ascending=False),
             use_container_width=True)

# -----------------------------
# Trends Over Time
# -----------------------------
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Heart Rate Trend")
    filtered_df["RECORD_DATE"] = pd.to_datetime(filtered_df["RECORD_DATE"])
    heart_trend = filtered_df.groupby(
        filtered_df["RECORD_DATE"].dt.to_period("M")
    )["HEART_RATE"].mean()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    heart_trend.plot(ax=ax2, marker="o", linewidth=2)
    ax2.set_ylabel("Avg Heart Rate")
    ax2.set_xlabel("Month")
    ax2.grid(alpha=0.3)

    st.pyplot(fig2)

with col2:
    st.subheader("🫁 Oxygen Level Trend")
    oxygen_trend = filtered_df.groupby(
        filtered_df["RECORD_DATE"].dt.to_period("M")
    )["OXYGEN_LEVEL"].mean()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    oxygen_trend.plot(ax=ax3, marker="o", linewidth=2)
    ax3.set_ylabel("Avg Oxygen Level")
    ax3.set_xlabel("Month")
    ax3.grid(alpha=0.3)

    st.pyplot(fig3)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "**Health Monitoring Dashboard** | Powered by Streamlit & Snowflake"
)
