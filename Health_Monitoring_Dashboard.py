import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Health Monitoring Dashboard",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# Generate Sample Health Data
# -----------------------------
@st.cache_data
def generate_health_data():
    np.random.seed(42)
    patients = [f"Patient-{i}" for i in range(1, 21)]
    genders = ["Male", "Female"]
    risk_levels = ["Low", "Medium", "High"]

    data = []
    for _ in range(500):
        patient = np.random.choice(patients)
        heart_rate = np.random.randint(55, 120)
        systolic = np.random.randint(100, 160)
        diastolic = np.random.randint(60, 100)
        temperature = round(np.random.uniform(36.0, 39.5), 1)
        oxygen = np.random.randint(88, 100)
        risk = np.random.choice(risk_levels, p=[0.5, 0.3, 0.2])

        data.append({
            "PATIENT_ID": patient,
            "GENDER": np.random.choice(genders),
            "DATE": datetime.now() - timedelta(days=np.random.randint(0, 90)),
            "HEART_RATE": heart_rate,
            "SYSTOLIC_BP": systolic,
            "DIASTOLIC_BP": diastolic,
            "BODY_TEMP": temperature,
            "OXYGEN_LEVEL": oxygen,
            "RISK_LEVEL": risk,
            "ALERT": heart_rate > 100 or oxygen < 92 or temperature > 38
        })

    return pd.DataFrame(data)

df = generate_health_data()

# -----------------------------
# Dashboard Title
# -----------------------------
st.title("🏥 Health Monitoring Dashboard")
st.markdown("### Real-Time Patient Vitals, Alerts, and Health Trends")

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🔍 Filters")

patient_filter = st.sidebar.multiselect(
    "Select Patient(s)",
    options=["All"] + sorted(df["PATIENT_ID"].unique().tolist()),
    default=["All"]
)

gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=["All"] + sorted(df["GENDER"].unique().tolist()),
    default=["All"]
)

risk_filter = st.sidebar.multiselect(
    "Select Risk Level",
    options=["All"] + sorted(df["RISK_LEVEL"].unique().tolist()),
    default=["All"]
)

# Apply filters
filtered_df = df.copy()

if "All" not in patient_filter:
    filtered_df = filtered_df[filtered_df["PATIENT_ID"].isin(patient_filter)]

if "All" not in gender_filter:
    filtered_df = filtered_df[filtered_df["GENDER"].isin(gender_filter)]

if "All" not in risk_filter:
    filtered_df = filtered_df[filtered_df["RISK_LEVEL"].isin(risk_filter)]

# -----------------------------
# KPI Metrics
# -----------------------------
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("👥 Patients Monitored", filtered_df["PATIENT_ID"].nunique())

with col2:
    st.metric("❤️ Avg Heart Rate", f"{filtered_df['HEART_RATE'].mean():.0f} bpm")

with col3:
    st.metric("🌡️ Avg Body Temp", f"{filtered_df['BODY_TEMP'].mean():.1f} °C")

with col4:
    alert_rate = (filtered_df["ALERT"].sum() / len(filtered_df)) * 100
    st.metric("🚨 Alert Rate", f"{alert_rate:.1f}%")

# -----------------------------
# Data Preview
# -----------------------------
st.markdown("---")
st.subheader("📋 Patient Data Preview")
st.dataframe(filtered_df.head(10), use_container_width=True)

# -----------------------------
# Vital Signs by Risk Level
# -----------------------------
st.markdown("---")
st.subheader("📊 Average Vitals by Risk Level")

vitals_by_risk = filtered_df.groupby("RISK_LEVEL")[
    ["HEART_RATE", "BODY_TEMP", "OXYGEN_LEVEL"]
].mean()

fig, ax = plt.subplots(figsize=(10, 6))
vitals_by_risk.plot(kind="bar", ax=ax)
ax.set_ylabel("Average Value")
ax.set_title("Vitals Comparison by Risk Level")
ax.grid(axis="y", alpha=0.3)

st.pyplot(fig)

# -----------------------------
# Health Alerts
# -----------------------------
st.markdown("---")
st.subheader("⚠️ Active Health Alerts")

alerts_df = filtered_df[filtered_df["ALERT"] == True]

alerts_summary = alerts_df.groupby("PATIENT_ID").agg({
    "HEART_RATE": "mean",
    "BODY_TEMP": "mean",
    "OXYGEN_LEVEL": "mean",
    "ALERT": "count"
}).reset_index()

alerts_summary.columns = [
    "Patient ID",
    "Avg Heart Rate",
    "Avg Body Temp",
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
    st.subheader("📈 Heart Rate Trend Over Time")
    filtered_df["DATE"] = pd.to_datetime(filtered_df["DATE"])
    hr_trend = filtered_df.groupby(filtered_df["DATE"].dt.to_period("W"))["HEART_RATE"].mean()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    hr_trend.plot(ax=ax2, marker="o", linewidth=2)
    ax2.set_ylabel("Avg Heart Rate")
    ax2.set_xlabel("Week")
    ax2.grid(alpha=0.3)

    st.pyplot(fig2)

with col2:
    st.subheader("🫁 Oxygen Level Trend Over Time")
    oxy_trend = filtered_df.groupby(filtered_df["DATE"].dt.to_period("W"))["OXYGEN_LEVEL"].mean()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    oxy_trend.plot(ax=ax3, marker="o", linewidth=2)
    ax3.set_ylabel("Avg Oxygen Level")
    ax3.set_xlabel("Week")
    ax3.grid(alpha=0.3)

    st.pyplot(fig3)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "**Health Monitoring Dashboard** | Patient Vitals • Alerts • Trends"
)
