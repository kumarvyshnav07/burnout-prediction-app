import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("burnout_model.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Burnout Predictor", layout="centered")

st.title("🎓 Student Burnout Prediction")
st.markdown("Enter details to predict burnout level")

st.divider()

# ==============================
# INPUTS
# ==============================

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 30)
    study = st.slider("Daily Study Hours", 0.0, 12.0)
    sleep = st.slider("Daily Sleep Hours", 0.0, 12.0)
    screen = st.slider("Screen Time Hours", 0.0, 12.0)

with col2:
    anxiety = st.slider("Anxiety Score", 0, 10)
    depression = st.slider("Depression Score", 0, 10)
    academic = st.slider("Academic Pressure", 0, 10)
    financial = st.slider("Financial Stress", 0, 10)

st.divider()

col3, col4 = st.columns(2)

with col3:
    social = st.slider("Social Support", 0, 10)
    activity = st.slider("Physical Activity (hrs)", 0.0, 5.0)

with col4:
    attendance = st.slider("Attendance %", 0.0, 100.0)
    cgpa = st.slider("CGPA", 0.0, 10.0)

# ==============================
# CREATE INPUT DATA
# ==============================

input_data = pd.DataFrame([{
    "age": age,
    "daily_study_hours": study,
    "daily_sleep_hours": sleep,
    "screen_time_hours": screen,
    "anxiety_score": anxiety,
    "depression_score": depression,
    "academic_pressure_score": academic,
    "financial_stress_score": financial,
    "social_support_score": social,
    "physical_activity_hours": activity,
    "attendance_percentage": attendance,
    "cgpa": cgpa
}])

# Align with training columns
input_data = input_data.reindex(columns=columns, fill_value=0)

# ==============================
# PREDICTION
# ==============================

if st.button("🚀 Predict Burnout"):
    pred = model.predict(input_data)[0]

    levels = {0: "🟢 Low", 1: "🟡 Medium", 2: "🔴 High"}
    st.success(f"Predicted Burnout Level: {levels[pred]}")