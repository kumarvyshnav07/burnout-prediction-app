import streamlit as st
import pandas as pd
import joblib
import os

# ==============================
# Load Model & Columns
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model   = joblib.load(os.path.join(BASE_DIR, "burnout_model.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title = "Student Burnout Predictor",
    page_icon  = "🎓",
    layout     = "centered"
)

st.title("🎓 Student Burnout Prediction")
st.markdown("Fill in the details below to predict your burnout level.")
st.divider()

# ==============================
# SECTION 1 — Personal Info
# ==============================
st.subheader("👤 Personal Information")
col1, col2 = st.columns(2)

with col1:
    age    = st.slider("Age", 18, 26, 20)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

with col2:
    course = st.selectbox("Course", ["BTech", "Medicine", "Arts", "Commerce", "Science", "MBA"])
    year   = st.selectbox("Year", ["1st", "2nd", "3rd", "4th"])

st.divider()

# ==============================
# SECTION 2 — Academic
# ==============================
st.subheader("📚 Academic Factors")
col3, col4 = st.columns(2)

with col3:
    study      = st.slider("Daily Study Hours", 0.0, 16.0, 6.0, step=0.5)
    attendance = st.slider("Attendance %", 30.0, 100.0, 75.0, step=0.5)

with col4:
    cgpa     = st.slider("CGPA", 0.0, 10.0, 7.0, step=0.1)
    academic = st.slider("Academic Pressure (1-10)", 1, 10, 5)

st.divider()

# ==============================
# SECTION 3 — Lifestyle
# ==============================
st.subheader("🌙 Lifestyle Factors")
col5, col6 = st.columns(2)

with col5:
    sleep       = st.slider("Daily Sleep Hours", 3.0, 10.0, 7.0, step=0.5)
    sleep_qual  = st.selectbox("Sleep Quality", ["Poor", "Average", "Good", "Excellent"])
    screen      = st.slider("Screen Time Hours", 0.0, 16.0, 4.0, step=0.5)

with col6:
    activity    = st.slider("Physical Activity Hours", 0.0, 4.0, 1.0, step=0.1)
    internet_q  = st.selectbox("Internet Quality", ["Poor", "Average", "Good", "Excellent"])

st.divider()

# ==============================
# SECTION 4 — Mental Health
# ==============================
st.subheader("🧠 Mental Health Factors")
col7, col8 = st.columns(2)

with col7:
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    anxiety      = st.slider("Anxiety Score (1-10)", 1.0, 10.0, 5.0, step=0.1)
    depression   = st.slider("Depression Score (1-10)", 1.0, 10.0, 5.0, step=0.1)

with col8:
    financial = st.slider("Financial Stress (1-10)", 1, 10, 5)
    social    = st.slider("Social Support (1-10)", 1.0, 10.0, 5.0, step=0.1)

st.divider()

# ==============================
# BUILD INPUT DATAFRAME
# ==============================
input_dict = {
    "age"                     : age,
    "daily_study_hours"       : study,
    "daily_sleep_hours"       : sleep,
    "screen_time_hours"       : screen,
    "anxiety_score"           : anxiety,
    "depression_score"        : depression,
    "academic_pressure_score" : academic,
    "financial_stress_score"  : financial,
    "social_support_score"    : social,
    "physical_activity_hours" : activity,
    "attendance_percentage"   : attendance,
    "cgpa"                    : cgpa,

    # Categorical one-hot — gender
    "gender_Female"           : 1 if gender == "Female" else 0,
    "gender_Male"             : 1 if gender == "Male"   else 0,
    "gender_Other"            : 1 if gender == "Other"  else 0,

    # Categorical one-hot — course
    "course_Arts"             : 1 if course == "Arts"      else 0,
    "course_BTech"            : 1 if course == "BTech"     else 0,
    "course_Commerce"         : 1 if course == "Commerce"  else 0,
    "course_MBA"              : 1 if course == "MBA"       else 0,
    "course_Medicine"         : 1 if course == "Medicine"  else 0,
    "course_Science"          : 1 if course == "Science"   else 0,

    # Categorical one-hot — year
    "year_1st"                : 1 if year == "1st" else 0,
    "year_2nd"                : 1 if year == "2nd" else 0,
    "year_3rd"                : 1 if year == "3rd" else 0,
    "year_4th"                : 1 if year == "4th" else 0,

    # Categorical one-hot — stress level (stored as string)
    **{f"stress_level_{i}": 1 if str(stress_level) == str(i) else 0 for i in range(1, 11)},

    # Categorical one-hot — sleep quality
    "sleep_quality_Average"   : 1 if sleep_qual == "Average"   else 0,
    "sleep_quality_Excellent" : 1 if sleep_qual == "Excellent" else 0,
    "sleep_quality_Good"      : 1 if sleep_qual == "Good"      else 0,
    "sleep_quality_Poor"      : 1 if sleep_qual == "Poor"      else 0,

    # Categorical one-hot — internet quality
    "internet_quality_Average"   : 1 if internet_q == "Average"   else 0,
    "internet_quality_Excellent" : 1 if internet_q == "Excellent" else 0,
    "internet_quality_Good"      : 1 if internet_q == "Good"      else 0,
    "internet_quality_Poor"      : 1 if internet_q == "Poor"      else 0,
}

input_df = pd.DataFrame([input_dict])

# Align exactly with training columns (fills any missing with 0)
input_df = input_df.reindex(columns=columns, fill_value=0)

# ==============================
# PREDICTION BUTTON
# ==============================
st.markdown("###")
predict_btn = st.button("🚀 Predict Burnout Level", use_container_width=True)

if predict_btn:
    pred = model.predict(input_df)[0]

    st.markdown("---")
    if pred == 0:
        st.success("## 🟢 Burnout Level: LOW")
        st.markdown("You're doing great! Keep maintaining healthy habits.")
    elif pred == 1:
        st.warning("## 🟡 Burnout Level: MEDIUM")
        st.markdown("Watch out! Consider reducing stress and improving sleep.")
    else:
        st.error("## 🔴 Burnout Level: HIGH")
        st.markdown("⚠️ High burnout detected. Please seek support and take a break.")

    st.markdown("---")

    # Show input summary
    with st.expander("📋 View Your Input Summary"):
        summary = {
            "Age": age, "Gender": gender, "Course": course, "Year": year,
            "Study Hours": study, "Sleep Hours": sleep, "Screen Time": screen,
            "CGPA": cgpa, "Attendance %": attendance,
            "Stress": stress_level, "Anxiety": anxiety,
            "Depression": depression, "Academic Pressure": academic,
            "Financial Stress": financial, "Social Support": social,
            "Sleep Quality": sleep_qual, "Internet Quality": internet_q,
            "Physical Activity": activity
        }
        st.table(pd.DataFrame(summary.items(), columns=["Factor", "Value"]))