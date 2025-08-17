import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Streamlit page config
st.set_page_config(page_title="🎓 Student Dropout Risk Predictor", layout="centered")
st.title("🎯 Predict Student Dropout Risk")
st.markdown("Enter the student details below to check if they are at risk of dropping out.")

# Load model with error handling
try:
    model = joblib.load("best_dropout_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file 'best_dropout_model.pkl' not found. Please make sure it's in the same directory.")
    st.stop()

# Input fields
time_spent = st.number_input("⏱ Weekly Time Spent on Platform", min_value=0.0)
quiz_score = st.number_input("📝 Average Quiz Score", min_value=0.0)
forum_posts = st.number_input("💬 Number of Forum Posts", min_value=0)
video_percent = st.slider("🎥 Video Watched (%)", 0.0, 100.0)
assignments = st.number_input("📚 Assignments Submitted", min_value=0)
login_freq = st.number_input("🔁 Login Frequency (per week)", min_value=0)
session_duration = st.number_input("⏳ Avg Session Duration (min)", min_value=0.0)

device_type = st.selectbox("📱 Device Type", ["Mobile", "Tablet", "Desktop"])
difficulty = st.selectbox("📊 Course Difficulty", ["Easy", "Medium", "Hard"])
region = st.selectbox("📍 Region", ["Urban", "Suburban", "Rural"])
gender = st.selectbox("🧑 Gender", ["male", "female"])
ethnicity = st.selectbox("🌍 Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
education = st.selectbox("🎓 Parental Education", [
    "some high school", "high school", "some college", 
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("🥗 Lunch Type", ["standard", "free/reduced"])
prep_course = st.selectbox("📘 Test Preparation Course", ["none", "completed"])
math_score = st.number_input("📐 Math Score", min_value=0)
reading_score = st.number_input("📖 Reading Score", min_value=0)
writing_score = st.number_input("✍️ Writing Score", min_value=0)

# Manual label encoding (must match training!)
device_map = {"Mobile": 0, "Tablet": 1, "Desktop": 2}
difficulty_map = {"Easy": 0, "Medium": 1, "Hard": 2}
region_map = {"Rural": 0, "Suburban": 1, "Urban": 2}
gender_map = {"female": 0, "male": 1}
ethnicity_map = {"group A": 0, "group B": 1, "group C": 2, "group D": 3, "group E": 4}
education_map = {
    "some high school": 0, "high school": 1, "some college": 2,
    "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5
}
lunch_map = {"free/reduced": 0, "standard": 1}
prep_map = {"none": 0, "completed": 1}

# Create feature vector
features = np.array([[
    time_spent,
    quiz_score,
    forum_posts,
    video_percent,
    assignments,
    login_freq,
    session_duration,
    device_map[device_type],
    difficulty_map[difficulty],
    region_map[region],
    gender_map[gender],
    ethnicity_map[ethnicity],
    education_map[education],
    lunch_map[lunch],
    prep_map[prep_course],
    math_score,
    reading_score,
    writing_score
]])

# Prediction
if st.button("Predict Dropout Risk"):
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]  # probability of class 1

    if pred == 1:
        st.error(f"⚠️ High Risk: This student is likely to drop out. (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ Low Risk: This student is likely to stay engaged. (Confidence: {1 - prob:.2%})")
