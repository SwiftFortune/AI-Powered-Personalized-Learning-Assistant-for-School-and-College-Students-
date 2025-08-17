import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load your saved XGBoost model
model = joblib.load('best_student_pass_fail_model.pkl')

# UI Title
st.set_page_config(page_title="Student Pass/Fail Predictor", layout="centered")
st.title("🎓 Student Pass/Fail Predictor")

# Input Form
st.markdown("Fill in the student's details below to predict **Pass or Fail**:")

# Input Fields
gender = st.selectbox("Gender", ['male', 'female'])
ethnicity = st.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
parent_edu = st.selectbox("Parental Level of Education", [
    "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch Type", ['standard', 'free/reduced'])
prep = st.selectbox("Test Preparation Course", ['none', 'completed'])

math_score = st.slider("Math Score", 0, 100, 50)
reading_score = st.slider("Reading Score", 0, 100, 50)
writing_score = st.slider("Writing Score", 0, 100, 50)

# Manual Label Encoding (same as training)
gender_map = {'male': 1, 'female': 0}
ethnicity_map = {'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4}
edu_map = {
    "high school": 0,
    "some college": 1,
    "associate's degree": 2,
    "bachelor's degree": 3,
    "master's degree": 4
}
lunch_map = {'standard': 1, 'free/reduced': 0}
prep_map = {'none': 1, 'completed': 0}

# Create Input DataFrame
input_df = pd.DataFrame([[
    gender_map[gender],
    ethnicity_map[ethnicity],
    edu_map[parent_edu],
    lunch_map[lunch],
    prep_map[prep],
    math_score,
    reading_score,
    writing_score
]], columns=[
    'gender',
    'race/ethnicity',
    'parental level of education',
    'lunch',
    'test preparation course',
    'math score',
    'reading score',
    'writing score'
])

# Standardize Scores — fit scaler like in training (0-100 range)
scaler = StandardScaler()
scaler.fit(np.array([[i, i, i] for i in range(0, 101)]))  # Simulated range
input_df[['math score', 'reading score', 'writing score']] = scaler.transform(
    input_df[['math score', 'reading score', 'writing score']]
)

# Predict
if st.button("🔍 Predict Pass/Fail"):
    prediction = model.predict(input_df)[0]
    result = "✅ PASS" if prediction == 1 else "❌ FAIL"
    st.subheader(f"🎯 Prediction: {result}")
    if prediction == 1:
        st.success("This student is likely to pass. Great job! 🎉")
    else:
        st.warning("This student may need more support to succeed. 📘")
