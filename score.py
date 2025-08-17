import streamlit as st
import numpy as np
import joblib

# Load saved model
model = joblib.load("xgboost_model.pkl")

# Streamlit UI configuration
st.set_page_config(page_title="🎓 Student Score Predictor", layout="centered")
st.title("📘 Predict Student Score Based on Activity")

st.markdown("Enter the student's behavioral and activity-related features below:")

# Input form
AveCarelessness = st.slider("Average Carelessness", 0.0, 1.0, 0.1)
AveResFrust = st.slider("Average Frustration", 0.0, 1.0, 0.1)
AveResOfftask = st.slider("Average Off-task", 0.0, 1.0, 0.1)
SY_usage = st.number_input("SY ASSISTments Usage (e.g., 2004)", min_value=2000, max_value=2030, value=2005)
Selective = st.selectbox("Selective (1 = Yes, 0 = No)", [0, 1])
Enrolled = st.number_input("Enrolled Count", min_value=0, value=1000)
Ln = st.number_input("Ln", min_value=0, value=0)
Ln_1 = st.number_input("Ln-1", min_value=0, value=0)
AveResBored = st.slider("Average Boredom", 0.0, 1.0, 0.1)
action_num = st.number_input("Action Number", min_value=0, value=1000)

# Prepare input
input_features = np.array([[AveCarelessness, AveResFrust, AveResOfftask,
                            SY_usage, Selective, Enrolled,
                            Ln, Ln_1, AveResBored, action_num]])

# Predict
if st.button("🎯 Predict Score"):
    score = model.predict(input_features)[0]
    st.success(f"📈 Predicted Student Score: **{score:.2f} / 100**")
