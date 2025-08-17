import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import cv2

# Load the trained model
model = load_model('digit_model.h5')

# -------------------------------
# PAGE SETUP
st.set_page_config(page_title="Handwritten Digit Recognition", layout="centered")

# -------------------------------
# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["🏠 Introduction", "🔍 Predict Digit", "👤 About Me"])

# -------------------------------
# INTRODUCTION PAGE
if app_mode == "🏠 Introduction":
    st.title("🧠 Handwritten Digit Recognition using CNN")
    st.write("""
    This project demonstrates a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset** to recognize handwritten digits from student math assignments.
    
    ### 🔍 What’s inside:
    - **Dataset**: MNIST (70,000 28x28 grayscale digit images)
    - **Model**: CNN with 2 Convolution + MaxPooling layers
    - **Framework**: TensorFlow / Keras
    - **UI**: Streamlit for web-based interaction
    
    📦 The model predicts digits 0 through 9 in real-time from uploaded images!
    """)

# -------------------------------
# PREDICTION PAGE
elif app_mode == "🔍 Predict Digit":
    st.title("✍️ Upload Your Handwritten Digit")

    uploaded_file = st.file_uploader("Upload an image of a digit (28x28 grayscale or similar)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        st.image(image, caption="Uploaded Digit", use_column_width=False)

        # Preprocess the image
        image = ImageOps.invert(image)  # Optional: invert if background is white
        image = image.resize((28, 28))  # Resize to 28x28
        img_array = np.array(image)

        # Normalize and reshape
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_array)
        pred_class = np.argmax(prediction)

        st.success(f"✅ Predicted Digit: **{pred_class}**")
        st.bar_chart(prediction[0])

# -------------------------------
# ABOUT ME PAGE
elif app_mode == "👤 About Me":
    st.title("👨‍💻 About Me")
    st.markdown("""
    - **Name:** Sachin Hembram  
    - **Role:** Aspiring Data Scientist / ML Engineer  
    - **Skills:** Python, Machine Learning, Deep Learning, Streamlit, Computer Vision  
    - **Projects:** Digit Recognition, Voice Gender Prediction, Amazon Delivery Time, Student Dropout Prediction, and more.  
    - **Connect:** [LinkedIn](https://www.linkedin.com) | [GitHub](https://github.com)

    📫 *Feel free to reach out for collaborations, questions, or feedback!*
    """)

