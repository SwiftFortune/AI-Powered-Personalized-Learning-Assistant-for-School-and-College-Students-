import streamlit as st
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------- Load Artifacts -------------------- #
model = load_model("topic_detection_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

label_encoder = joblib.load("label_encoder.pkl")

MAX_SEQUENCE_LENGTH = 200

# -------------------- Label Mapping -------------------- #
label_mapping = {
    0: 'BUSINESS',
    1: 'ENTERTAINMENT',
    2: 'HEALTH',
    3: 'NATION',
    4: 'SCIENCE',
    5: 'SPORTS',
    6: 'TECHNOLOGY',
    7: 'WORLD'
}

# -------------------- Prediction Function -------------------- #
def predict_topic(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    pred = model.predict(padded)
    pred_class = np.argmax(pred, axis=1)[0]
    topic = label_mapping[pred_class]
    return topic

# -------------------- Streamlit UI -------------------- #

# Navigation Sidebar
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🧠 Topic Prediction", "📌 About"])

# Home Page
if page == "🏠 Home":
    st.title("📰 News Topic Detection")
    st.image("https://cdn.pixabay.com/photo/2016/12/20/22/51/news-1927130_1280.png", use_column_width=True)
    st.markdown("""
        Welcome to the **News Topic Detection App**!  
        This tool uses a **BiLSTM deep learning model** trained on labeled news titles to predict the **topic** of a news headline.

        ### 🔍 Topics:
        - BUSINESS
        - ENTERTAINMENT
        - HEALTH
        - NATION
        - SCIENCE
        - SPORTS
        - TECHNOLOGY
        - WORLD

        ---
        ✅ Enter a **news headline** under the *Topic Prediction* tab to try it out.
    """)

# Prediction Page
elif page == "🧠 Topic Prediction":
    st.title("🧠 Predict News Topic")
    user_input = st.text_area("Enter a news headline:", "")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a news headline.")
        else:
            topic = predict_topic(user_input)
            st.success(f"🧾 Predicted Topic: **{topic}**")

# About Page
elif page == "📌 About":
    st.title("📌 About This Project")
    st.markdown("""
        This app was developed as part of a **deep learning NLP project** to classify news titles into categories using a **BiLSTM model**.

        #### 👨‍💻 Technologies Used:
        - Python 🐍
        - TensorFlow / Keras 🤖
        - NLTK / Text Preprocessing 📚
        - Streamlit 🧑‍💻
        - Tokenizer, LabelEncoder, Padding ⛏️

        #### 📁 Files Required:
        - `topic_detection_model.h5`
        - `tokenizer.pkl`
        - `label_encoder.pkl`

        #### 🧑‍🎓 Developer:
        - **Your Name**
        - [GitHub Profile](https://github.com/yourusername)

        ---
        Made with ❤️ and Deep Learning.
    """)
