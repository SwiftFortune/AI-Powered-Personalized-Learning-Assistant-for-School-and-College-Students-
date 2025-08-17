import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load trained models
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
cluster_labels = joblib.load('cluster_labels.pkl')

# Features used in clustering
features = [
    'time_spent_weekly',
    'quiz_score_avg',
    'forum_posts',
    'video_watched_percent',
    'math score',
    'session_duration_avg'
]

# Streamlit UI Setup
st.set_page_config(page_title="🎓 Student Learning Cluster App", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🚀 Navigation")
menu = ["📘 Introduction", "🔧 Pipeline", "📊 Prediction", "📈 Visualization", "🙋‍♂️ About Me"]
choice = st.sidebar.radio("Go to:", menu)

# 1. Introduction
if choice == "📘 Introduction":
    st.title("🎓 Student Learning Cluster Project")
    st.markdown("""
    Welcome to the **Student Learning Cluster App**! This project uses **unsupervised machine learning** to group students into:

    - 🧠 **Visual Learners**: Engage best with videos
    - 🐢 **Slow Learners**: Require extra time and support
    - ⚡ **Fast Responders**: Learn quickly and perform efficiently

    🎯 The goal is to help educators **personalize teaching** based on student learning patterns.

    🔍 Techniques used: **K-Means**, **PCA**, **StandardScaler**, **Silhouette Score**
    """)
    st.image("https://cdn.dribbble.com/users/1304671/screenshots/14930102/media/740ff299fe93f51e01d0d777d0eb82b1.png", use_column_width=True)

# 2. Pipeline
elif choice == "🔧 Pipeline":
    st.title("🛠️ Data Science Pipeline")
    st.markdown("""
    ### 🧪 Steps followed:

    1. 📥 Combined 3 datasets on student learning and performance
    2. 🧹 Filled missing values (mean/mode)
    3. 🏷️ Encoded categorical variables
    4. 📊 Outlier detection using boxplots
    5. 🔍 Selected key features: engagement + scores
    6. ⚖️ Standardized the features
    7. 📉 Reduced dimensionality with PCA
    8. 🤖 Clustered students using K-Means
    9. 🧪 Evaluated with Silhouette Score
    10. 💾 Saved models using `joblib`
    """)
    st.image("https://miro.medium.com/v2/resize:fit:1200/format:webp/1*DeqS6Crh8qM8C3ZbLnTj8g.png", width=700)

# 3. Prediction
elif choice == "📊 Prediction":
    st.title("📊 Predict Student Learning Type")
    st.markdown("Fill in the student's details to get the predicted learning type:")

    input_data = {}
    col1, col2 = st.columns(2)
    with col1:
        input_data['time_spent_weekly'] = st.number_input("🕓 Time Spent Weekly (mins)", min_value=0.0)
        input_data['quiz_score_avg'] = st.number_input("📝 Quiz Score Avg", min_value=0.0)
        input_data['forum_posts'] = st.number_input("💬 Forum Posts", min_value=0.0)
    with col2:
        input_data['video_watched_percent'] = st.number_input("🎥 Video Watched (%)", min_value=0.0)
        input_data['math score'] = st.number_input("➗ Math Score", min_value=0.0)
        input_data['session_duration_avg'] = st.number_input("🖥️ Session Duration Avg (mins)", min_value=0.0)

    if st.button("🔍 Predict Learner Type"):
        new_data = pd.DataFrame([input_data])
        st.subheader("📄 Input Summary")
        st.table(new_data)

        new_scaled = scaler.transform(new_data)
        new_pca = pca.transform(new_scaled)
        cluster = kmeans.predict(new_pca)[0]
        label = cluster_labels.get(cluster, "Unknown")

        if label == "Unknown":
            st.warning("⚠️ Prediction returned an unknown cluster. Check if cluster_labels.pkl matches model.")
        else:
            st.success(f"🎯 This student is likely a: **{label}**")

            df_plot = pd.read_csv("merged_output.csv")
            df_plot = df_plot.fillna(df_plot.mean(numeric_only=True))
            df_plot_scaled = scaler.transform(df_plot[features])
            df_plot_pca = pca.transform(df_plot_scaled)
            df_plot['Cluster'] = kmeans.predict(df_plot_pca)
            cluster_means = df_plot.groupby('Cluster')[features].mean().round(2)

            st.subheader("📊 Why this prediction?")
            st.markdown("This chart shows how the input compares to each cluster's average:")

            radar_df = cluster_means.T.copy()
            radar_df['Student'] = new_data.values[0]
            radar_df = radar_df.reset_index().rename(columns={'index': 'Feature'})
            radar_df = pd.melt(radar_df, id_vars='Feature', var_name='Group', value_name='Value')

            fig = px.line_polar(radar_df, r='Value', theta='Feature', color='Group', line_close=True,
                                title="🧭 Radar Chart: Student vs Cluster Averages")
            fig.update_traces(fill='toself')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📌 Cluster Feature Averages")
            st.dataframe(cluster_means.style.highlight_max(axis=0), use_container_width=True)

# 4. Visualization
elif choice == "📈 Visualization":
    st.title("📈 PCA Cluster Visualization")
    st.markdown("This scatter plot shows how students are grouped into clusters using PCA:")

    df_plot = pd.read_csv("merged_output.csv")
    df_plot = df_plot.fillna(df_plot.mean(numeric_only=True))
    df_plot_scaled = scaler.transform(df_plot[features])
    df_plot_pca = pca.transform(df_plot_scaled)
    df_plot['PCA1'] = df_plot_pca[:, 0]
    df_plot['PCA2'] = df_plot_pca[:, 1]
    df_plot['Cluster'] = kmeans.predict(df_plot_pca)
    df_plot['Learner_Type'] = df_plot['Cluster'].map(cluster_labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_plot, x="PCA1", y="PCA2", hue="Learner_Type", palette="Set2", s=60, ax=ax)
    plt.title("PCA 2D Scatter Plot of Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    st.pyplot(fig)

# 5. About Me
elif choice == "🙋‍♂️ About Me":
    st.title("🙋‍♂️ Meet the Developer")
    st.markdown("""
    👋 Hello! I'm **Sachin Hembram**, a passionate Data Science learner enthusiastic about building smart learning tools.

    💼 **Skills:**
    - Python, Pandas, NumPy
    - Scikit-learn, Streamlit
    - Data Visualization with Matplotlib, Seaborn, Plotly

    📫 **Contact:**
    - 📧 Email: sachincmf@gmail.com
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
    st.markdown("Thank you for visiting! Feel free to reach out with any questions or feedback.")   
