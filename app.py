# ==============================
# IMPORT LIBRARIES
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ==============================
# LOAD MODEL
# ==============================
model = pickle.load(open("student_performance_model.pkl", "rb"))

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# ==============================
# TITLE
# ==============================
st.title("🎓 Student Performance Predictor")
st.write("Predict student performance using Multiple Linear Regression")

# ==============================
# USER INPUTS
# ==============================
st.sidebar.header("Enter Student Details")

age = st.sidebar.slider("Age", 10, 25, 18)
study_hours = st.sidebar.slider("Study Hours per Day", 0.0, 12.0, 5.0)
attendance = st.sidebar.slider("Attendance (%)", 0.0, 100.0, 75.0)
math_score = st.sidebar.slider("Math Score", 0.0, 100.0, 60.0)
science_score = st.sidebar.slider("Science Score", 0.0, 100.0, 60.0)
english_score = st.sidebar.slider("English Score", 0.0, 100.0, 60.0)

# ==============================
# INPUT DATAFRAME
# ==============================
input_data = pd.DataFrame({
    "age": [age],
    "study_hours": [study_hours],
    "attendance_percentage": [attendance],
    "math_score": [math_score],
    "science_score": [science_score],
    "english_score": [english_score]
})

st.subheader("📥 Input Data")
st.write(input_data)

# ==============================
# PREDICTION
# ==============================
if st.button("Predict Performance"):
    try:
        prediction = model.predict(input_data)
        st.success(f"🎯 Predicted Performance Score: {prediction[0]:.2f}")
    except Exception as e:
        st.error("⚠️ Input features mismatch. Make sure model and inputs align.")

# ==============================
# OPTIONAL: SHOW METRICS (STATIC)
# ==============================
st.subheader("📊 Model Info")
st.write("""
- Model: Multiple Linear Regression  
- Metrics:
  - R² Score ≈ 0.96  
  - MSE ≈ 0.0037  
  - MAE ≈ very low  
""")

# ==============================
# OPTIONAL: SAMPLE GRAPH
# ==============================
st.subheader("📈 Sample Visualization")

x = np.linspace(0, 10, 50)
y = x + np.random.randn(50)

plt.figure()
plt.scatter(x, y)
plt.xlabel("Study Hours")
plt.ylabel("Performance")
plt.title("Sample Trend")

st.pyplot(plt)
