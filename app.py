# ==============================
# IMPORTS
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================
# LOAD MODEL & COLUMNS
# ==============================
model = pickle.load(open("student_performance_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="Student Performance Predictor")

# ==============================
# TITLE
# ==============================
st.title("🎓 Student Performance Predictor")

# ==============================
# INPUTS
# ==============================
st.sidebar.header("Enter Details")

age = st.sidebar.slider("Age", 10, 25, 18)
study_hours = st.sidebar.slider("Study Hours", 0.0, 12.0, 5.0)
attendance = st.sidebar.slider("Attendance %", 0.0, 100.0, 75.0)
math_score = st.sidebar.slider("Math Score", 0.0, 100.0, 60.0)
science_score = st.sidebar.slider("Science Score", 0.0, 100.0, 60.0)
english_score = st.sidebar.slider("English Score", 0.0, 100.0, 60.0)

# ==============================
# CREATE INPUT
# ==============================
input_data = pd.DataFrame({
    "age": [age],
    "study_hours": [study_hours],
    "attendance_percentage": [attendance],
    "math_score": [math_score],
    "science_score": [science_score],
    "english_score": [english_score]
})

st.write("### 📥 Input Data")
st.write(input_data)

# ==============================
# PREDICTION
# ==============================
if st.button("Predict"):
    
    # Create full input with all columns
    input_full = pd.DataFrame(columns=columns)
    input_full.loc[0] = 0

    # Fill known values
    for col in input_data.columns:
        if col in input_full.columns:
            input_full[col] = input_data[col].values

    # Prediction
    prediction = model.predict(input_full)

    st.success(f"🎯 Predicted Performance: {prediction[0]:.2f}")

# ==============================
# INFO
# ==============================
st.write("### 📊 Model Info")
st.write("""
- Model: Multiple Linear Regression  
- R² Score ≈ 0.96  
- MSE ≈ 0.0037  
- MAE ≈ Low  
""")
