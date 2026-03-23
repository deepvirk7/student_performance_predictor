import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load('student_model.pkl')

model = load_model()

# --- UI HEADER ---
st.title("🎓 Student Performance Predictor")
st.markdown("""
Enter the student details below to predict the **Overall Score**. 
This model uses an optimized Ridge Regression pipeline.
""")

# --- INPUT UI ---
st.sidebar.header("Student Information")

# Categorical Inputs
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
school_type = st.sidebar.selectbox("School Type", ["Public", "Private"])
parent_edu = st.sidebar.selectbox("Parent Education", ["High School", "College", "University", "None"])
internet = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
travel_time = st.sidebar.selectbox("Travel Time", ["Low", "Medium", "High"])
extra_act = st.sidebar.selectbox("Extra Activities", ["Yes", "No"])
study_method = st.sidebar.selectbox("Study Method", ["Individual", "Group", "Tutor"])

# Numeric Inputs
st.subheader("Academic Metrics")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=25, value=16)
    study_hours = st.slider("Weekly Study Hours", 0, 50, 15)
    attendance = st.slider("Attendance %", 0, 100, 90)

with col2:
    math = st.number_input("Math Score", 0, 100, 75)
    science = st.number_input("Science Score", 0, 100, 75)
    english = st.number_input("English Score", 0, 100, 75)

# --- PREDICTION LOGIC ---
if st.button("Predict Performance"):
    # 1. Create Raw DataFrame (Must match the columns used in X_train)
    input_data = pd.DataFrame({
        'age': [age],
        'study_hours': [study_hours],
        'attendance_percentage': [attendance],
        'math_score': [math],
        'science_score': [science],
        'english_score': [english],
        'gender': [gender],
        'school_type': [school_type],
        'parent_education': [parent_edu],
        'internet_access': [internet],
        'travel_time': [travel_time],
        'extra_activities': [extra_act],
        'study_method': [study_method]
    })

    # 2. Replicate Feature Engineering (Matches your training script)
    input_data['study_efficiency'] = input_data['study_hours'] * input_data['attendance_percentage'] / 100
    input_data['avg_subject_score'] = (input_data['math_score'] + input_data['science_score'] + input_data['english_score']) / 3
    input_data['score_std'] = input_data[['math_score', 'science_score', 'english_score']].std(axis=1)

    # 3. Prediction
    # The pipeline handles scaling and one-hot encoding automatically
    prediction = model.predict(input_data)[0]
    
    # 4. Display Results
    st.divider()
    st.balloons()
    
    col_res1, col_res2 = st.columns(2)
    col_res1.metric("Predicted Overall Score", f"{prediction:.2f}/100")
    
    # Simple advice logic based on efficiency
    efficiency = input_data['study_efficiency'].values[0]
    if efficiency < 10:
        st.warning("⚠️ Recommendation: Increasing study hours or attendance could significantly boost the score.")
    else:
        st.success("✅ Student shows strong study consistency!")

# --- FOOTER ---
st.info("Note: This prediction is based on historical data trends and model training.")