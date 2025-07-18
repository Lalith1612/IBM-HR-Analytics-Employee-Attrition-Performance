import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- HOW TO RUN ---
# 1. Make sure you have the following files in the same directory:
#    - app.py (this file)
#    - best_hr_attrition_model.pkl (the trained model file)
# 2. Open your terminal or command prompt.
# 3. Navigate to the directory containing these files.
# 4. Run the command: streamlit run app.py

# --- Load The Model ---
try:
    with open('best_hr_attrition_model.pkl', 'rb') as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'best_hr_attrition_model.pkl' not found.")
    st.info("Please make sure the trained model file is in the same directory as this script.")
    st.stop()


# --- Page Configuration ---
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="�",
    layout="wide"
)


# --- App Title and Description ---
st.title("HR Employee Attrition Prediction")
st.markdown("""
This application predicts whether an employee is likely to leave the company (attrition).
Please provide the employee's details in the sidebar on the left. The prediction will be displayed below.
The model used is a **Logistic Regression** classifier, identified as the best performer by Optuna.
""")


# --- Sidebar For User Input ---
st.sidebar.header("Employee Details")

def user_input_features():
    """
    Creates sidebar widgets to collect user input for all necessary features.
    """
    # Define options for categorical features based on the original dataset
    business_travel_options = ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
    department_options = ['Sales', 'Research & Development', 'Human Resources']
    education_field_options = ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources']
    gender_options = ['Male', 'Female']
    job_role_options = [
        'Sales Executive', 'Research Scientist', 'Laboratory Technician',
        'Manufacturing Director', 'Healthcare Representative', 'Manager',
        'Sales Representative', 'Research Director', 'Human Resources'
    ]
    marital_status_options = ['Married', 'Single', 'Divorced']
    overtime_options = ['No', 'Yes']

    # --- Numerical Inputs ---
    age = st.sidebar.slider('Age', 18, 60, 35)
    daily_rate = st.sidebar.slider('Daily Rate', 100, 1500, 800)
    distance_from_home = st.sidebar.slider('Distance From Home (km)', 1, 30, 10)
    education = st.sidebar.selectbox('Education Level', [1, 2, 3, 4, 5], index=2)
    environment_satisfaction = st.sidebar.selectbox('Environment Satisfaction', [1, 2, 3, 4], index=2)
    hourly_rate = st.sidebar.slider('Hourly Rate', 30, 100, 65)
    job_involvement = st.sidebar.selectbox('Job Involvement', [1, 2, 3, 4], index=2)
    job_level = st.sidebar.selectbox('Job Level', [1, 2, 3, 4, 5], index=1)
    job_satisfaction = st.sidebar.selectbox('Job Satisfaction', [1, 2, 3, 4], index=3)
    monthly_income = st.sidebar.slider('Monthly Income', 1000, 20000, 5000)
    monthly_rate = st.sidebar.slider('Monthly Rate', 2000, 27000, 14000)
    num_companies_worked = st.sidebar.slider('Number of Companies Worked', 0, 9, 1)
    percent_salary_hike = st.sidebar.slider('Percent Salary Hike', 11, 25, 15)
    performance_rating = st.sidebar.selectbox('Performance Rating', [3, 4], index=0)
    relationship_satisfaction = st.sidebar.selectbox('Relationship Satisfaction', [1, 2, 3, 4], index=2)
    stock_option_level = st.sidebar.selectbox('Stock Option Level', [0, 1, 2, 3], index=0)
    total_working_years = st.sidebar.slider('Total Working Years', 0, 40, 10)
    training_times_last_year = st.sidebar.slider('Training Times Last Year', 0, 6, 3)
    work_life_balance = st.sidebar.selectbox('Work Life Balance', [1, 2, 3, 4], index=1)
    years_at_company = st.sidebar.slider('Years At Company', 0, 40, 5)
    years_in_current_role = st.sidebar.slider('Years In Current Role', 0, 18, 4)
    years_since_last_promotion = st.sidebar.slider('Years Since Last Promotion', 0, 15, 2)
    years_with_curr_manager = st.sidebar.slider('Years With Current Manager', 0, 17, 4)

    # --- Categorical Inputs ---
    business_travel = st.sidebar.selectbox('Business Travel', business_travel_options)
    department = st.sidebar.selectbox('Department', department_options)
    education_field = st.sidebar.selectbox('Education Field', education_field_options)
    gender = st.sidebar.selectbox('Gender', gender_options)
    job_role = st.sidebar.selectbox('Job Role', job_role_options)
    marital_status = st.sidebar.selectbox('Marital Status', marital_status_options)
    over_time = st.sidebar.selectbox('OverTime', overtime_options)


    # Create a dictionary of the input data
    data = {
        'Age': age,
        'DailyRate': daily_rate,
        'DistanceFromHome': distance_from_home,
        'Education': education,
        'EnvironmentSatisfaction': environment_satisfaction,
        'HourlyRate': hourly_rate,
        'JobInvolvement': job_involvement,
        'JobLevel': job_level,
        'JobSatisfaction': job_satisfaction,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': monthly_rate,
        'NumCompaniesWorked': num_companies_worked,
        'PercentSalaryHike': percent_salary_hike,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager,
        'BusinessTravel': business_travel,
        'Department': department,
        'EducationField': education_field,
        'Gender': gender,
        'JobRole': job_role,
        'MaritalStatus': marital_status,
        'OverTime': over_time
    }

    # Convert the dictionary to a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()


# --- Main Panel for Displaying Input and Prediction ---
st.subheader("Employee's Input Details")
st.write(input_df)

# Prediction button
if st.button('Predict Attrition'):
    # Make prediction using the loaded pipeline
    prediction = pipeline.predict(input_df)
    prediction_proba = pipeline.predict_proba(input_df)

    st.subheader('Prediction Result')

    # Display the prediction
    if prediction[0] == 1:
        st.error('**Prediction: Attrition** (The employee is likely to leave)')
    else:
        st.success('**Prediction: No Attrition** (The employee is likely to stay)')

    # Display the prediction probability
    st.subheader('Prediction Probability')
    proba_df = pd.DataFrame(
        prediction_proba,
        columns=['Probability (No Attrition)', 'Probability (Attrition)'],
        index=['Probability']
    )
    st.write(proba_df)

    # Explain the probability
    st.info(f"The model is **{prediction_proba.max()*100:.2f}%** confident in its prediction.")
