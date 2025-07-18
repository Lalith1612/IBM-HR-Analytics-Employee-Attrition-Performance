# IBM-HR-Analytics-Employee-Attrition-Performance

1. Project Overview
This project aims to predict employee attrition within a company using machine learning. By analyzing various employee attributes, the model identifies key factors that contribute to an employee leaving their job. The final outcome is a web application where HR personnel can input an employee's details and receive a real-time prediction on their likelihood of attrition.

The project covers the entire machine learning lifecycle:

Data Loading and Initial Analysis

Data Cleaning and Preprocessing

In-depth Exploratory Data Analysis (EDA)

Automated Model Selection and Hyperparameter Tuning with Optuna

Model Evaluation and Interpretation

Saving the Best Model for Deployment

Building and Deploying a Web Application with Streamlit

2. Dataset
The project utilizes the IBM HR Analytics Employee Attrition & Performance dataset.

Source: Kaggle

Content: This dataset created by IBM data scientists, containing 1470 rows and 35 features describing various aspects of an employee's profile and work environment. The target variable is Attrition.

3. Project Structure
The repository for this project should contain the following files:

HR_Attrition_Prediction_Full_Project.ipynb: A Jupyter Notebook containing the complete, step-by-step code for data analysis, model training, and evaluation.

app.py: A Python script that uses Streamlit to create and run the interactive web application.

best_hr_attrition_model.pkl: The serialized, trained machine learning pipeline (preprocessor + model) saved using pickle.

requirements.txt: A text file listing all the Python dependencies required to run the project.

README.md: This file.

4. Methodology
Step 1: Data Analysis and Cleaning
The dataset was loaded and inspected for basic properties like shape, data types, and missing values. It was found to be clean with no missing data. Non-informative columns (EmployeeCount, StandardHours, etc.) were dropped.

Step 2: Exploratory Data Analysis (EDA)
A comprehensive EDA was performed to uncover insights and relationships within the data. Over 10 visualizations were generated, including:

A pie chart showing the class imbalance in the Attrition variable.

Count plots to analyze attrition across different Departments, Job Roles, and OverTime status.

Violin and box plots to understand the distribution of MonthlyIncome and JobSatisfaction for employees who left versus those who stayed.

A correlation heatmap to identify relationships between numerical features.

Key Insight: Attrition was found to be higher among employees who work overtime, have lower monthly incomes, are younger, and travel frequently for business.

Step 3: Model Selection with Optuna
To find the best-performing algorithm, Optuna was used to automate hyperparameter tuning for three candidate models: Logistic Regression, Random Forest, and Gradient Boosting. The models were evaluated using a 5-fold cross-validation on the F1-score, which is suitable for imbalanced datasets.

Result: Logistic Regression emerged as the top-performing model, indicating that the relationships in the data are largely linear.

1. LogisticRegression: F1-score = 0.8729

2. GradientBoosting: F1-score = 0.8666

3. RandomForest: F1-score = 0.8283

Step 4: Model Evaluation
The top model, Logistic Regression, was trained on the full training set and evaluated on the held-out test set. Its performance was assessed using:

Accuracy, Precision, Recall, and F1-Score

Classification Report

Confusion Matrix: To visualize true vs. predicted labels.

ROC Curve: To assess the model's ability to distinguish between classes.

The model demonstrated strong predictive performance on the unseen test data.

Step 5: Model Deployment
The entire pipeline, including the feature preprocessor (StandardScaler for numerical data, OneHotEncoder for categorical data) and the trained Logistic Regression model, was saved into a single best_hr_attrition_model.pkl file.

A user-friendly web application was built using Streamlit. This app allows a user to input an employee's details through a simple UI and get an instant prediction of attrition risk and a confidence score.
