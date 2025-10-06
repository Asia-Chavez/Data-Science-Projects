# Car-Insurance-Claim-Prediction-Model
**Project Type:** Personal Machine  Learning Project<br>
**Status:** Completed (October 2025)<br>
**Skills Demonstrated:** Data Cleaning, Machine Learning, Python<br>
**Note:** Data and business scenario from DataCamp [https://projects.datacamp.com/projects/2264]

# Project Overview
Dive into the heart of data science with a project that merges healthcare insights and predictive analytics. As a Data Scientist at a leading Health Insurance company, my goal is to predict individual healthcare costs using customer demographic and lifestyle data.

This project demonstrates my ability to apply machine learning, feature engineering, and model evaluation techniques to real-world healthcare data.

# Objectives
- Build a regression model that predicts medical insurance charges for customers based on their profile.
- Once trained, the model is validated on new data (validation_dataset.csv) to assess its ability to generalize and predict unseen healthcare costs accurately.

# Machine Learning Problem Definition
- Type: Regression
- Goal: Predict individual healthcare costs using customer demographic and lifestyle data
- Target Variable: charges

# Methodology
1. Data Exploration
- Loaded the dataset and performed an initial inspection for nulls, data types, and distribution
- Handled missing values and categorical encoding
- Visualized distributions and relationships between features and medical costs

2. Feature Engineering
- Investigated correlations between features and the target variable (charges)
- Normalized continuous variables where needed

3. Modeling
- Explored multiple regression models:
    - Linear Regression – baseline interpretable model.
    - Random Forest Regressor – captures non-linear relationships.
    - Gradient Boosting Regressor – optimized for accuracy.
- Each model was evaluated using:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R² Score / Cross-Validation R²

4. Model Selection
After training on insurance.csv, the final model was applied to validation_dataset.csv to predict medical charges for unseen data.

# Results
- Gradient Boosting Regressor was best performing model
    - Root Mean Squared Error (RMSE): 4136.93
    - Mean Absolute Error (MAE): 2369.29
    - R²: 0.8790356697465435
    - Cross-Validation R²: 0.853
- Successfully predicted medical charges for unseen data

# Data Insights
- Medical charges for smokers are ~3.8x that of non-smokers
- Being a smoker is the most significant feature for predicting medical charges
- BMI and age are high on feature importance (higher BME and older age leads to higher medical charges)
- Sex, region, and children have much lower feature importance

#  Deliverables
- Notebook / Script: Predicting_Insurance_Charges.ipynb / Predicting_Insurance_Charges.py