# Car-Insurance-Claim-Prediction-Model
**Project Type:** Personal Data Science Project<br>
**Status:** Completed (October 2025)<br>
**Skills Demonstrated:** Data Cleaning, Machine Learning, Python<br>
**Note:** Data and business scenario from DataCamp [https://projects.datacamp.com/projects/1645]

# Project Overview
Insurance companies invest significant time and money into optimizing their pricing strategies and accurately estimating the likelihood that customers will file a claim. Because car insurance is legally required for most drivers, this is a vast and competitive market. On the Road Car Insurance has requested a predictive model to estimate whether a customer will make a claim during their policy period. As the company currently lacks machine learning infrastructure, they’ve asked for a simple, interpretable solution — specifically, to identify the single most predictive feature that results in the best-performing model (based on accuracy).

# Objectives
- Explore the car insurance dataset (car_insurance.csv) and understand its structure
- Build multiple predictive models to classify whether a customer will make a claim
- Evaluate each model using accuracy as the performance metric
- Identify the single feature that produces the best-performing model
- Recommend a simple, production-ready model based on that feature

# Machine Learning Problem Definition
Type: Binary Classification
Goal: Predict if a customer will make a claim (Yes/No) during their policy period
Target Variable: outcome

# Methodology
1. Data Exploration
- Loaded the dataset and performed an initial inspection for nulls, data types, and distribution
- Analyzed class imbalance for the target variable
- Created visualizations to understand feature relationships

2. Data Preprocessing
- Handled missing values and categorical encoding
- Undersampled for better class balance
- Split data into train and test sets

3. Feature Evaluation
- Trained multiple single-feature models, each using one column at a time - Logistic Regression, Decision Tree, Random Forest, Support Vector Machine
- Computed accuracy for each model on the test set

4. Model Selection
- Compared model performance across single features
- Identified the feature that produced the highest test accuracy
- Reported that feature as the most predictive candidate for initial production deployment

# Results
- All models showed 'driving_experience' variable to be best single feature for predicting claim outcome
- LogisticRegression: driving_experience (Accuracy: 0.7560)
- DecisionTreeClassifier: driving_experience (Accuracy: 0.7616)
- RandomForestClassifier: driving_experience (Accuracy: 0.7544)
- Standard Vector Machine: driving_experience (Accuracy: 0.7671)

#  Deliverables
- Notebook / Script: Modeling_Car_Insurance_Claims_Outcomes.ipynb / Modeling_Car_Insurance_Claims_Outcomes.py