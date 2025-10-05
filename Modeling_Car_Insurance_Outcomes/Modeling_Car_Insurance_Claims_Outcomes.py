# https://projects.datacamp.com/projects/1645

# Import required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Import CSV Data
df = pd.read_csv('car_insurance.csv')

# Review Data
# df.info() # Get a summary of the DataFrame, including data types and non-null values
# df.head() # View the first few rows of the DataFrame
# df.describe() # Get descriptive statistics for numerical columns

# -- Check for Imbalanced Data
class_counts = df['outcome'].value_counts()
class_proportions = df['outcome'].value_counts(normalize=True) * 100
# print("Class Counts:\n", class_counts)
# print("\nClass Proportions (%):\n", class_proportions)
# plt.figure(figsize=(8, 6))
# sns.countplot(x='outcome', data=df)
# plt.title('Distribution of Outcome Classes')
# plt.xlabel('Class')
# plt.ylabel('Count')
# plt.show()

# -- Plot Data
# for col in df.columns:
#     if is_numeric_dtype(df[col]):
#         plt.hist(df[[col]])
#         plt.xlabel(col)
#         plt.ylabel("Frequency")
#         plt.show()


# -- Objects: driving_experience, education, income, vehicle_year, vehicle_type
# -- Missing Data: credit_score, anual_mileage

# Clean / Transform Data
# -- Convert Objects to Numbers
# -- Convert object type to a categorical type
df['driving_experience'] = df['driving_experience'].astype('category')
df['education'] = df['education'].astype('category')
df['income'] = df['income'].astype('category')
df['vehicle_year'] = df['vehicle_year'].astype('category')
df['vehicle_type'] = df['vehicle_type'].astype('category')
# -- Convert the categorical column to numerical codes
df['driving_experience'] = df['driving_experience'].cat.codes
df['education'] = df['education'].cat.codes
df['income'] = df['income'].cat.codes
df['vehicle_year'] = df['vehicle_year'].cat.codes
df['vehicle_type'] = df['vehicle_type'].cat.codes

# -- Fill missing values with a specific value (e.g., mean, median, or a constant)
df['credit_score'].fillna(df['credit_score'].mean(), inplace=True)
df['annual_mileage'].fillna(df['annual_mileage'].mean(), inplace=True)

# -- Handle Duplicates (None)
# df.drop_duplicates(inplace=True)

# -- Review Data
# df.info() # Get a summary of the DataFrame, including data types and non-null values
# df.head() # View the first few rows of the DataFrame
# df.describe() # Get descriptive statistics for numerical columns

# -- Plot Data
# for col in df.columns:
#     plt.hist(df[[col]])
#     plt.xlabel(col)
#     plt.ylabel("Frequency")
#     plt.show()

# Identify best feature, as measured by accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def find_best_feature(df, model, scale=False, target='outcome', show_confusion=True):

    feature_metrics = []

    # Reuse the same train/test split for fairness across features
    X_full = df.drop(columns=[target,'id'])
    y = df[target]
    # print("Original class distribution:", Counter(y))
    
    # Undersampling using RandomUnderSampler
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_full, y = undersample.fit_resample(X_full, y)
    # print("Undersampled class distribution:", Counter(y))

    # # Oversampling using RandomOverSampler - Yields Same Overall Results as Undersample
    # oversample = RandomOverSampler(sampling_strategy='minority')
    # X_full, y = oversample.fit_resample(X_full, y)
    # print("Oversampled class distribution:", Counter(y))
    
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42)

    for col in X_full.columns:
        X_train = X_train_full[[col]].copy()
        X_test = X_test_full[[col]].copy()

        # Optional scaling
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Collect metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        unique_preds = np.unique(y_pred, return_counts=True)

        feature_metrics.append({
            'Feature': col,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'Predicted_Classes': dict(zip(unique_preds[0], unique_preds[1]))
        })

    # Convert to DataFrame for readability
    results = pd.DataFrame(feature_metrics).sort_values(by='Accuracy', ascending=False)
    best_feature = results.iloc[0]['Feature']
    best_acc = results.iloc[0]['Accuracy']

    print(f"\n Best Feature: {best_feature} (Accuracy: {best_acc:.4f})")
    print("\n Summary of All Features:\n")
    display(results)

    # Optional confusion matrix for best feature
    if show_confusion:
        print(f"\n🔍 Confusion Matrix for Best Feature: {best_feature}")
        X_train = X_train_full[[best_feature]]
        X_test = X_test_full[[best_feature]]

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix – {best_feature}')
        plt.show()

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))


# -- LogisticRegression
print("Logisitic Regression")
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear', random_state=42)      
find_best_feature(df, model, scale=False, target='outcome', show_confusion=False)

# -- DecisionTreeClassifier
print("Decision Tree Classifier")
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='gini', max_depth=10)
find_best_feature(df, model, scale=False, target='outcome', show_confusion=False)

# -- RandomForestClassifier
print("Random Forest Classifier")
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
find_best_feature(df, model, scale=False, target='outcome', show_confusion=False)

# -- Standard Vector Machine
print("Standard Vector Machine Classifier")
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1.0, random_state=42) # Linear kernel for binary class
find_best_feature(df, model, scale=True, target='outcome', show_confusion=False)

# Results
# -- Original class distribution: Counter({0.0: 6867, 1.0: 3133})
# -- Undersampled class distribution: Counter({0.0: 3133, 1.0: 3133})
# -- LogisticRegression: driving_experience (Accuracy: 0.7560)
# -- DecisionTreeClassifier: driving_experience (Accuracy: 0.7616)
# -- RandomForestClassifier: driving_experience (Accuracy: 0.7544)
# -- Standard Vector Machine: driving_experience (Accuracy: 0.7671)