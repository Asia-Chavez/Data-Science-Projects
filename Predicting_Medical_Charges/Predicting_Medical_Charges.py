# https://projects.datacamp.com/projects/2264

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load Data
insurance_data_path = 'insurance.csv'
df = pd.read_csv(insurance_data_path)
#df.info() # Get a summary of the DataFrame, including data types and non-null values
#df.head() # View the first few rows of the DataFrame
#df.describe() # Get descriptive statistics for numerical columns

# Review Data
#import seaborn as sns
#sns.boxplot(x=df['age'])
# Outlier detection using histogram
#import matplotlib.pyplot as plt
#plt.hist(df.age)
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.title("Histogram")
#plt.show()

# Clean Data
# -- Drop rows with any missing values
#print(df.isnull().sum())
df.dropna(inplace=True) 
#print(df.isnull().sum())
#df.info() # Get a summary of the DataFrame, including data types and non-null values
#df.head() # View the first few rows of the DataFrame
#df.describe() # Get descriptive statistics for numerical columns

# -- Clean Text / Object Data
df['sex'] = df['sex'].str.lower() # Convert to lowercase
df['smoker'] = df['smoker'].str.lower() # Convert to lowercase
df['region'] = df['region'].str.lower() # Convert to lowercase

# -- Convert Object Types to Numeric
#df['sex'].value_counts().plot(kind='bar')
#plt.title('Frequency of Categories (Matplotlib)')
#plt.xlabel('sex')
#plt.ylabel('Count')
#plt.show()
# Update sex for consistencym, woman, man, f
df['sex'] = df['sex'].replace({'m': 'male', 'man': 'male', 'f': 'female', 'woman': 'female'})
sex_mapping = {'male': 0, 'female': 1}
df['sex'] = df['sex'].replace(sex_mapping)

smoker_mapping = {'yes': 0, 'no': 1}
df['smoker'] = df['smoker'].replace(smoker_mapping)

region_mapping = {'southwest': 0, 'southeast': 1,'northwest': 2, 'northeast': 3}
df['region'] = df['region'].replace(region_mapping)

# -- Convert charges to Numeric
df['charges'] = df['charges'].str.replace('$', '', regex=True) # Remove special characters
df['charges'] = df['charges'].str.replace(' ', '', regex=True) # Remove special characters
df['charges'] = pd.to_numeric(df['charges'], errors='coerce') # 'coerce' turns invalid into NaN
df['charges'] = df['charges'].round(2)

# -- Fix Negative Values
df['children'] = df['children'].abs()
df['age'] = df['age'].abs()

# -- Handle Duplicates
df.drop_duplicates(inplace=True)

#print(df.isnull().sum())
df.dropna(inplace=True) 

# -- Scale Continous Data
continuous_cols = ['age', 'bmi']
scaler = StandardScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

# -- Final Data Review
#print(df.isnull().sum())
#df.info() # Get a summary of the DataFrame, including data types and non-null values
#df.head() # View the first few rows of the DataFrame
#df.describe() # Get descriptive statistics for numerical columns

# -- Data Analysis
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
# ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
# Age - Ages: 18-64 Create: 4 equal-sized quantiles (e.g., 'Young', 'Adult', 'Middle-aged', 'Senior')
df['age_group'] = pd.qcut(df['age'], q=4, labels=['Young', 'Adult', 'Middle-aged', 'Senior'])
avg_by_age = df.groupby('age_group')['charges'].mean()
print("\nAverage Charges by Age:\n", avg_by_age)
sns.boxplot(x='age_group', y='charges', data=df)
plt.title("Costs by Age Group")
plt.show()

# Compare Sex
avg_by_sex = df.groupby('sex')['charges'].mean()
print("\nAverage Charges by Sex:\n", avg_by_sex)
sns.boxplot(x='sex', y='charges', data=df)
plt.title("Costs by Sex")
plt.show()

# BMI
df['bmi_group'] = pd.qcut(df['bmi'], q=3, labels=['Low', 'Mid', 'High'])
avg_by_bmi = df.groupby('bmi_group')['charges'].mean()
print("\nAverage Charges by BMI:\n", avg_by_bmi)
sns.boxplot(x='bmi_group', y='charges', data=df)
plt.title("Costs by BMI Group")
plt.show()

# Average predicted cost per number of children
avg_by_children = df.groupby('children')['charges'].mean()
print("Average Charges by Children:\n", avg_by_children)
sns.boxplot(x='children', y='charges', data=df)
plt.title("Costs by Children")
plt.show()

# Compare smokers vs non-smokers
avg_by_smoker = df.groupby('smoker')['charges'].mean()
print("\nAverage Charges by Smoking Status:\n", avg_by_smoker)
sns.boxplot(x='smoker', y='charges', data=df)
plt.title("Costs by Smoker")
plt.show()

# Average predicted cost per region
avg_by_region = df.groupby('region')['charges'].mean().sort_values(ascending=False)
print("Average Charges by Region:\n", avg_by_region)
sns.boxplot(x='region', y='charges', data=df)
plt.title("Costs by Region")
plt.show()

df.info() # Get a summary of the DataFrame, including data types and non-null values
df.head() # View the first few rows of the DataFrame
df.describe() # Get descriptive statistics for numerical columns


# Model Development
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
# Define independent and dependent variables
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
print("Linear Regression")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
# -- Make predictions
y_pred = model.predict(X_test)
# -- Evaluate the model
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R²: {r2}") # Goal > 0.65
print(f"Cross-Validation R²: {cv_scores.mean():.3f}")

# Random Forest
print("\nRandom Forest Regressor")
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# -- Make predictions
y_pred = model.predict(X_test)
# -- Evaluate the model
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R²: {r2}") # Goal > 0.65
print(f"Cross-Validation R²: {cv_scores.mean():.3f}")

# Gradient Boosting
print("\nGradient Boosting Regressor")
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)
# -- Make predictions
y_pred = model.predict(X_test)
# -- Evaluate the model
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R²: {r2}") # Goal > 0.65
print(f"Cross-Validation R²: {cv_scores.mean():.3f}")

# Get feature importances
importances = model.feature_importances_
# Create a pandas Series for easier plotting
feature_names = X.columns # If X is a DataFrame
feature_importance_series = pd.Series(importances, index=feature_names)
# Sort the features by importance
feature_importance_series = feature_importance_series.sort_values(ascending=False)
# Plot the feature importances
plt.figure(figsize=(10, 6))
feature_importance_series.plot(kind='barh')
plt.title('Feature Importance for Gradent Boost')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Validation of Unseen Data
validation_data_path = 'validation_dataset.csv'
validation_data = pd.read_csv(validation_data_path)

# Review Data
#print(validation_data.isnull().sum())
#validation_data.info() # Get a summary of the DataFrame, including data types and non-null values
#validation_data.head() # View the first few rows of the DataFrame
#validation_data.describe() # Get descriptive statistics for numerical columns

# Update sex for consistencym, woman, man, f
validation_data['sex'] = validation_data['sex'].replace({'m': 'male', 'man': 'male', 'f': 'female', 'woman': 'female'})
sex_mapping = {'male': 0, 'female': 1}
validation_data['sex'] = validation_data['sex'].replace(sex_mapping)
smoker_mapping = {'yes': 0, 'no': 1}
validation_data['smoker'] = validation_data['smoker'].replace(smoker_mapping)
region_mapping = {'southwest': 0, 'southeast': 1,'northwest': 2, 'northeast': 3}
validation_data['region'] = validation_data['region'].replace(region_mapping)

# Scale Data
validation_data[continuous_cols] = scaler.transform(validation_data[continuous_cols])

# Predict Using Best Model - Gradient Boosting 
validation_data['predicted_charges'] = model.predict(validation_data)
validation_data.loc[validation_data['predicted_charges'] < 1000, 'predicted_charges'] = 1000

# -- Do not have actual charges for review, data is unseen in datacamp workbook