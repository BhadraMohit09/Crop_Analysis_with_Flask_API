# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('agriculture_dataset.csv')

# Display the first few rows of the dataset
data.head()

# Check for missing values
data.isnull().sum()

# Encode categorical variables
categorical_features = ['Crop_Type', 'Irrigation_Type', 'Soil_Type', 'Season']
numerical_features = ['Farm_Area(acres)', 'Fertilizer_Used(tons)', 'Pesticide_Used(kg)', 'Water_Usage(cubic meters)']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Define the target variable
X = data.drop('Yield(tons)', axis=1)
y = data['Yield(tons)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Support Vector Machine': SVR(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Neural Network': MLPRegressor(random_state=42)
}

# Train and evaluate the models
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MSE': mse, 'MAE': mae, 'R²': r2}

# Display the results
results_df = pd.DataFrame(results).T
results_df

# Visualize the results
plt.figure(figsize=(12, 8))
sns.barplot(x=results_df.index, y=results_df['R²'])
plt.title('R² Score Comparison of Different Models')
plt.ylabel('R² Score')
plt.xlabel('Model')
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x=results_df.index, y=results_df['MSE'])
plt.title('MSE Comparison of Different Models')
plt.ylabel('MSE')
plt.xlabel('Model')
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x=results_df.index, y=results_df['MAE'])
plt.title('MAE Comparison of Different Models')
plt.ylabel('MAE')
plt.xlabel('Model')
plt.show()