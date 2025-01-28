import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd

# Preprocessing and model pipeline
categorical_features = ['Crop_Type', 'Irrigation_Type', 'Soil_Type', 'Season']
numerical_features = ['Farm_Area(acres)', 'Fertilizer_Used(tons)', 'Pesticide_Used(kg)', 'Water_Usage(cubic meters)']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Train the model (using Random Forest as an example)
model = RandomForestRegressor(random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Fit the pipeline on the entire dataset
data = pd.read_csv('agriculture_dataset.csv')
X = data.drop('Yield(tons)', axis=1)
y = data['Yield(tons)']
pipeline.fit(X, y)

# Save the model to a file
joblib.dump(pipeline, 'model.pkl')