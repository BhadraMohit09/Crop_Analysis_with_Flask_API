from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    farm_area = float(request.form['farm_area'])
    fertilizer_used = float(request.form['fertilizer_used'])
    pesticide_used = float(request.form['pesticide_used'])
    water_usage = float(request.form['water_usage'])
    crop_type = request.form['crop_type']
    irrigation_type = request.form['irrigation_type']
    soil_type = request.form['soil_type']
    season = request.form['season']

    # Create a DataFrame from the input
    input_data = pd.DataFrame({
        'Farm_Area(acres)': [farm_area],
        'Fertilizer_Used(tons)': [fertilizer_used],
        'Pesticide_Used(kg)': [pesticide_used],
        'Water_Usage(cubic meters)': [water_usage],
        'Crop_Type': [crop_type],
        'Irrigation_Type': [irrigation_type],
        'Soil_Type': [soil_type],
        'Season': [season]
    })

    # Make a prediction
    prediction = model.predict(input_data)[0]

    # Render the result template with the prediction
    return render_template('index.html', prediction_text=f'Predicted Yield: {prediction:.2f} tons')

if __name__ == '__main__':
    app.run(debug=True)