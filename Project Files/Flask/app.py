from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load your model and feature list
model = joblib.load("best_model_xgboost.pkl")
feature_names = joblib.load("model_features_list.pkl")

@app.route('/')
def home():
    return render_template('index.html')  # HTML should be inside 'templates/' folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        form_data = request.form
        input_data = {}

        # Get values from the form, convert to float where possible
        for feature in feature_names:
            value = form_data.get(feature, 0)
            try:
                input_data[feature] = float(value)
            except:
                input_data[feature] = value

        # Create a DataFrame with proper feature order
        df = pd.DataFrame([input_data], columns=feature_names)

        # Convert all columns to numeric (int/float), filling errors with 0
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        print(df.head())

        # Make prediction
        probability = model.predict_proba(df)[0][1]  # Probability of class 1 (disease)
        percent = round(probability * 100, 2)

        if percent >= 50:
            result = f"Liver Disease Detected with {percent}% confidence"
        else:
            result = f"No Liver Disease detected with {100 - percent}% confidence"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)