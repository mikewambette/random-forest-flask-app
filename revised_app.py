from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the saved Random Forest model
model = joblib.load('random_forest_model.pkl')

# Define a route for the home page
@app.route('/')
def home():
    return "Welcome to the Random Forest Model API!"

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request (expects JSON input)
    data = request.get_json(force=True)

    # Convert data into numpy array
    features = np.array([data['features']])

    # Make a prediction using the loaded model
    prediction = model.predict(features)

    # Return the prediction result as a JSON response
    return jsonify({'prediction': int(prediction[0])})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
