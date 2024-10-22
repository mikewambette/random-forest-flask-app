from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the saved Random Forest model
model = joblib.load('random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json(force=True)
        features = np.array([data['features']])  # Assuming features is a list

        # Make prediction using the model
        prediction = model.predict(features)

        # Return the result as a JSON response
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        # Log the error and return it as a response
        app.logger.error(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
