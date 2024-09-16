import numpy as np
from flask import Flask, request, jsonify
import joblib

# Load the trained models
uts_model = joblib.load('rf_uts_model.pkl')
elongation_model = joblib.load('rf_elongation_model.pkl')
conductivity_model = joblib.load('rf_conductivity_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the data from the React app
    
    # Extract features and model choice from request
    features = np.array(data['features']).reshape(1, -1)
    model_choice = data.get('model', 'uts')  # Default to neural network if no model is specified
    
    # Select the model based on user input
    if model_choice == 'uts':
        prediction = uts_model.predict(features)[0]
    elif model_choice == 'elongation':
        prediction = elongation_model.predict(features)[0]
    elif model_choice == 'conductivity':
        prediction = conductivity_model.predict(features)[0]
    else:
        return jsonify({'error': 'Invalid model choice'}), 400
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
