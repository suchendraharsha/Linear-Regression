from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import preprocess_data
import os
from scaler import save_scaler, load_scaler  # Import scaler functions

app = Flask(__name__)
CORS(app, resources={r"/predict_lr": {"origins": "http://localhost:3000"}})

# Attempt to load the pre-trained model and scaler first
LR_model = None
LR_scaler = None
try:
    LR_model = joblib.load('models/lr_model.pkl')
    LR_scaler = load_scaler()
    print("Logistic Regression model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading Logistic Regression model/scaler: {e}")

# Only train and save the model if loading fails
if LR_model is None or LR_scaler is None:
    try:
        diabet = pd.read_csv('hospital_readmissions.csv')
        X_train, X_test, y_train, y_test, scaler = preprocess_data(diabet.copy())

        params = {
            "penalty": 'l2',
            "C": 1.0,
            "solver": 'liblinear',
            "max_iter": 1000,
            "random_state": 123,
            'multi_class': 'auto'
        }

        LR = LogisticRegression(**params)
        LR.fit(X_train, y_train)

        save_scaler(scaler)  # Use the save_scaler function
        joblib.dump(LR, 'models/lr_model.pkl')
        print("Logistic Regression model and scaler trained and saved successfully.")
        LR_model = LR
        LR_scaler = scaler
    except Exception as e:
        print(f"Error during Logistic Regression model training/saving: {e}")

from preprocess_input import preprocess_input_lr  # Import specific input preprocessing

@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    if LR_scaler is None:
        return jsonify({'error': 'Scaler not loaded. Prediction cannot be made.'}), 500
    if LR_model is None:
        return jsonify({'error': 'Model not loaded. Prediction cannot be made.'}), 500
    try:
        data = request.get_json()
        processed_data = preprocess_input_lr(data, LR_scaler)
        prediction = LR_model.predict(processed_data)[0]
        return jsonify({'prediction_lr': str(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))