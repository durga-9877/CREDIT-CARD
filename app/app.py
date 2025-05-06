from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from ml.model import FraudDetector
import os
import logging
from hashlib import sha256
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = FraudDetector()

# Check if model exists, if not train it
MODEL_PATH = 'models/fraud_detector.joblib'
SCALER_PATH = 'models/scaler.joblib'

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    logger.info("Training new model...")
    try:
        # Load and preprocess data
        data = model.load_data('data/creditcard.csv')
        X_train_resampled, X_test_scaled, y_train_resampled, y_test = model.preprocess_data(data)
        
        # Train model
        model.train_model(X_train_resampled, y_train_resampled)
        
        # Evaluate model
        metrics = model.evaluate_model(X_test_scaled, y_test)
        logger.info(f"Model metrics: {metrics}")
        
        # Save model
        model.save_model(MODEL_PATH, SCALER_PATH)
        logger.info("Model trained and saved successfully")
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise
else:
    logger.info("Loading existing model...")
    model.load_saved_model(MODEL_PATH, SCALER_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        amount = float(data.get('amount', 0))
        
        # Get prediction from model
        try:
            prediction, probability = model.predict_single(amount)
        except Exception as e:
            logger.error(f"Model prediction error: {str(e)}")
            # Retry with a new model
            model.load_saved_model(MODEL_PATH, SCALER_PATH)
            prediction, probability = model.predict_single(amount)
        
        # Get the previous hash from the last transaction
        previous_hash = model.get_last_hash() or 'Genesis Block'
        
        # Get current transaction ID
        transaction_id = len(model.get_blockchain())
        
        # Create transaction data string for hashing with consistent format
        transaction_data = f"AMOUNT:{amount:.2f}|PRED:{prediction}|PROB:{probability:.4f}|PREV:{previous_hash}|ID:{transaction_id}"
        
        # Calculate current hash
        current_hash = hashlib.sha256(transaction_data.encode()).hexdigest()
        
        # Create transaction record with proper hash chaining
        transaction = {
            'amount': amount,
            'prediction': prediction,
            'probability': probability,
            'previous_hash': previous_hash,
            'block_hash': current_hash,
            'transaction_id': transaction_id
        }
        
        # Add to blockchain
        model.add_to_blockchain(transaction)
        
        # Get all transactions from blockchain
        transactions = model.get_blockchain()
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': probability,
            'transactions': transactions,
            'metrics': model.get_metrics()
        })
    except Exception as e:
        logger.error(f"Model prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/metrics')
def get_metrics():
    try:
        metrics = {
            'accuracy': round(model.metrics['accuracy'] * 100, 2),
            'precision': round(model.metrics['precision'] * 100, 2),
            'recall': round(model.metrics['recall'] * 100, 2),
            'f1_score': round(model.metrics['f1_score'] * 100, 2)
        }
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear_blockchain', methods=['POST'])
def clear_blockchain():
    try:
        # Clear the blockchain file
        with open('logs/ledger.txt', 'w') as f:
            f.write('')  # Clear the file contents
        return jsonify({'status': 'success', 'message': 'Blockchain history cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing blockchain: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 