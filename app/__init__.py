from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from models.fraud_detector import FraudDetector
from models.blockchain import Blockchain
import logging
from logging.handlers import RotatingFileHandler
import os
import time
import joblib
import hashlib
import json
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')
file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=5)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Fraud Detection System startup')

# Load dataset
try:
    dataset = pd.read_csv('data/creditcard.csv')
    app.logger.info('Dataset loaded successfully')
except Exception as e:
    app.logger.error(f'Error loading dataset: {str(e)}')
    dataset = None

# Initialize blockchain and statistics
blockchain = Blockchain()
statistics = {
    'total_transactions': 0,
    'legitimate_transactions': 0,
    'fraudulent_transactions': 0
}

# Initialize fraud detector
try:
    fraud_detector = FraudDetector()
    print("Fraud detector initialized successfully")
except Exception as e:
    print(f"Error initializing fraud detector: {str(e)}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        amount = float(data.get('amount', 0))
        
        # Get matching transactions
        matching_transactions = fraud_detector.get_matching_transactions(amount)
        
        if not matching_transactions:
            return jsonify({'error': f'No transactions found with amount ${amount:.2f}'}), 404
        
        # Process each transaction
        results = []
        for transaction in matching_transactions:
            # Get prediction and probability
            prediction, probability = fraud_detector.predict(transaction)
            
            # Add transaction to blockchain
            block = blockchain.add_block(
                prediction=prediction,
                amount=transaction['Amount'],
                time=transaction['Time']
            )
            
            results.append({
                'prediction': prediction,
                'probability': probability,
                'amount': transaction['Amount'],
                'time': transaction['Time'],
                'block_index': block.index,
                'block_hash': block.hash,
                'previous_hash': block.previous_hash
            })
        
        return jsonify({
            'transactions': results,
            'blockchain': blockchain.get_chain()
        })
        
    except Exception as e:
        app.logger.error(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def get_metrics():
    try:
        metrics = fraud_detector.get_metrics()
        return jsonify(metrics)
    except Exception as e:
        app.logger.error(f'Error getting metrics: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/transactions/<amount>')
def get_transaction(amount):
    try:
        amount = float(amount)
        transaction = blockchain.get_transaction_by_amount(amount)
        if transaction:
            return jsonify(transaction)
        return jsonify({'error': 'Transaction not found'}), 404
    except Exception as e:
        app.logger.error(f'Error getting transaction: {str(e)}')
        return jsonify({'error': str(e)}), 400

@app.route('/validate-chain')
def validate_chain():
    try:
        is_valid = blockchain.validate_chain()
        return jsonify({'is_valid': is_valid})
    except Exception as e:
        app.logger.error(f'Error validating chain: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 