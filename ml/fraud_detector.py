import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import sklearn
import json

class FraudDetector:
    def __init__(self):
        self.model_with_smote = None
        self.model_without_smote = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.metrics = {
            'with_smote': {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0
            },
            'without_smote': {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0
            }
        }
        self.ledger_file = 'logs/ledger.txt'
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Log scikit-learn version
        logging.info(f"Using scikit-learn version: {sklearn.__version__}")
        
    def load_data(self, filepath):
        """Load and prepare the credit card dataset."""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Dataset file not found at {filepath}")
            data = pd.read_csv(filepath)
            if data.empty:
                raise ValueError("Dataset is empty")
            return data
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, data):
        """Preprocess the data and split into training and testing sets."""
        try:
            # Separate features and target
            X = data.drop(['Class', 'Time'], axis=1)
            y = data['Class']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            return X_train_scaled, X_test_scaled, y_train, y_test
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def train_models(self, X_train_scaled, y_train):
        """Train both SMOTE and non-SMOTE models."""
        try:
            # Train model without SMOTE
            self.model_without_smote = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            self.model_without_smote.fit(X_train_scaled, y_train)
            
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            
            # Train model with SMOTE
            self.model_with_smote = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            self.model_with_smote.fit(X_train_resampled, y_train_resampled)
            
            # Get feature importance
            self.feature_importance = pd.DataFrame({
                'feature': [f'V{i+1}' for i in range(28)] + ['Amount'],
                'importance': self.model_with_smote.feature_importances_
            }).sort_values('importance', ascending=False)
        except Exception as e:
            logging.error(f"Error training models: {str(e)}")
            raise
    
    def evaluate_models(self, X_test_scaled, y_test):
        """Evaluate both models and store metrics."""
        try:
            for model_name, model in [('with_smote', self.model_with_smote),
                                    ('without_smote', self.model_without_smote)]:
                if model is None:
                    raise ValueError(f"{model_name} model is not trained")
                    
                y_pred = model.predict(X_test_scaled)
                
                self.metrics[model_name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                # Log the metrics
                logging.info(f"{model_name} model metrics:")
                logging.info(f"Accuracy: {self.metrics[model_name]['accuracy']:.4f}")
                logging.info(f"Precision: {self.metrics[model_name]['precision']:.4f}")
                logging.info(f"Recall: {self.metrics[model_name]['recall']:.4f}")
                logging.info(f"F1 Score: {self.metrics[model_name]['f1']:.4f}")
                
        except Exception as e:
            logging.error(f"Error evaluating models: {str(e)}")
            raise
    
    def predict_transaction(self, transaction_data):
        """Predict fraud for a single transaction."""
        try:
            if self.model_with_smote is None:
                raise ValueError("Model is not trained")
            
            # Ensure transaction_data is 2D array
            if len(transaction_data.shape) == 1:
                transaction_data = transaction_data.reshape(1, -1)
                
            # Scale the transaction data
            scaled_data = self.scaler.transform(transaction_data)
            
            try:
                # Get prediction and probability from SMOTE model
                prediction = self.model_with_smote.predict(scaled_data)[0]
                probability = self.model_with_smote.predict_proba(scaled_data)[0][1]
                
                return {
                    'prediction': int(prediction),
                    'probability': float(probability),
                    'status': 'success'
                }
            except Exception as e:
                logging.error(f"Prediction error: {str(e)}")
                return {
                    'prediction': None,
                    'probability': None,
                    'status': 'error',
                    'message': str(e)
                }
                
        except Exception as e:
            logging.error(f"Error in predict_transaction: {str(e)}")
            return {
                'prediction': None,
                'probability': None,
                'status': 'error',
                'message': str(e)
            }
    
    def log_to_blockchain(self, transaction_index, prediction, time_value):
        """Log transaction to blockchain-style ledger."""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            record = f"{timestamp}|{transaction_index}|{prediction}|{time_value}"
            hash_value = hashlib.sha256(record.encode()).hexdigest()
            
            with open(self.ledger_file, 'a') as f:
                f.write(f"{record}|{hash_value}\n")
        except Exception as e:
            logging.error(f"Error logging to blockchain: {str(e)}")
            raise
    
    def plot_model_comparison(self):
        """Create bar chart comparing model performance."""
        try:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            x = np.arange(len(metrics))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width/2, [self.metrics['with_smote'][m] for m in metrics],
                   width, label='With SMOTE')
            ax.bar(x + width/2, [self.metrics['without_smote'][m] for m in metrics],
                   width, label='Without SMOTE')
            
            ax.set_ylabel('Score (%)')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig('static/model_comparison.png')
            plt.close()
        except Exception as e:
            logging.error(f"Error plotting model comparison: {str(e)}")
            raise
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix for SMOTE model."""
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(self.metrics['with_smote']['confusion_matrix'],
                       annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix (SMOTE Model)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig('static/confusion_matrix.png')
            plt.close()
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {str(e)}")
            raise
    
    def save_model(self, model_path, scaler_path):
        """Save the trained models and scaler."""
        try:
            if self.model_with_smote is None:
                raise ValueError("Model is not trained")
                
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            joblib.dump(self.model_with_smote, model_path)
            joblib.dump(self.scaler, scaler_path)
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
    
    def predict_single(self, amount):
        """Predict fraud for a single transaction amount."""
        try:
            if self.model_with_smote is None:
                raise ValueError("Model is not trained")
            
            # Create a sample transaction with the amount
            sample = np.zeros((1, 29))  # 28 V features + Amount
            sample[0, -1] = amount  # Set the amount
            
            # Scale the transaction data
            scaled_data = self.scaler.transform(sample)
            
            try:
                # Get prediction and probability from SMOTE model
                prediction = self.model_with_smote.predict(scaled_data)[0]
                probability = self.model_with_smote.predict_proba(scaled_data)[0][1]
                
                return int(prediction), float(probability)
            except AttributeError as e:
                # Handle version mismatch by retraining the model
                logging.warning(f"Model version mismatch detected: {str(e)}")
                logging.info("Retraining model with current scikit-learn version...")
                
                # Load the data and retrain
                data = self.load_data('data/creditcard.csv')
                X_train_scaled, X_test_scaled, y_train, y_test = self.preprocess_data(data)
                self.train_models(X_train_scaled, y_train)
                self.evaluate_models(X_test_scaled, y_test)
                
                # Save the new model
                self.save_model('models/fraud_detector.joblib', 'models/scaler.joblib')
                
                # Try prediction again with new model
                prediction = self.model_with_smote.predict(scaled_data)[0]
                probability = self.model_with_smote.predict_proba(scaled_data)[0][1]
                
                return int(prediction), float(probability)
                
        except Exception as e:
            logging.error(f"Error in predict_single: {str(e)}")
            raise

    def load_saved_model(self, model_path, scaler_path):
        """Load saved models and scaler."""
        try:
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                raise FileNotFoundError("Model or scaler file not found")
            
            try:
                self.model_with_smote = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                logging.info("Successfully loaded saved model and scaler")
            except Exception as e:
                logging.warning(f"Error loading saved model: {str(e)}")
                logging.info("Retraining model with current scikit-learn version...")
                
                # Load the data and retrain
                data = self.load_data('data/creditcard.csv')
                X_train_scaled, X_test_scaled, y_train, y_test = self.preprocess_data(data)
                self.train_models(X_train_scaled, y_train)
                self.evaluate_models(X_test_scaled, y_test)
                
                # Save the new model
                self.save_model(model_path, scaler_path)
                logging.info("Model retrained and saved successfully")
                
        except Exception as e:
            logging.error(f"Error in load_saved_model: {str(e)}")
            raise

    def get_last_hash(self):
        """Get the hash of the last transaction in the blockchain."""
        try:
            if os.path.exists('logs/ledger.txt'):
                with open('logs/ledger.txt', 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_transaction = json.loads(lines[-1])
                        return last_transaction.get('block_hash')
        except Exception as e:
            logging.error(f"Error getting last hash: {str(e)}")
        return None

    def verify_transaction_hash(self, transaction):
        """Verify if the transaction hash is valid."""
        try:
            # Create the same transaction data string used for hashing
            transaction_data = f"AMOUNT:{transaction['amount']:.2f}|PRED:{transaction['prediction']}|PROB:{transaction['probability']:.4f}|PREV:{transaction['previous_hash']}|ID:{transaction.get('transaction_id', 0)}"
            calculated_hash = hashlib.sha256(transaction_data.encode()).hexdigest()
            return calculated_hash == transaction['block_hash']
        except Exception as e:
            logging.error(f"Error verifying transaction hash: {str(e)}")
            return False

    def verify_blockchain_integrity(self, transactions):
        """Verify the integrity of the entire blockchain."""
        try:
            if not transactions:
                return True
            
            # Verify each transaction's hash
            for i, transaction in enumerate(transactions):
                if not self.verify_transaction_hash(transaction):
                    logging.error(f"Invalid hash in transaction {i}")
                    return False
                
                # Verify previous hash links (except for first transaction)
                if i > 0:
                    if transaction['previous_hash'] != transactions[i-1]['block_hash']:
                        logging.error(f"Invalid previous hash link in transaction {i}")
                        return False
                    
            return True
        except Exception as e:
            logging.error(f"Error verifying blockchain integrity: {str(e)}")
            return False

    def add_to_blockchain(self, transaction):
        """Add a transaction to the blockchain."""
        try:
            # Get existing transactions
            existing_transactions = self.get_blockchain()
            
            # Verify blockchain integrity before adding new transaction
            if not self.verify_blockchain_integrity(existing_transactions):
                raise ValueError("Blockchain integrity check failed")
            
            # Verify that the previous hash matches the last transaction's hash
            if existing_transactions and transaction['previous_hash'] != existing_transactions[-1]['block_hash']:
                raise ValueError("Previous hash does not match last transaction's hash")
            
            # Verify the transaction hash before adding
            if not self.verify_transaction_hash(transaction):
                raise ValueError("Invalid transaction hash")
            
            os.makedirs('logs', exist_ok=True)
            with open('logs/ledger.txt', 'a') as f:
                f.write(json.dumps(transaction) + '\n')
        except Exception as e:
            logging.error(f"Error adding to blockchain: {str(e)}")
            raise

    def get_blockchain(self):
        """Get all transactions from the blockchain."""
        try:
            if os.path.exists('logs/ledger.txt'):
                with open('logs/ledger.txt', 'r') as f:
                    transactions = [json.loads(line) for line in f.readlines()]
                    # Verify blockchain integrity
                    if not self.verify_blockchain_integrity(transactions):
                        logging.warning("Blockchain integrity check failed")
                    return transactions
        except Exception as e:
            logging.error(f"Error reading blockchain: {str(e)}")
        return [] 