import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    try:
        # Load the dataset
        logger.info("Loading dataset...")
        df = pd.read_csv('data/creditcard.csv')
        
        # Separate features and target
        X = df[['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']]
        y = df['Class']
        
        # Split the data
        logger.info("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        logger.info("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save the model and scaler
        logger.info("Saving model and scaler...")
        joblib.dump(model, 'models/fraud_detector.joblib')
        joblib.dump(scaler, 'models/scaler.joblib')
        
        # Calculate metrics
        y_pred = model.predict(X_test_scaled)
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred))
        }
        
        # Save metrics to a JSON file
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        logger.info("Model Performance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric.capitalize()}: {value:.4f}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error in training model: {str(e)}")
        return False

if __name__ == '__main__':
    train_model() 