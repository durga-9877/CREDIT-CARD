import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        
    def load_data(self, file_path):
        """Load and preprocess the credit card data"""
        df = pd.read_csv(file_path)
        
        # Basic preprocessing
        df['Time'] = df['Time'].apply(lambda x: x / 3600)  # Convert to hours
        df['Amount'] = np.log1p(df['Amount'])  # Log transform amount
        
        # Feature engineering
        df['Transaction_Rate'] = df['Amount'] / (df['Time'] + 1)  # Avoid division by zero
        df['Amount_Time_Interaction'] = df['Amount'] * df['Time']
        
        # Separate features and target
        X = df.drop(['Class', 'Time', 'Amount'], axis=1)
        y = df['Class']
        
        return X, y
    
    def train(self, X, y):
        """Train the model with hyperparameter tuning and SMOTE"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE to training data
        X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train_scaled, y_train)
        
        # Define hyperparameters for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced']
        }
        
        # Initialize and train the model with GridSearchCV
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_resampled, y_train_resampled)
        self.model = grid_search.best_estimator_
        
        # Make predictions and calculate metrics
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        return metrics, X_test_scaled, y_test
    
    def predict(self, features):
        """Make predictions with confidence scores"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        # Scale the features
        features_scaled = self.scaler.transform(features)
        
        # Get predictions and probabilities
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get confidence score (probability of predicted class)
        confidence = probabilities[prediction]
        
        return {
            'prediction': 'Fraudulent' if prediction == 1 else 'Legitimate',
            'probability': confidence
        }
    
    def save_model(self, model_path='model.joblib', scaler_path='scaler.joblib'):
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
    
    def load_model(self, model_path='model.joblib', scaler_path='scaler.joblib'):
        """Load a trained model and scaler"""
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model or scaler files not found")
            
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) 