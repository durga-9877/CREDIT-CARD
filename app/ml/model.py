import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        self.metrics = {}

    def load_data(self, file_path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self, data: pd.DataFrame) -> tuple:
        try:
            # Basic preprocessing
            data['Time'] = data['Time'].apply(lambda x: x / 3600)  # Convert to hours
            data['Amount'] = np.log1p(data['Amount'])  # Log transform amount

            # Feature engineering
            data['Transaction_Rate'] = data['Amount'] / (data['Time'] + 1)
            data['Amount_Time_Interaction'] = data['Amount'] * data['Time']

            # Separate features and target
            X = data.drop('Class', axis=1)
            y = data['Class']

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Apply SMOTE to training data
            X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train_scaled, y_train)

            logger.info("Data preprocessing completed successfully")
            return X_train_resampled, X_test_scaled, y_train_resampled, y_test
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def train_model(self, X_train: np.ndarray, y_train: pd.Series):
        try:
            # Perform hyperparameter tuning
            best_params = self.hyperparameter_tuning(X_train, y_train)
            
            # Initialize and train the model with best parameters
            self.model = RandomForestClassifier(**best_params, random_state=42)
            self.model.fit(X_train, y_train)
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def evaluate_model(self, X_test: np.ndarray, y_test: pd.Series):
        try:
            y_pred = self.model.predict(X_test)
            
            self.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred)
            }
            
            logger.info("Model evaluation completed successfully")
            logger.info(f"Model metrics: {self.metrics}")
            return self.metrics
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise

    def save_model(self, model_path: str, scaler_path: str):
        try:
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Model and scaler saved successfully to {model_path} and {scaler_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_saved_model(self, model_path: str, scaler_path: str):
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("Model and scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, features: np.ndarray) -> tuple:
        try:
            # Scale the features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0][1]
            
            return {
                'prediction': 'Fraudulent' if prediction == 1 else 'Legitimate',
                'probability': probability if prediction == 1 else 1 - probability
            }
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def predict_single(self, amount: float) -> dict:
        try:
            # Create a synthetic transaction with the given amount
            # We'll use average values for other features
            features = np.zeros((1, 30))  # Assuming 30 features
            features[0, 0] = amount  # Set the amount
            features[0, 1] = np.random.normal(0, 1)  # Time feature
            
            # Scale the features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0][1]
            
            return {
                'prediction': 'Fraudulent' if prediction == 1 else 'Legitimate',
                'probability': float(probability if prediction == 1 else 1 - probability)
            }
        except Exception as e:
            logger.error(f"Error in single prediction: {str(e)}")
            raise

    def get_metrics(self) -> dict:
        """Return the current model metrics"""
        return self.metrics

    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: pd.Series):
        try:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced']
            }
            
            grid_search = GridSearchCV(
                estimator=RandomForestClassifier(random_state=42),
                param_grid=param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            logger.info(f"Best parameters found: {grid_search.best_params_}")
            return grid_search.best_params_
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            raise 