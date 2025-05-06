# Credit Card Fraud Detection System

This is a machine learning-based credit card fraud detection system that uses both SMOTE and non-SMOTE models to detect fraudulent transactions. The system includes a web interface for real-time transaction analysis and blockchain-style logging of predictions.

## Features

- Real-time transaction analysis using transaction index
- Both SMOTE and non-SMOTE models for comparison
- Model performance metrics (Accuracy, Precision, Recall, F1 Score)
- Feature importance visualization
- Blockchain-style transaction logging
- Modern web interface

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd credit-card-fraud-detection
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the project in development mode:
```bash
pip install -e .
```

5. Place your credit card dataset in the `data` directory:
- The dataset should be named `creditcard.csv`
- It should contain the standard credit card fraud detection features

6. Create necessary directories:
```bash
mkdir -p data models logs static
```

## Usage

1. Start the application:
```bash
python app/app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Enter a transaction index (0-284806) to analyze a specific transaction

4. View the prediction results, model metrics, and feature importance

## Project Structure

```
credit-card-fraud-detection/
├── app/
│   ├── app.py              # Flask application
│   ├── templates/
│   │   └── index.html      # Web interface
│   └── static/             # Static files
├── ml/
│   └── fraud_detector.py   # ML model implementation
├── data/
│   └── creditcard.csv      # Dataset
├── models/                 # Saved models
├── logs/                   # Transaction logs
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

## Model Details

The system uses two Random Forest models:
1. Standard model (non-SMOTE)
2. SMOTE-balanced model for handling class imbalance

Both models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score

## Logging

All predictions are logged in a blockchain-style format in `logs/ledger.txt`. Each log entry includes:
- Transaction index
- Prediction (Fraudulent/Legitimate)
- Timestamp
- Hash of the previous entry

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 