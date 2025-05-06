import pytest
from app import app
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Credit Card Fraud Detection' in response.data

def test_predict_endpoint(client):
    data = {
        'transaction_id': 'test123',
        'amount': 100.0
    }
    response = client.post('/predict',
                         data=json.dumps(data),
                         content_type='application/json')
    assert response.status_code == 200
    result = json.loads(response.data)
    assert 'transaction_id' in result
    assert 'prediction' in result
    assert 'probability' in result
    assert 'block_hash' in result

def test_metrics_endpoint(client):
    response = client.get('/metrics')
    assert response.status_code == 200
    result = json.loads(response.data)
    assert 'accuracy' in result
    assert 'precision' in result
    assert 'recall' in result
    assert 'f1_score' in result

def test_transaction_endpoint(client):
    # First, add a transaction
    data = {
        'transaction_id': 'test123',
        'amount': 100.0
    }
    client.post('/predict',
               data=json.dumps(data),
               content_type='application/json')
    
    # Then, try to get it
    response = client.get('/transactions/test123')
    assert response.status_code == 200
    result = json.loads(response.data)
    assert result['transaction_id'] == 'test123'

def test_validate_chain_endpoint(client):
    response = client.get('/validate-chain')
    assert response.status_code == 200
    result = json.loads(response.data)
    assert 'is_valid' in result

def test_invalid_transaction_id(client):
    response = client.get('/transactions/nonexistent')
    assert response.status_code == 404

def test_invalid_predict_request(client):
    # Missing required fields
    data = {
        'amount': 100.0
    }
    response = client.post('/predict',
                         data=json.dumps(data),
                         content_type='application/json')
    assert response.status_code == 500 