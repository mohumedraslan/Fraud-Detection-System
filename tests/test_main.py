from fastapi.testclient import TestClient
import sys
import os
import json
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import app, feature_names

client = TestClient(app)

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["status"] == "operational"

def test_health():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_info" in data
    assert "uptime_seconds" in data

def test_stats():
    """Test stats endpoint"""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "fraud_detected" in data
    assert "model_accuracy" in data

def test_predict_valid():
    """Test valid prediction"""
    transaction = {
        "features": np.random.randn(len(feature_names)).tolist()
    }
    
    response = client.post("/predict", json=transaction)
    assert response.status_code == 200
    
    data = response.json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert "transaction_id" in data
    assert "timestamp" in data
    assert "processing_time_ms" in data
    
    assert isinstance(data["is_fraud"], bool)
    assert 0 <= data["fraud_probability"] <= 1
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

def test_predict_invalid_feature_count():
    """Test prediction with wrong number of features"""
    transaction = {
        "features": [0.0] * 10  # Wrong count
    }
    
    response = client.post("/predict", json=transaction)
    assert response.status_code == 422  # Validation error

def test_predict_invalid_values():
    """Test prediction with invalid values"""
    transaction = {
        "features": [float('nan')] * len(feature_names)
    }
    
    response = client.post("/predict", json=transaction)
    assert response.status_code == 422

def test_batch_predict():
    """Test batch prediction"""
    batch = {
        "transactions": [
            np.random.randn(len(feature_names)).tolist() for _ in range(3)
        ]
    }
    
    response = client.post("/batch-predict", json=batch)
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert "count" in data
    assert len(data["predictions"]) == 3
    
    for pred in data["predictions"]:
        assert "is_fraud" in pred
        assert "fraud_probability" in pred

def test_metrics_endpoint():
    """Test Prometheus metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "fraud_predictions_total" in response.text

def test_prediction_consistency():
    """Test that same input gives same prediction"""
    transaction = {
        "features": [0.5] * len(feature_names)
    }
    
    response1 = client.post("/predict", json=transaction)
    response2 = client.post("/predict", json=transaction)
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    # Same features should give same probability
    prob1 = response1.json()["fraud_probability"]
    prob2 = response2.json()["fraud_probability"]
    assert abs(prob1 - prob2) < 1e-6