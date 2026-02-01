from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict
import joblib
import numpy as np
import json
import logging
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter('fraud_predictions_total', 'Total predictions made')
fraud_detected_counter = Counter('fraud_detected_total', 'Total fraud cases detected')
prediction_duration = Histogram('prediction_duration_seconds', 'Time spent processing prediction')

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection system",
    version="1.0.0"
)

# Load model artifacts
MODEL_PATH = "models/fraud_model.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURES_PATH = "models/feature_names.json"
METRICS_PATH = "models/metrics.json"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    with open(FEATURES_PATH, 'r') as f:
        feature_names = json.load(f)
    
    with open(METRICS_PATH, 'r') as f:
        model_metrics = json.load(f)
    
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {str(e)}")
    raise

# Request/Response models
class Transaction(BaseModel):
    features: List[float] = Field(..., description="Transaction features (30 values)")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.0] * 30  # Example with 30 features
            }
        }

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    timestamp: str
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_metrics: Dict

class BatchTransaction(BaseModel):
    transactions: List[List[float]]

# Endpoints
@app.get("/", response_model=Dict)
def root():
    """Root endpoint with API information"""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "health": "/health",
            "metrics": "/metrics"
        }
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_metrics": model_metrics
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: Transaction):
    """Predict if a transaction is fraudulent"""
    try:
        with prediction_duration.time():
            # Validate input
            if len(transaction.features) != len(feature_names):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Expected {len(feature_names)} features, got {len(transaction.features)}"
                )
            
            # Prepare input
            features_array = np.array(transaction.features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
            
            # Update metrics
            prediction_counter.inc()
            if prediction == 1:
                fraud_detected_counter.inc()
            
            # Determine risk level
            if probability < 0.3:
                risk_level = "low"
            elif probability < 0.7:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            logger.info(f"Prediction made: fraud={prediction}, probability={probability:.4f}")
            
            return {
                "is_fraud": bool(prediction),
                "fraud_probability": float(probability),
                "risk_level": risk_level,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/batch-predict")
def batch_predict(batch: BatchTransaction):
    """Predict multiple transactions at once"""
    try:
        results = []
        
        for features in batch.transactions:
            if len(features) != len(feature_names):
                results.append({"error": f"Invalid feature count: {len(features)}"})
                continue
            
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
            
            prediction_counter.inc()
            if prediction == 1:
                fraud_detected_counter.inc()
            
            results.append({
                "is_fraud": bool(prediction),
                "fraud_probability": float(probability)
            })
        
        return {"predictions": results, "count": len(results)}
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)