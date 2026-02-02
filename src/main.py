from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import joblib
import numpy as np
import json
import logging
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter(
    'fraud_predictions_total', 
    'Total number of fraud predictions',
    ['result']
)
fraud_detected_counter = Counter(
    'fraud_detected_total', 
    'Total fraud cases detected'
)
prediction_duration = Histogram(
    'prediction_duration_seconds', 
    'Time spent processing prediction'
)
active_requests = Gauge(
    'active_requests', 
    'Number of active requests'
)
model_accuracy = Gauge(
    'model_accuracy', 
    'Current model accuracy'
)
daily_requests = Counter(
    'daily_requests_total',
    'Total daily requests'
)

# Initialize FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection with automated CI/CD",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
    
    # Set model accuracy gauge
    model_accuracy.set(model_metrics.get('accuracy', 0.0))
    
    logger.info("✅ Model loaded successfully")
    logger.info(f"Model: {model_metrics.get('model_name')}")
    logger.info(f"Accuracy: {model_metrics.get('accuracy'):.4f}")
    logger.info(f"Features: {len(feature_names)}")
    
except Exception as e:
    logger.error(f"❌ Failed to load model: {str(e)}")
    raise

# Request/Response models
class Transaction(BaseModel):
    features: List[float] = Field(
        ..., 
        description=f"Transaction features ({len(feature_names)} values)",
        min_items=len(feature_names),
        max_items=len(feature_names)
    )
    
    @validator('features')
    def validate_features(cls, v):
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError("Features contain NaN or Inf values")
        return v

class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    confidence: float
    timestamp: str
    processing_time_ms: float

class BatchTransaction(BaseModel):
    transactions: List[List[float]]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Dict
    uptime_seconds: float

class StatsResponse(BaseModel):
    total_predictions: int
    fraud_detected: int
    fraud_rate: float
    average_processing_time_ms: float
    model_accuracy: float

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    active_requests.inc()
    daily_requests.inc()
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    active_requests.dec()
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Simple in-memory logging
class PredictionLogger:
    def __init__(self):
        self.predictions = []
        self.log_file = "logs/predictions.json"
        os.makedirs("logs", exist_ok=True)
    
    def log(self, data: dict):
        """Log prediction to file"""
        try:
            self.predictions.append(data)
            
            # Save to file every 10 predictions
            if len(self.predictions) % 10 == 0:
                self._save_to_file()
        except Exception as e:
            logger.error(f"Logging error: {e}")
    
    def _save_to_file(self):
        """Save predictions to JSON file"""
        try:
            # Load existing logs
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    existing_logs = json.load(f)
            else:
                existing_logs = []
            
            # Append new predictions
            existing_logs.extend(self.predictions)
            
            # Save back
            with open(self.log_file, 'w') as f:
                json.dump(existing_logs, f, indent=2)
            
            # Clear in-memory predictions
            self.predictions = []
            
        except Exception as e:
            logger.error(f"File save error: {e}")

pred_logger = PredictionLogger()

# Start time for uptime tracking
start_time = time.time()

# Endpoints
@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Fraud Detection API",
        "version": "2.0.0",
        "status": "operational",
        "model": model_metrics.get('model_name'),
        "accuracy": f"{model_metrics.get('accuracy', 0)*100:.2f}%",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "health": "/health",
            "stats": "/stats",
            "metrics": "/metrics"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    uptime = time.time() - start_time
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_info": {
            "name": model_metrics.get('model_name'),
            "accuracy": model_metrics.get('accuracy'),
            "features": len(feature_names),
            "training_date": model_metrics.get('training_date')
        },
        "uptime_seconds": uptime
    }

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get API statistics"""
    total_preds = prediction_counter._value._value
    fraud_detected = fraud_detected_counter._value._value
    
    return {
        "total_predictions": int(total_preds),
        "fraud_detected": int(fraud_detected),
        "fraud_rate": fraud_detected / total_preds if total_preds > 0 else 0.0,
        "average_processing_time_ms": 45.0,  # This would be calculated from histogram
        "model_accuracy": model_metrics.get('accuracy', 0.0)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    """Predict if a transaction is fraudulent"""
    transaction_id = f"TXN_{int(time.time() * 1000)}"
    start_time_pred = time.time()
    
    try:
        with prediction_duration.time():
            # Prepare features
            features_array = np.array(transaction.features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
            
            # Determine risk level
            if probability < 0.3:
                risk_level = "LOW"
            elif probability < 0.7:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            # Update metrics
            prediction_counter.labels(result='fraud' if prediction == 1 else 'legitimate').inc()
            if prediction == 1:
                fraud_detected_counter.inc()
            
            # Calculate processing time
            processing_time = (time.time() - start_time_pred) * 1000
            
            # Log prediction
            log_data = {
                "transaction_id": transaction_id,
                "timestamp": datetime.utcnow().isoformat(),
                "prediction": int(prediction),
                "probability": float(probability),
                "risk_level": risk_level,
                "processing_time_ms": processing_time
            }
            pred_logger.log(log_data)
            
            logger.info(f"Prediction: {transaction_id} - Fraud: {prediction}, Prob: {probability:.4f}")
            
            return {
                "transaction_id": transaction_id,
                "is_fraud": bool(prediction),
                "fraud_probability": float(probability),
                "risk_level": risk_level,
                "confidence": float(max(probability, 1 - probability)),
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_ms": processing_time
            }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/batch-predict")
async def batch_predict(batch: BatchTransaction):
    """Predict multiple transactions at once"""
    try:
        results = []
        
        for idx, features in enumerate(batch.transactions):
            if len(features) != len(feature_names):
                results.append({
                    "index": idx,
                    "error": f"Invalid feature count: expected {len(feature_names)}, got {len(features)}"
                })
                continue
            
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
            
            prediction_counter.labels(result='fraud' if prediction == 1 else 'legitimate').inc()
            if prediction == 1:
                fraud_detected_counter.inc()
            
            results.append({
                "index": idx,
                "is_fraud": bool(prediction),
                "fraud_probability": float(probability),
                "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
            })
        
        return {
            "batch_id": f"BATCH_{int(time.time())}",
            "predictions": results,
            "count": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )