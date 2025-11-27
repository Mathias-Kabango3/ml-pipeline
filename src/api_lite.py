"""
Lightweight FastAPI Application using TensorFlow Lite
=====================================================
Optimized for low-memory environments like Render free tier.
Uses TFLite interpreter which requires ~10x less memory than full TensorFlow.
"""

import os
import io
import json
import time
import logging
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Only import TFLite interpreter (much lighter than full TensorFlow)
import tflite_runtime.interpreter as tflite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Model URL - TFLite version
TFLITE_MODEL_URL = os.environ.get(
    "TFLITE_MODEL_URL",
    "https://huggingface.co/mathiaskabango/plantvillage/resolve/main/plant_disease_model.tflite"
)


# Global state
class AppState:
    interpreter: Optional[tflite.Interpreter] = None
    input_details: Optional[list] = None
    output_details: Optional[list] = None
    model_loaded_at: Optional[datetime] = None
    class_indices: Dict[int, str] = {}
    prediction_count: int = 0


app_state = AppState()


# Pydantic models
class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float
    all_predictions: Dict[str, float]
    inference_time_ms: float


class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int
    predictions_made: int
    uptime_seconds: Optional[float]


def download_model(url: str, dest_path: Path) -> bool:
    """Download model from URL."""
    try:
        logger.info(f"Downloading TFLite model from {url}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(dest_path))
        logger.info(f"Model downloaded to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False


def load_model_and_classes():
    """Load TFLite model and class indices."""
    try:
        model_path = MODELS_DIR / "plant_disease_model.tflite"
        
        # Download if not exists
        if not model_path.exists():
            download_model(TFLITE_MODEL_URL, model_path)
        
        if not model_path.exists():
            logger.error("Model file not found")
            return None, None, None, {}
        
        logger.info(f"Loading TFLite model from {model_path}")
        
        # Create interpreter
        interpreter = tflite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"Model loaded. Input shape: {input_details[0]['shape']}")
        
        # Load class indices
        class_indices = {}
        class_indices_path = DATA_DIR / "index_to_class.json"
        if class_indices_path.exists():
            with open(class_indices_path, 'r') as f:
                class_indices = {int(k): v for k, v in json.load(f).items()}
            logger.info(f"Loaded {len(class_indices)} classes")
        
        return interpreter, input_details, output_details, class_indices
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None, None, {}


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image for model inference."""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


# Create FastAPI app
app = FastAPI(
    title="Plant Disease Classification API (Lite)",
    description="Lightweight API using TensorFlow Lite for plant disease detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting API server...")
    interpreter, input_details, output_details, class_indices = load_model_and_classes()
    app_state.interpreter = interpreter
    app_state.input_details = input_details
    app_state.output_details = output_details
    app_state.class_indices = class_indices
    if interpreter:
        app_state.model_loaded_at = datetime.now()
        logger.info("Model loaded successfully!")
    else:
        logger.warning("Failed to load model on startup")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {"message": "Plant Disease Classification API (Lite)", "docs": "/docs"}


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": app_state.interpreter is not None}


@app.get("/status", response_model=StatusResponse, tags=["Health"])
async def get_status():
    """Get API status."""
    uptime = None
    if app_state.model_loaded_at:
        uptime = (datetime.now() - app_state.model_loaded_at).total_seconds()
    
    return StatusResponse(
        status="healthy" if app_state.interpreter else "degraded",
        model_loaded=app_state.interpreter is not None,
        num_classes=len(app_state.class_indices),
        predictions_made=app_state.prediction_count,
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """Predict plant disease from image."""
    if app_state.interpreter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Read and preprocess image
    contents = await file.read()
    input_data = preprocess_image(contents)
    
    # Run inference
    start_time = time.time()
    app_state.interpreter.set_tensor(app_state.input_details[0]['index'], input_data)
    app_state.interpreter.invoke()
    predictions = app_state.interpreter.get_tensor(app_state.output_details[0]['index'])[0]
    inference_time = (time.time() - start_time) * 1000
    
    # Get results
    predicted_class_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class_idx])
    predicted_class = app_state.class_indices.get(predicted_class_idx, f"Class_{predicted_class_idx}")
    
    # Top 5 predictions
    top_indices = np.argsort(predictions)[-5:][::-1]
    all_predictions = {
        app_state.class_indices.get(int(i), f"Class_{i}"): float(predictions[i])
        for i in top_indices
    }
    
    app_state.prediction_count += 1
    
    return PredictionResponse(
        success=True,
        prediction=predicted_class,
        confidence=confidence,
        all_predictions=all_predictions,
        inference_time_ms=inference_time
    )


@app.get("/classes", tags=["Info"])
async def get_classes():
    """Get list of all disease classes."""
    return {
        "num_classes": len(app_state.class_indices),
        "classes": list(app_state.class_indices.values())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
