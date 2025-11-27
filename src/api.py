"""
FastAPI Application for Plant Disease Classification
=====================================================
This module provides REST API endpoints for prediction,
model retraining, and health monitoring.
"""

import os
import io
import json
import time
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras

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
RETRAIN_DIR = DATA_DIR / "retrain_data"


# Global state
class AppState:
    model: Optional[keras.Model] = None
    model_loaded_at: Optional[datetime] = None
    class_indices: Dict[int, str] = {}
    is_retraining: bool = False
    retrain_progress: float = 0.0
    retrain_status: str = ""
    prediction_count: int = 0
    total_inference_time: float = 0.0
    
app_state = AppState()


# Pydantic models for request/response
class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float
    all_predictions: Dict[str, float]
    inference_time_ms: float


class RetrainRequest(BaseModel):
    epochs: int = 10
    learning_rate: float = 0.0001
    model_type: str = "resnet50"


class RetrainResponse(BaseModel):
    success: bool
    message: str
    task_id: Optional[str] = None


class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    model_loaded_at: Optional[str]
    uptime_seconds: float
    num_classes: int
    prediction_count: int
    avg_inference_time_ms: float
    is_retraining: bool
    retrain_progress: float
    retrain_status: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str


class DatasetStatsResponse(BaseModel):
    total_train_images: int
    total_val_images: int
    total_test_images: int
    num_classes: int
    class_distribution: Dict[str, int]


# Helper functions
def load_model_and_classes():
    """Load the trained model and class mappings."""
    try:
        # Try to find the best model
        model_paths = [
            MODELS_DIR / "plant_disease_resnet50_best.keras",
            MODELS_DIR / "plant_disease_resnet50_final.keras",
            MODELS_DIR / "plant_disease_mobilenetv2_best.keras",
            MODELS_DIR / "plant_disease_efficientnet_best.keras",
        ]
        
        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            logger.warning("No trained model found. API will work but predictions won't be available.")
            return None, {}
        
        logger.info(f"Loading model from {model_path}")
        model = keras.models.load_model(str(model_path))
        
        # Load class indices
        class_indices_path = DATA_DIR / "index_to_class.json"
        if class_indices_path.exists():
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
                class_indices = {int(k): v for k, v in class_indices.items()}
        else:
            # Try class_info.json
            class_info_path = DATA_DIR / "class_info.json"
            if class_info_path.exists():
                with open(class_info_path, 'r') as f:
                    class_info = json.load(f)
                    class_names = class_info.get("class_names", [])
                    class_indices = {i: name for i, name in enumerate(class_names)}
            else:
                class_indices = {}
        
        logger.info(f"Model loaded successfully with {len(class_indices)} classes")
        return model, class_indices
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, {}


def preprocess_image(image_bytes: bytes, img_size: tuple = (224, 224)) -> np.ndarray:
    """Preprocess image for model inference."""
    # Open and resize image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(img_size, Image.Resampling.LANCZOS)
    
    # Convert to array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


async def run_retraining(
    epochs: int = 10,
    learning_rate: float = 0.0001,
    model_type: str = "resnet50"
):
    """Background task for model retraining."""
    global app_state
    
    try:
        app_state.is_retraining = True
        app_state.retrain_status = "Starting retraining..."
        app_state.retrain_progress = 0.0
        
        logger.info(f"Starting retraining with epochs={epochs}, lr={learning_rate}")
        
        # Import retrain module
        from retrain import retrain_model
        
        app_state.retrain_status = "Preparing data..."
        app_state.retrain_progress = 0.1
        await asyncio.sleep(0.1)
        
        # Run retraining (this is CPU/GPU bound, so we run it in executor)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            retrain_model,
            epochs,
            learning_rate,
            model_type,
            lambda p, s: update_progress(p, s)
        )
        
        if result.get("success"):
            # Reload model
            app_state.model, app_state.class_indices = load_model_and_classes()
            app_state.model_loaded_at = datetime.now()
            app_state.retrain_status = "Retraining completed successfully!"
            app_state.retrain_progress = 1.0
            logger.info("Retraining completed successfully")
        else:
            app_state.retrain_status = f"Retraining failed: {result.get('error', 'Unknown error')}"
            logger.error(f"Retraining failed: {result.get('error')}")
            
    except Exception as e:
        app_state.retrain_status = f"Retraining error: {str(e)}"
        logger.error(f"Retraining error: {e}")
        
    finally:
        app_state.is_retraining = False


def update_progress(progress: float, status: str):
    """Update retraining progress."""
    global app_state
    app_state.retrain_progress = progress
    app_state.retrain_status = status


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # Startup
    global app_state
    logger.info("Starting Plant Disease Classification API...")
    
    # Load model
    app_state.model, app_state.class_indices = load_model_and_classes()
    if app_state.model is not None:
        app_state.model_loaded_at = datetime.now()
    
    # Create retrain directory
    RETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Plant Disease Classification API",
    description="API for classifying plant diseases using deep learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - basic health check."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if app_state.model is not None else "degraded",
        timestamp=datetime.now().isoformat()
    )


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get detailed API status including model uptime and metrics."""
    uptime = 0.0
    if app_state.model_loaded_at:
        uptime = (datetime.now() - app_state.model_loaded_at).total_seconds()
    
    avg_inference = 0.0
    if app_state.prediction_count > 0:
        avg_inference = (app_state.total_inference_time / app_state.prediction_count) * 1000
    
    return StatusResponse(
        status="healthy" if app_state.model is not None else "degraded",
        model_loaded=app_state.model is not None,
        model_loaded_at=app_state.model_loaded_at.isoformat() if app_state.model_loaded_at else None,
        uptime_seconds=uptime,
        num_classes=len(app_state.class_indices),
        prediction_count=app_state.prediction_count,
        avg_inference_time_ms=avg_inference,
        is_retraining=app_state.is_retraining,
        retrain_progress=app_state.retrain_progress,
        retrain_status=app_state.retrain_status
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from an uploaded image.
    
    - **file**: Image file (JPEG, PNG, etc.)
    
    Returns prediction with confidence scores.
    """
    global app_state
    
    # Check if model is loaded
    if app_state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {allowed_types}"
        )
    
    try:
        # Read image
        image_bytes = await file.read()
        
        # Preprocess
        start_time = time.time()
        img_array = preprocess_image(image_bytes)
        
        # Predict
        predictions = app_state.model.predict(img_array, verbose=0)
        inference_time = time.time() - start_time
        
        # Update stats
        app_state.prediction_count += 1
        app_state.total_inference_time += inference_time
        
        # Get top prediction
        pred_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][pred_idx])
        
        # Get class name
        class_name = app_state.class_indices.get(pred_idx, f"Class_{pred_idx}")
        
        # Get all predictions with class names
        all_predictions = {}
        for idx, prob in enumerate(predictions[0]):
            class_label = app_state.class_indices.get(idx, f"Class_{idx}")
            all_predictions[class_label] = float(prob)
        
        # Sort by confidence
        all_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)[:10])
        
        logger.info(f"Prediction: {class_name} ({confidence:.4f}) in {inference_time*1000:.2f}ms")
        
        return PredictionResponse(
            success=True,
            prediction=class_name,
            confidence=confidence,
            all_predictions=all_predictions,
            inference_time_ms=inference_time * 1000
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict plant diseases for multiple images.
    
    - **files**: List of image files
    
    Returns list of predictions.
    """
    if app_state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )
    
    results = []
    
    for file in files:
        try:
            image_bytes = await file.read()
            img_array = preprocess_image(image_bytes)
            
            start_time = time.time()
            predictions = app_state.model.predict(img_array, verbose=0)
            inference_time = time.time() - start_time
            
            pred_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][pred_idx])
            class_name = app_state.class_indices.get(pred_idx, f"Class_{pred_idx}")
            
            results.append({
                "filename": file.filename,
                "success": True,
                "prediction": class_name,
                "confidence": confidence,
                "inference_time_ms": inference_time * 1000
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {"results": results}


@app.post("/retrain", response_model=RetrainResponse)
async def trigger_retrain(
    background_tasks: BackgroundTasks,
    request: RetrainRequest = RetrainRequest()
):
    """
    Trigger model retraining.
    
    - **epochs**: Number of training epochs (default: 10)
    - **learning_rate**: Learning rate (default: 0.0001)
    - **model_type**: Model type to train (default: resnet50)
    
    Retraining runs in background and can be monitored via /status endpoint.
    """
    if app_state.is_retraining:
        return RetrainResponse(
            success=False,
            message="Retraining already in progress. Check /status for progress."
        )
    
    # Check for retrain data
    retrain_data_exists = RETRAIN_DIR.exists() and any(RETRAIN_DIR.iterdir())
    
    # Start retraining in background
    background_tasks.add_task(
        run_retraining,
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        model_type=request.model_type
    )
    
    return RetrainResponse(
        success=True,
        message="Retraining started. Monitor progress at /status endpoint.",
        task_id=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


@app.post("/upload/retrain_data")
async def upload_retrain_data(
    files: List[UploadFile] = File(...),
    class_name: str = Query(..., description="Class name for the uploaded images")
):
    """
    Upload images for retraining.
    
    - **files**: List of image files
    - **class_name**: The class/label for these images
    
    Images are saved to the retrain_data folder for later retraining.
    """
    # Create class directory
    class_dir = RETRAIN_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for file in files:
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            ext = Path(file.filename).suffix or ".jpg"
            new_filename = f"{timestamp}{ext}"
            
            file_path = class_dir / new_filename
            
            # Save file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            saved_files.append({
                "original_name": file.filename,
                "saved_as": new_filename,
                "class": class_name
            })
            
        except Exception as e:
            logger.error(f"Error saving file {file.filename}: {e}")
    
    return {
        "success": True,
        "message": f"Saved {len(saved_files)} files for retraining",
        "files": saved_files
    }


@app.get("/classes")
async def get_classes():
    """Get list of all plant disease classes."""
    if not app_state.class_indices:
        # Try to load from file
        class_info_path = DATA_DIR / "class_info.json"
        if class_info_path.exists():
            with open(class_info_path, 'r') as f:
                class_info = json.load(f)
                return {"classes": class_info.get("class_names", [])}
    
    return {
        "classes": list(app_state.class_indices.values()),
        "num_classes": len(app_state.class_indices)
    }


@app.get("/dataset/stats", response_model=DatasetStatsResponse)
async def get_dataset_stats():
    """Get dataset statistics."""
    def count_images(directory: Path) -> tuple:
        if not directory.exists():
            return 0, {}
        
        total = 0
        distribution = {}
        
        for class_dir in directory.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*")))
                distribution[class_dir.name] = count
                total += count
        
        return total, distribution
    
    train_count, train_dist = count_images(DATA_DIR / "train")
    val_count, _ = count_images(DATA_DIR / "val")
    test_count, _ = count_images(DATA_DIR / "test")
    
    return DatasetStatsResponse(
        total_train_images=train_count,
        total_val_images=val_count,
        total_test_images=test_count,
        num_classes=len(train_dist),
        class_distribution=train_dist
    )


@app.get("/model/info")
async def get_model_info():
    """Get information about the current model."""
    metadata_path = MODELS_DIR / "model_metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    
    return {
        "status": "no_model_info",
        "model_loaded": app_state.model is not None
    }


@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics."""
    metrics_path = MODELS_DIR / "evaluation_results.json"
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    
    return {"status": "no_metrics_available"}


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
