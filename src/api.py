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

import ssl
import certifi

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Memory optimization for TensorFlow - MUST be before importing tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import tensorflow as tf
from tensorflow import keras

# Configure TensorFlow for low memory environments
tf.config.set_soft_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Limit CPU memory usage
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

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
    retrain_cancelled: bool = False  # Flag to cancel retraining
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
# Hugging Face model URL (direct download link)
# Use .keras format for Keras 3 compatibility
HUGGINGFACE_MODEL_URL = "https://huggingface.co/mathiaskabango/plantvillagev2/resolve/main/plant_disease_model_v2.keras"

# Model configuration - must match the model architecture
MODEL_INPUT_SIZE = (190, 190)  # Input size used during training


def download_model_from_url(url: str, dest_path: Path) -> bool:
    """Download model from external URL if not exists locally."""
    import requests as req
    
    if dest_path.exists():
        file_size = dest_path.stat().st_size
        # Check if file is valid (at least 1MB for a real model)
        if file_size > 1_000_000:
            logger.info(f"Model already exists at {dest_path} ({file_size/1_000_000:.1f} MB)")
            return True
        else:
            logger.warning(f"Model file exists but seems corrupted ({file_size} bytes), re-downloading...")
            dest_path.unlink()
    
    try:
        logger.info(f"Downloading model from {url}...")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use requests library which handles redirects properly
        response = req.get(
            url,
            allow_redirects=True,
            stream=True,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; PlantDiseaseAPI/1.0)'},
            timeout=300  # 5 minute timeout for large files
        )
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Downloading {total_size/1_000_000:.1f} MB...")
        
        # Download with progress
        downloaded = 0
        with open(dest_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:  # Log every 10MB
                        logger.info(f"Downloaded {downloaded/1_000_000:.1f}/{total_size/1_000_000:.1f} MB")
        
        logger.info(f"Model downloaded successfully to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        # Clean up partial download
        if dest_path.exists():
            dest_path.unlink()
        return False


def load_model_and_classes():
    """Load the trained model and class mappings."""
    try:
        # Always use the hardcoded HuggingFace URL (ignore environment variable)
        model_url = HUGGINGFACE_MODEL_URL
        default_model_path = MODELS_DIR / "plant_disease_model_v2.keras"
        
        # Clean up any old/corrupted model files
        if MODELS_DIR.exists():
            for old_file in MODELS_DIR.glob("*.h5"):
                logger.info(f"Removing old model file: {old_file}")
                old_file.unlink()
            for old_file in MODELS_DIR.glob("*.keras"):
                if old_file.name != "plant_disease_model_v2.keras":
                    logger.info(f"Removing old model file: {old_file}")
                    old_file.unlink()
        
        # Check if existing file is corrupted (too small)
        if default_model_path.exists():
            file_size = default_model_path.stat().st_size
            if file_size < 50_000_000:  # Less than 50MB is definitely wrong for this model
                logger.warning(f"Model file seems corrupted ({file_size} bytes), removing...")
                default_model_path.unlink()
        
        # Download if model doesn't exist locally
        if not default_model_path.exists():
            logger.info("Model not found locally, attempting to download from Hugging Face...")
            logger.info(f"URL: {model_url}")
            download_model_from_url(model_url, default_model_path)
        
        # Use only the specific model file
        model_path = default_model_path if default_model_path.exists() else None
        
        if model_path is None:
            logger.warning(f"No trained model found in {MODELS_DIR}. Available files: {list(MODELS_DIR.glob('*')) if MODELS_DIR.exists() else 'DIR NOT FOUND'}")
            logger.warning("Set MODEL_DOWNLOAD_URL environment variable to download the model automatically.")
            return None, {}
        
        logger.info(f"Found model: {model_path}")
        logger.info(f"Loading model from {model_path}")
        # Load model - .keras format (no Lambda layer, so safe_mode not needed)
        try:
            model = keras.models.load_model(str(model_path), compile=False)
            logger.info(f"Model loaded successfully - Input shape: {model.input_shape}, Output shape: {model.output_shape}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None, {}
        
        # Recompile with minimal settings for inference only
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # Clear any cached data
        keras.backend.clear_session()
        
        # Default class names for the 15-class PlantVillage model
        DEFAULT_CLASS_NAMES = {
            0: "Apple___Apple_scab",
            1: "Apple___Black_rot",
            2: "Apple___Cedar_apple_rust",
            3: "Apple___healthy",
            4: "Blueberry___healthy",
            5: "Cherry_(including_sour)___Powdery_mildew",
            6: "Cherry_(including_sour)___healthy",
            7: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            8: "Corn_(maize)___Common_rust_",
            9: "Corn_(maize)___Northern_Leaf_Blight",
            10: "Corn_(maize)___healthy",
            11: "Grape___Black_rot",
            12: "Grape___Esca_(Black_Measles)",
            13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            14: "Grape___healthy"
        }
        
        # Load class indices from file or use defaults
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
                # Use default class names
                logger.info("Using default class names (no index_to_class.json found)")
                class_indices = DEFAULT_CLASS_NAMES
        
        logger.info(f"Model loaded successfully with {len(class_indices)} classes")
        return model, class_indices
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, {}


def preprocess_image(image_bytes: bytes, img_size: tuple = None) -> np.ndarray:
    """Preprocess image for model inference with ResNet50 preprocessing."""
    if img_size is None:
        img_size = MODEL_INPUT_SIZE
    
    # Open and resize image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model's expected input size
    image = image.resize(img_size, Image.Resampling.LANCZOS)
    
    # Convert to array
    img_array = np.array(image, dtype=np.float32)
    
    # Apply ResNet50 preprocessing (since we removed the Lambda layer from the model)
    # This converts RGB to BGR and zero-centers each color channel
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


async def run_retraining(
    epochs: int = 10,
    learning_rate: float = 0.0001,
    model_type: str = "resnet50",
    use_retrain_data: bool = False
):
    """Background task for model retraining."""
    global app_state
    
    try:
        app_state.is_retraining = True
        app_state.retrain_status = "Starting retraining..."
        app_state.retrain_progress = 0.0
        
        logger.info(f"Starting retraining with epochs={epochs}, lr={learning_rate}, use_retrain_data={use_retrain_data}")
        
        if use_retrain_data:
            # Lightweight retraining using uploaded retrain_data
            result = await run_lightweight_retrain(epochs, learning_rate)
        else:
            # Full retraining using train/val directories
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


async def run_lightweight_retrain(epochs: int = 2, learning_rate: float = 0.0001):
    """
    Lightweight retraining using uploaded retrain_data.
    Fine-tunes the existing model on new data.
    Designed to work on Railway (CPU, limited memory).
    """
    global app_state
    
    try:
        app_state.retrain_status = "Preparing retrain data..."
        app_state.retrain_progress = 0.1
        
        # Check retrain_data directory
        if not RETRAIN_DIR.exists():
            return {"success": False, "error": "No retrain_data directory found"}
        
        # Get all classes and images
        classes = []
        all_images = []
        all_labels = []
        
        for class_dir in sorted(RETRAIN_DIR.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                classes.append(class_name)
                class_idx = len(classes) - 1
                
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                        all_images.append(str(img_path))
                        all_labels.append(class_idx)
        
        if not all_images:
            return {"success": False, "error": "No images found in retrain_data/"}
        
        logger.info(f"Found {len(all_images)} images in {len(classes)} classes")
        app_state.retrain_status = f"Found {len(all_images)} images in {len(classes)} classes"
        app_state.retrain_progress = 0.2
        
        # Check if we have the current model
        if app_state.model is None:
            return {"success": False, "error": "No model loaded. Cannot fine-tune."}
        
        # Check class compatibility
        model_num_classes = app_state.model.output_shape[-1]
        if len(classes) > model_num_classes:
            return {"success": False, "error": f"Too many classes ({len(classes)}). Model supports {model_num_classes} classes."}
        
        app_state.retrain_status = "Loading and preprocessing images..."
        app_state.retrain_progress = 0.3
        await asyncio.sleep(0.1)
        
        # Load images in batches to save memory
        batch_size = 16
        X_data = []
        y_data = []
        
        for i, (img_path, label) in enumerate(zip(all_images, all_labels)):
            # Check for cancellation
            if app_state.retrain_cancelled:
                return {"success": False, "error": "Training cancelled by user."}
            
            try:
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()
                img_array = preprocess_image(img_bytes)[0]  # Remove batch dim
                X_data.append(img_array)
                y_data.append(label)
                
                if (i + 1) % 50 == 0:
                    progress = 0.3 + (0.2 * (i + 1) / len(all_images))
                    app_state.retrain_status = f"Loaded {i + 1}/{len(all_images)} images..."
                    app_state.retrain_progress = progress
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
                continue
        
        if len(X_data) < 2:
            return {"success": False, "error": "Not enough valid images loaded"}
        
        X_data = np.array(X_data)
        y_data = tf.keras.utils.to_categorical(y_data, num_classes=model_num_classes)
        
        logger.info(f"Data shape: X={X_data.shape}, y={y_data.shape}")
        
        app_state.retrain_status = "Starting fine-tuning..."
        app_state.retrain_progress = 0.5
        await asyncio.sleep(0.1)
        
        # Create a copy of the model for fine-tuning
        # Freeze early layers, only train the last few layers
        model = app_state.model
        
        # Freeze all layers except the last 10
        for layer in model.layers[:-10]:
            layer.trainable = False
        for layer in model.layers[-10:]:
            layer.trainable = True
        
        # Compile with low learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with progress callback
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                # Check for cancellation
                if app_state.retrain_cancelled:
                    self.model.stop_training = True
                    app_state.retrain_status = "Training cancelled by user."
                    return
                    
                progress = 0.5 + (0.4 * (epoch + 1) / epochs)
                acc = logs.get('accuracy', 0) * 100 if logs else 0
                loss = logs.get('loss', 0) if logs else 0
                app_state.retrain_status = f"Epoch {epoch + 1}/{epochs} - Acc: {acc:.1f}%, Loss: {loss:.4f}"
                app_state.retrain_progress = progress
                logger.info(f"Epoch {epoch + 1}/{epochs} - acc: {acc:.1f}%, loss: {loss:.4f}")
            
            def on_batch_end(self, batch, logs=None):
                # Check for cancellation between batches
                if app_state.retrain_cancelled:
                    self.model.stop_training = True
        
        # Run training in executor to not block
        loop = asyncio.get_event_loop()
        
        def train_model():
            history = model.fit(
                X_data, y_data,
                epochs=epochs,
                batch_size=min(batch_size, len(X_data)),
                validation_split=0.2 if len(X_data) > 10 else 0.0,
                callbacks=[ProgressCallback()],
                verbose=0
            )
            return history
        
        history = await loop.run_in_executor(None, train_model)
        
        # Check if cancelled
        if app_state.retrain_cancelled:
            app_state.retrain_status = "Training cancelled by user."
            return {"success": False, "error": "Training cancelled by user."}
        
        app_state.retrain_status = "Saving model..."
        app_state.retrain_progress = 0.95
        
        # Save the fine-tuned model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "plant_disease_model_v2.keras"
        model.save(str(model_path))
        
        # Update class indices if we have new classes
        # Map the classes from retrain_data to existing class names
        for i, class_name in enumerate(classes):
            if i not in app_state.class_indices:
                app_state.class_indices[i] = class_name
        
        final_acc = history.history.get('accuracy', [0])[-1] * 100
        final_loss = history.history.get('loss', [0])[-1]
        
        logger.info(f"Fine-tuning complete! Final accuracy: {final_acc:.1f}%")
        
        return {
            "success": True,
            "message": f"Fine-tuning complete! Accuracy: {final_acc:.1f}%, Loss: {final_loss:.4f}",
            "metrics": {
                "accuracy": final_acc,
                "loss": final_loss,
                "epochs": epochs,
                "images_trained": len(X_data),
                "classes": classes
            }
        }
        
    except Exception as e:
        logger.error(f"Lightweight retrain error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


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
    
    **Note for cloud deployments (Railway):**
    - Retraining on Railway is not recommended (no GPU, very slow)
    - Use /upload/retrain_data to collect new training images
    - Download collected images and retrain locally or on Kaggle/Colab with GPU
    - Upload new model to Hugging Face for automatic deployment
    
    **Demo Mode:** If train/val directories don't exist but retrain_data has images,
    the API will do lightweight fine-tuning on the uploaded images (works on Railway CPU).
    """
    if app_state.is_retraining:
        return RetrainResponse(
            success=False,
            message="Retraining already in progress. Check /status for progress."
        )
    
    # Check if training data exists
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    
    # Check retrain_data for uploaded images
    retrain_count = 0
    retrain_classes = []
    if RETRAIN_DIR.exists():
        for class_dir in RETRAIN_DIR.iterdir():
            if class_dir.is_dir():
                class_images = len(list(class_dir.glob("*")))
                if class_images > 0:
                    retrain_count += class_images
                    retrain_classes.append(class_dir.name)
    
    use_retrain_data = False
    
    if not train_dir.exists() or not val_dir.exists():
        # No train/val directories - check if we can use retrain_data
        if retrain_count > 0:
            # Use lightweight fine-tuning on retrain_data
            use_retrain_data = True
            logger.info(f"Using lightweight fine-tuning with {retrain_count} images from retrain_data")
        else:
            return RetrainResponse(
                success=False,
                message="No training data available. Upload images via POST /upload/retrain_data first."
            )
    else:
        # Check if directories have data
        train_classes = list(train_dir.iterdir()) if train_dir.exists() else []
        val_classes = list(val_dir.iterdir()) if val_dir.exists() else []
        
        if not train_classes or not val_classes:
            if retrain_count > 0:
                use_retrain_data = True
            else:
                return RetrainResponse(
                    success=False,
                    message="Training directories are empty. Upload images via POST /upload/retrain_data first."
                )
    
    # Limit epochs for cloud deployment
    max_epochs = min(request.epochs, 5)  # Cap at 5 epochs on Railway
    
    # Reset cancel flag
    app_state.retrain_cancelled = False
    
    # Start retraining in background
    background_tasks.add_task(
        run_retraining,
        epochs=max_epochs,
        learning_rate=request.learning_rate,
        model_type=request.model_type,
        use_retrain_data=use_retrain_data
    )
    
    mode = "fine-tuning on uploaded images" if use_retrain_data else "full training"
    
    return RetrainResponse(
        success=True,
        message=f"Retraining started ({mode}, {max_epochs} epochs, {retrain_count} images in {len(retrain_classes)} classes). Monitor progress at /status endpoint.",
        task_id=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


@app.post("/retrain/cancel")
async def cancel_retrain():
    """
    Cancel an ongoing retraining process.
    
    Returns immediately but the actual cancellation may take a moment
    as it waits for the current batch/epoch to complete.
    """
    if not app_state.is_retraining:
        return {
            "success": False,
            "message": "No retraining in progress."
        }
    
    app_state.retrain_cancelled = True
    app_state.retrain_status = "Cancellation requested... waiting for current operation to complete."
    
    return {
        "success": True,
        "message": "Cancellation requested. Retraining will stop after the current batch completes."
    }


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


@app.post("/upload/retrain_zip")
async def upload_retrain_zip(
    file: UploadFile = File(..., description="ZIP file containing training images organized in class folders")
):
    """
    Upload a ZIP file with training images.
    
    The ZIP should be organized as:
    ```
    retrain_data.zip
    ├── Tomato___Late_blight/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── Apple___healthy/
    │   └── image3.jpg
    └── ...
    ```
    
    Each folder name becomes the class label.
    """
    import zipfile
    import tempfile
    
    if not file.filename or not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")
    
    # Save uploaded ZIP to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    extracted_count = 0
    classes_found = {}
    
    try:
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            for zip_info in zip_ref.infolist():
                # Skip directories and hidden files
                if zip_info.is_dir() or zip_info.filename.startswith('__') or '/.DS_Store' in zip_info.filename:
                    continue
                
                # Parse path: folder/class_name/image.jpg or class_name/image.jpg
                parts = zip_info.filename.split('/')
                
                # Find the class name and filename
                if len(parts) >= 2:
                    # Could be retrain_data/class_name/file.jpg or class_name/file.jpg
                    if parts[0] in ['retrain_data', 'train', 'data']:
                        class_name = parts[1]
                        img_filename = parts[-1]
                    else:
                        class_name = parts[0]
                        img_filename = parts[-1]
                else:
                    continue
                
                # Check if it's an image
                if not img_filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    continue
                
                # Create class directory
                class_dir = RETRAIN_DIR / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract with unique name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                ext = Path(img_filename).suffix
                new_filename = f"{timestamp}{ext}"
                dest_path = class_dir / new_filename
                
                # Extract file
                with zip_ref.open(zip_info) as src, open(dest_path, 'wb') as dst:
                    dst.write(src.read())
                
                extracted_count += 1
                classes_found[class_name] = classes_found.get(class_name, 0) + 1
        
        return {
            "success": True,
            "message": f"Extracted {extracted_count} images from ZIP",
            "classes": classes_found,
            "total_images": extracted_count
        }
        
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        logger.error(f"Error extracting ZIP: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract ZIP: {str(e)}")
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/retrain_data/stats")
async def get_retrain_data_stats():
    """
    Get statistics about uploaded retrain data.
    
    Returns count of images per class in the retrain_data directory.
    """
    stats = {
        "total_images": 0,
        "classes": {},
        "ready_for_download": False
    }
    
    if not RETRAIN_DIR.exists():
        return stats
    
    for class_dir in RETRAIN_DIR.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*"))
            image_count = len([f for f in images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']])
            if image_count > 0:
                stats["classes"][class_dir.name] = image_count
                stats["total_images"] += image_count
    
    stats["ready_for_download"] = stats["total_images"] > 0
    
    return stats


@app.get("/retrain_data/download")
async def download_retrain_data():
    """
    Download all uploaded retrain data as a ZIP file.
    
    Use this to download images uploaded via /upload/retrain_data,
    then retrain the model locally or on Kaggle/Colab with GPU.
    """
    import zipfile
    import tempfile
    from fastapi.responses import FileResponse
    
    if not RETRAIN_DIR.exists():
        raise HTTPException(status_code=404, detail="No retrain data available")
    
    # Check if there are any images
    total_images = 0
    for class_dir in RETRAIN_DIR.iterdir():
        if class_dir.is_dir():
            total_images += len(list(class_dir.glob("*")))
    
    if total_images == 0:
        raise HTTPException(status_code=404, detail="No images in retrain_data directory")
    
    # Create a temporary ZIP file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        zip_path = tmp_file.name
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for class_dir in RETRAIN_DIR.iterdir():
            if class_dir.is_dir():
                for img_file in class_dir.iterdir():
                    if img_file.is_file():
                        arcname = f"retrain_data/{class_dir.name}/{img_file.name}"
                        zipf.write(img_file, arcname)
    
    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename=f"retrain_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    )


@app.delete("/retrain_data/clear")
async def clear_retrain_data():
    """
    Clear all uploaded retrain data.
    
    Use after downloading and successfully retraining the model.
    """
    import shutil
    
    if not RETRAIN_DIR.exists():
        return {"success": True, "message": "Retrain data directory does not exist"}
    
    cleared_count = 0
    for class_dir in RETRAIN_DIR.iterdir():
        if class_dir.is_dir():
            cleared_count += len(list(class_dir.glob("*")))
            shutil.rmtree(class_dir)
    
    return {
        "success": True,
        "message": f"Cleared {cleared_count} images from retrain data"
    }


@app.post("/model/reload")
async def reload_model():
    """
    Reload the model from Hugging Face.
    
    Use after uploading a new model to Hugging Face to update the deployed model.
    This will download the latest model version.
    """
    global app_state
    
    # Delete existing model to force re-download
    model_path = MODELS_DIR / "plant_disease_model_v2.keras"
    if model_path.exists():
        model_path.unlink()
        logger.info(f"Deleted existing model: {model_path}")
    
    # Reload model
    app_state.model, app_state.class_indices = load_model_and_classes()
    
    if app_state.model is not None:
        app_state.model_loaded_at = datetime.now()
        return {
            "success": True,
            "message": "Model reloaded successfully from Hugging Face",
            "model_loaded_at": app_state.model_loaded_at.isoformat(),
            "num_classes": len(app_state.class_indices)
        }
    else:
        return {
            "success": False,
            "message": "Failed to reload model. Check logs for details."
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


@app.get("/debug")
async def debug_info():
    """Debug endpoint to check model loading status."""
    import os as os_module
    
    # Check directories
    models_exists = MODELS_DIR.exists()
    data_exists = DATA_DIR.exists()
    
    # List files in models directory
    model_files = []
    if models_exists:
        model_files = [{"name": f.name, "size": f.stat().st_size} for f in MODELS_DIR.iterdir()]
    
    # List files in data directory
    data_files = []
    if data_exists:
        data_files = [f.name for f in DATA_DIR.iterdir() if f.is_file()]
    
    # Check environment - IGNORE env var, always use the correct URL
    model_url = HUGGINGFACE_MODEL_URL  # Always use the hardcoded correct URL
    
    # Try to load model now
    load_error = None
    model_load_error = None
    try:
        test_path = MODELS_DIR / "plant_disease_resnet50_best.keras"
        # Check if file exists and is valid (>1MB)
        if test_path.exists() and test_path.stat().st_size < 1_000_000:
            logger.warning(f"Model file is corrupted ({test_path.stat().st_size} bytes), removing...")
            test_path.unlink()
        
        if not test_path.exists():
            # Try downloading
            download_model_from_url(model_url, test_path)
        
        # Try to actually load the model
        if test_path.exists() and app_state.model is None:
            try:
                logger.info("Attempting to load model...")
                # safe_mode=False needed because model contains Lambda layers
                # compile=False reduces memory usage significantly
                loaded_model = keras.models.load_model(str(test_path), safe_mode=False, compile=False)
                loaded_model.compile(optimizer='adam', loss='categorical_crossentropy')
                app_state.model = loaded_model
                app_state.model_loaded_at = datetime.now()
                logger.info("Model loaded successfully!")
                
                # Clear memory
                keras.backend.clear_session()
                
                # Load class indices
                class_indices_path = DATA_DIR / "index_to_class.json"
                if class_indices_path.exists():
                    with open(class_indices_path, 'r') as f:
                        app_state.class_indices = {int(k): v for k, v in json.load(f).items()}
                    logger.info(f"Loaded {len(app_state.class_indices)} classes")
            except Exception as model_err:
                model_load_error = str(model_err)
                logger.error(f"Failed to load model: {model_err}")
    except Exception as e:
        load_error = str(e)
    
    return {
        "base_dir": str(BASE_DIR),
        "models_dir": str(MODELS_DIR),
        "models_dir_exists": models_exists,
        "data_dir_exists": data_exists,
        "model_files": model_files,
        "data_files": data_files,
        "model_url": model_url,
        "model_loaded": app_state.model is not None,
        "load_error": load_error,
        "model_load_error": model_load_error,
        "class_indices_count": len(app_state.class_indices),
        "huggingface_url": HUGGINGFACE_MODEL_URL
    }


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
