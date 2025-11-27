"""
Automated Retraining Script for PlantVillage Classification
============================================================
This module handles automated model retraining when new data
is added or when triggered manually.
"""

import os
import sys
import json
import shutil
import datetime
import logging
from pathlib import Path
from typing import Dict, Optional, Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
RETRAIN_DIR = DATA_DIR / "retrain_data"

# Configuration - must match deployed model
IMG_SIZE = (190, 190)  # Model input size
BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.0001


class RetrainDataHandler(FileSystemEventHandler):
    """Watches for new files in the retrain_data directory."""
    
    def __init__(self, threshold: int = 10, callback: Optional[Callable] = None):
        """
        Args:
            threshold: Minimum number of new images before triggering retrain
            callback: Function to call when threshold is reached
        """
        self.new_files_count = 0
        self.threshold = threshold
        self.callback = callback
        self.last_retrain = datetime.datetime.now()
        self.cooldown_minutes = 30  # Minimum time between retrains
        
    def on_created(self, event):
        if isinstance(event, FileCreatedEvent):
            # Check if it's an image file
            if event.src_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                self.new_files_count += 1
                logger.info(f"New image detected: {event.src_path}")
                logger.info(f"New files count: {self.new_files_count}/{self.threshold}")
                
                # Check if we should trigger retrain
                if self.new_files_count >= self.threshold:
                    time_since_last = datetime.datetime.now() - self.last_retrain
                    if time_since_last.total_seconds() >= self.cooldown_minutes * 60:
                        logger.info("Threshold reached! Triggering automatic retraining...")
                        self.trigger_retrain()
                    else:
                        logger.info(f"Cooldown active. {self.cooldown_minutes - time_since_last.seconds//60} minutes remaining.")
    
    def trigger_retrain(self):
        """Trigger the retraining process."""
        self.new_files_count = 0
        self.last_retrain = datetime.datetime.now()
        
        if self.callback:
            self.callback()
        else:
            # Default retrain
            retrain_model()


def check_for_new_data() -> Dict:
    """
    Check the retrain_data folder for new images.
    
    Returns:
        Dictionary with statistics about new data
    """
    stats = {
        "has_new_data": False,
        "num_images": 0,
        "classes": {},
        "new_classes": []
    }
    
    if not RETRAIN_DIR.exists():
        RETRAIN_DIR.mkdir(parents=True, exist_ok=True)
        return stats
    
    # Get existing classes
    existing_classes = set()
    if TRAIN_DIR.exists():
        existing_classes = {d.name for d in TRAIN_DIR.iterdir() if d.is_dir()}
    
    # Check retrain data
    for class_dir in RETRAIN_DIR.iterdir():
        if class_dir.is_dir():
            images = []
            for ext in ["*.JPG", "*.jpg", "*.png", "*.jpeg", "*.JPEG"]:
                images.extend(list(class_dir.glob(ext)))
            
            if images:
                stats["classes"][class_dir.name] = len(images)
                stats["num_images"] += len(images)
                
                if class_dir.name not in existing_classes:
                    stats["new_classes"].append(class_dir.name)
    
    stats["has_new_data"] = stats["num_images"] > 0
    
    return stats


def merge_retrain_data(clear_after: bool = True) -> Dict:
    """
    Merge retrain data into the training set.
    
    Args:
        clear_after: Whether to clear the retrain folder after merging
        
    Returns:
        Dictionary with merge statistics
    """
    stats = check_for_new_data()
    
    if not stats["has_new_data"]:
        logger.info("No new data to merge")
        return {"merged": 0}
    
    merged_count = 0
    
    for class_name, count in stats["classes"].items():
        src_dir = RETRAIN_DIR / class_name
        dest_dir = TRAIN_DIR / class_name
        
        # Create destination if it doesn't exist
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        for img_path in src_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                # Generate unique filename to avoid conflicts
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                new_name = f"retrain_{timestamp}{img_path.suffix}"
                dest_path = dest_dir / new_name
                
                shutil.copy2(img_path, dest_path)
                merged_count += 1
        
        # Clear source if requested
        if clear_after:
            shutil.rmtree(src_dir)
    
    logger.info(f"Merged {merged_count} images into training set")
    
    return {
        "merged": merged_count,
        "classes": list(stats["classes"].keys()),
        "new_classes": stats["new_classes"]
    }


def create_data_generators(batch_size: int = BATCH_SIZE):
    """Create data generators for training and validation."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.resnet50 import preprocess_input
    
    # Training data generator with augmentation and ResNet50 preprocessing
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation data generator - only ResNet50 preprocessing
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator


def retrain_model(
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    model_type: str = "resnet50",
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict:
    """
    Retrain the model with new data.
    
    Args:
        epochs: Number of training epochs
        learning_rate: Learning rate for fine-tuning
        model_type: Type of model to train
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Dictionary with retraining results
    """
    result = {
        "success": False,
        "message": "",
        "metrics": {}
    }
    
    def update_progress(progress: float, status: str):
        if progress_callback:
            progress_callback(progress, status)
        logger.info(f"[{progress*100:.0f}%] {status}")
    
    try:
        update_progress(0.0, "Starting retraining process...")
        
        # Step 1: Merge new data
        update_progress(0.05, "Checking for new data...")
        merge_stats = merge_retrain_data(clear_after=True)
        
        if merge_stats["merged"] > 0:
            update_progress(0.1, f"Merged {merge_stats['merged']} new images")
        else:
            update_progress(0.1, "No new data to merge, continuing with existing data")
        
        # Step 2: Create data generators
        update_progress(0.15, "Creating data generators...")
        train_gen, val_gen = create_data_generators()
        num_classes = train_gen.num_classes
        
        logger.info(f"Training samples: {train_gen.samples}")
        logger.info(f"Validation samples: {val_gen.samples}")
        logger.info(f"Number of classes: {num_classes}")
        
        # Step 3: Load or create model
        update_progress(0.2, "Loading model...")
        
        # Try to load existing model
        model_path = MODELS_DIR / f"plant_disease_{model_type}_best.keras"
        if not model_path.exists():
            model_path = MODELS_DIR / f"plant_disease_{model_type}_final.keras"
        
        if model_path.exists():
            logger.info(f"Loading existing model from {model_path}")
            model = keras.models.load_model(str(model_path))
            
            # Check if number of classes matches
            if model.output_shape[-1] != num_classes:
                logger.warning(f"Number of classes changed from {model.output_shape[-1]} to {num_classes}")
                logger.info("Creating new model with updated number of classes...")
                model = create_new_model(num_classes, model_type)
        else:
            logger.info("No existing model found, creating new model...")
            model = create_new_model(num_classes, model_type)
        
        # Step 4: Compile model for fine-tuning
        update_progress(0.25, "Compiling model...")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Step 5: Setup callbacks
        update_progress(0.3, "Setting up training...")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(MODELS_DIR / f"plant_disease_{model_type}_best.keras"),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ProgressCallback(progress_callback, start_progress=0.3, end_progress=0.9)
        ]
        
        # Step 6: Train model
        update_progress(0.35, f"Starting training for {epochs} epochs...")
        
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Step 7: Save model and metadata
        update_progress(0.9, "Saving model...")
        
        # Save final model
        final_model_path = MODELS_DIR / f"plant_disease_{model_type}_final.keras"
        model.save(final_model_path)
        
        # Save versioned model
        versioned_path = MODELS_DIR / f"plant_disease_{model_type}_{timestamp}.keras"
        model.save(versioned_path)
        
        # Save training history
        history_path = MODELS_DIR / f"plant_disease_{model_type}_history.json"
        
        # Load existing history if exists and append
        if history_path.exists():
            with open(history_path, 'r') as f:
                existing_history = json.load(f)
            # Append new history
            for key in history.history:
                if key in existing_history:
                    existing_history[key].extend(history.history[key])
                else:
                    existing_history[key] = history.history[key]
        else:
            existing_history = history.history
        
        with open(history_path, 'w') as f:
            json.dump(existing_history, f, indent=2)
        
        # Update metadata
        metadata = {
            "model_type": model_type,
            "num_classes": num_classes,
            "img_size": IMG_SIZE,
            "class_indices": train_gen.class_indices,
            "trained_at": timestamp,
            "final_accuracy": float(history.history['val_accuracy'][-1]),
            "best_accuracy": float(max(history.history['val_accuracy'])),
            "retrain_epochs": epochs,
            "learning_rate": learning_rate
        }
        
        metadata_path = MODELS_DIR / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save class mappings
        class_indices = train_gen.class_indices
        with open(DATA_DIR / "class_indices.json", 'w') as f:
            json.dump(class_indices, f, indent=2)
        
        index_to_class = {v: k for k, v in class_indices.items()}
        with open(DATA_DIR / "index_to_class.json", 'w') as f:
            json.dump(index_to_class, f, indent=2)
        
        update_progress(1.0, "Retraining completed successfully!")
        
        result["success"] = True
        result["message"] = "Model retrained successfully"
        result["metrics"] = {
            "final_val_accuracy": float(history.history['val_accuracy'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "best_val_accuracy": float(max(history.history['val_accuracy'])),
            "epochs_trained": len(history.history['accuracy'])
        }
        result["model_path"] = str(final_model_path)
        result["version"] = timestamp
        
        logger.info(f"Retraining completed! Best accuracy: {result['metrics']['best_val_accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        result["success"] = False
        result["error"] = str(e)
        if progress_callback:
            progress_callback(0.0, f"Error: {str(e)}")
    
    return result


def create_new_model(num_classes: int, model_type: str = "resnet50") -> keras.Model:
    """Create a new model based on the specified type."""
    from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
    from tensorflow.keras import layers
    
    if model_type == "resnet50":
        base_model = ResNet50(weights='imagenet', include_top=False, 
                             input_shape=(*IMG_SIZE, 3))
    elif model_type == "mobilenetv2":
        base_model = MobileNetV2(weights='imagenet', include_top=False,
                                input_shape=(*IMG_SIZE, 3))
    elif model_type == "efficientnet":
        base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                   input_shape=(*IMG_SIZE, 3))
    else:
        # Default to ResNet50
        base_model = ResNet50(weights='imagenet', include_top=False,
                             input_shape=(*IMG_SIZE, 3))
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model


class ProgressCallback(keras.callbacks.Callback):
    """Custom callback to report training progress."""
    
    def __init__(self, callback: Optional[Callable], 
                 start_progress: float = 0.0, 
                 end_progress: float = 1.0):
        super().__init__()
        self.callback = callback
        self.start_progress = start_progress
        self.end_progress = end_progress
        self.total_epochs = None
        
    def on_train_begin(self, logs=None):
        self.total_epochs = self.params.get('epochs', 1)
        
    def on_epoch_end(self, epoch, logs=None):
        if self.callback and self.total_epochs:
            progress_range = self.end_progress - self.start_progress
            progress = self.start_progress + (progress_range * (epoch + 1) / self.total_epochs)
            status = f"Epoch {epoch + 1}/{self.total_epochs} - " \
                    f"loss: {logs.get('loss', 0):.4f} - " \
                    f"accuracy: {logs.get('accuracy', 0):.4f} - " \
                    f"val_loss: {logs.get('val_loss', 0):.4f} - " \
                    f"val_accuracy: {logs.get('val_accuracy', 0):.4f}"
            self.callback(progress, status)


def start_file_watcher(threshold: int = 10):
    """
    Start watching the retrain_data folder for new images.
    
    Args:
        threshold: Number of new images before auto-retrain is triggered
    """
    RETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    
    event_handler = RetrainDataHandler(threshold=threshold)
    observer = Observer()
    observer.schedule(event_handler, str(RETRAIN_DIR), recursive=True)
    observer.start()
    
    logger.info(f"Started watching {RETRAIN_DIR} for new images...")
    logger.info(f"Auto-retrain will trigger after {threshold} new images")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("File watcher stopped")
    
    observer.join()


def scheduled_retrain():
    """Run retraining on a schedule (e.g., daily)."""
    logger.info("Running scheduled retraining...")
    
    # Check if there's new data
    stats = check_for_new_data()
    
    if stats["has_new_data"]:
        logger.info(f"Found {stats['num_images']} new images for retraining")
        result = retrain_model()
        return result
    else:
        logger.info("No new data for retraining")
        return {"success": True, "message": "No new data to retrain"}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plant Disease Model Retraining")
    parser.add_argument("--mode", choices=["retrain", "watch", "check"], 
                       default="retrain", help="Operation mode")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--model", type=str, default="resnet50", 
                       help="Model type (resnet50, mobilenetv2, efficientnet)")
    parser.add_argument("--threshold", type=int, default=10,
                       help="Number of new images before auto-retrain (watch mode)")
    
    args = parser.parse_args()
    
    if args.mode == "retrain":
        print("Starting model retraining...")
        result = retrain_model(
            epochs=args.epochs,
            learning_rate=args.lr,
            model_type=args.model
        )
        print(f"\nResult: {result}")
        
    elif args.mode == "watch":
        print("Starting file watcher for automatic retraining...")
        start_file_watcher(threshold=args.threshold)
        
    elif args.mode == "check":
        print("Checking for new data...")
        stats = check_for_new_data()
        print(f"\nNew data statistics:")
        print(f"  Has new data: {stats['has_new_data']}")
        print(f"  Number of images: {stats['num_images']}")
        print(f"  Classes: {list(stats['classes'].keys())}")
        print(f"  New classes: {stats['new_classes']}")
