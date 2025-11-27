"""
Model Pipeline for PlantVillage Classification
===============================================
This module handles model creation, training, and saving using 
transfer learning with ResNet50.
"""

import os
import json
import datetime
import ssl
import certifi
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)
from tensorflow.keras.optimizers import Adam
import mlflow
import mlflow.keras

from data_pipeline import (
    create_data_generators, 
    create_tf_dataset,
    IMG_SIZE, 
    BATCH_SIZE,
    MODELS_DIR,
    DATA_DIR,
    BASE_DIR
)


# Configuration
EPOCHS = 50
LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.0001
FINE_TUNE_EPOCHS = 20


def create_resnet50_model(
    num_classes: int,
    img_size: Tuple[int, int] = IMG_SIZE,
    dropout_rate: float = 0.5,
    freeze_base: bool = True
) -> Model:
    """
    Create a ResNet50-based transfer learning model.
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size (height, width)
        dropout_rate: Dropout rate for regularization
        freeze_base: Whether to freeze the base model weights
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained ResNet50 without top layers
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )
    
    # Freeze base model if specified
    base_model.trainable = not freeze_base
    
    # Build custom top layers
    inputs = keras.Input(shape=(*img_size, 3))
    
    # Data augmentation layers (applied during training)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.2)(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Custom classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model


def create_mobilenetv2_model(
    num_classes: int,
    img_size: Tuple[int, int] = IMG_SIZE,
    dropout_rate: float = 0.5,
    freeze_base: bool = True
) -> Model:
    """
    Create a MobileNetV2-based transfer learning model (lighter alternative).
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )
    
    base_model.trainable = not freeze_base
    
    inputs = keras.Input(shape=(*img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model


def create_efficientnet_model(
    num_classes: int,
    img_size: Tuple[int, int] = IMG_SIZE,
    dropout_rate: float = 0.5,
    freeze_base: bool = True
) -> Model:
    """
    Create an EfficientNetB0-based transfer learning model.
    """
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )
    
    base_model.trainable = not freeze_base
    
    inputs = keras.Input(shape=(*img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model


def create_custom_cnn_model(
    num_classes: int,
    img_size: Tuple[int, int] = IMG_SIZE,
    dropout_rate: float = 0.5
) -> Model:
    """
    Create a custom CNN model from scratch (for baseline comparison).
    """
    inputs = keras.Input(shape=(*img_size, 3))
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 4
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model


def compile_model(
    model: Model,
    learning_rate: float = LEARNING_RATE,
    use_class_weights: bool = False
) -> Model:
    """
    Compile the model with optimizer and loss function.
    """
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc', multi_label=True)
        ]
    )
    
    return model


def get_callbacks(
    model_name: str = "plant_disease_model",
    patience: int = 10,
    log_dir: Optional[str] = None
) -> list:
    """
    Create training callbacks for model checkpointing, early stopping, etc.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if log_dir is None:
        log_dir = str(BASE_DIR / "logs" / f"{model_name}_{timestamp}")
    
    callbacks = [
        # Save best model based on validation accuracy
        ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{model_name}_best.keras"),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Save model checkpoints during training
        ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{model_name}_checkpoint_{{epoch:02d}}.keras"),
            monitor='val_loss',
            save_freq='epoch',
            save_best_only=False,
            verbose=0
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        
        # CSV logging
        CSVLogger(
            str(MODELS_DIR / f"{model_name}_training_log.csv"),
            separator=',',
            append=True
        )
    ]
    
    return callbacks


def calculate_class_weights(train_generator) -> Dict[int, float]:
    """
    Calculate class weights to handle imbalanced classes.
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Get class labels
    labels = train_generator.classes
    classes = np.unique(labels)
    
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    
    return dict(enumerate(class_weights))


def train_model(
    model_type: str = "resnet50",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    use_class_weights: bool = True,
    fine_tune: bool = True,
    fine_tune_epochs: int = FINE_TUNE_EPOCHS,
    use_mlflow: bool = True
) -> Tuple[Model, Dict]:
    """
    Train the model with the specified configuration.
    
    Args:
        model_type: Type of model ('resnet50', 'mobilenetv2', 'efficientnet', 'custom_cnn')
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        use_class_weights: Whether to use class weights for imbalanced data
        fine_tune: Whether to fine-tune the base model after initial training
        fine_tune_epochs: Number of epochs for fine-tuning
        use_mlflow: Whether to log to MLflow
        
    Returns:
        Trained model and training history
    """
    print(f"Starting training with {model_type} model...")
    
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators(batch_size=batch_size)
    num_classes = train_gen.num_classes
    
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    
    # Create model based on type
    model_creators = {
        "resnet50": create_resnet50_model,
        "mobilenetv2": create_mobilenetv2_model,
        "efficientnet": create_efficientnet_model,
        "custom_cnn": create_custom_cnn_model
    }
    
    if model_type not in model_creators:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_creators.keys())}")
    
    model = model_creators[model_type](num_classes, freeze_base=True)
    model = compile_model(model, learning_rate=learning_rate)
    
    model.summary()
    
    # Calculate class weights if needed
    class_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights(train_gen)
        print(f"Using class weights for {len(class_weights)} classes")
    
    # Get callbacks
    callbacks = get_callbacks(model_name=f"plant_disease_{model_type}")
    
    # MLflow tracking
    if use_mlflow:
        mlflow.set_tracking_uri(str(BASE_DIR / "mlruns"))
        mlflow.set_experiment("plant_disease_classification")
        
        with mlflow.start_run(run_name=f"{model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params({
                "model_type": model_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_classes": num_classes,
                "img_size": IMG_SIZE,
                "use_class_weights": use_class_weights,
                "fine_tune": fine_tune
            })
            
            # Phase 1: Train with frozen base
            print("\nPhase 1: Training with frozen base model...")
            history1 = model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                class_weight=class_weights
            )
            
            # Phase 2: Fine-tuning (unfreeze top layers of base model)
            if fine_tune and model_type != "custom_cnn":
                print("\nPhase 2: Fine-tuning base model...")
                
                # Unfreeze base model
                base_model = model.layers[4] if model_type == "resnet50" else model.layers[1]
                base_model.trainable = True
                
                # Freeze early layers, fine-tune later layers
                if model_type == "resnet50":
                    for layer in base_model.layers[:100]:
                        layer.trainable = False
                elif model_type == "mobilenetv2":
                    for layer in base_model.layers[:100]:
                        layer.trainable = False
                elif model_type == "efficientnet":
                    for layer in base_model.layers[:200]:
                        layer.trainable = False
                
                # Recompile with lower learning rate
                model = compile_model(model, learning_rate=FINE_TUNE_LEARNING_RATE)
                
                # Continue training
                history2 = model.fit(
                    train_gen,
                    epochs=fine_tune_epochs,
                    validation_data=val_gen,
                    callbacks=callbacks,
                    class_weight=class_weights
                )
                
                # Combine histories
                history = {
                    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
                    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
                    'loss': history1.history['loss'] + history2.history['loss'],
                    'val_loss': history1.history['val_loss'] + history2.history['val_loss']
                }
            else:
                history = history1.history
            
            # Log metrics
            final_metrics = {
                "final_train_accuracy": history['accuracy'][-1],
                "final_val_accuracy": history['val_accuracy'][-1],
                "final_train_loss": history['loss'][-1],
                "final_val_loss": history['val_loss'][-1],
                "best_val_accuracy": max(history['val_accuracy'])
            }
            mlflow.log_metrics(final_metrics)
            
            # Log model
            mlflow.keras.log_model(model, "model")
            
            print(f"\nTraining complete!")
            print(f"Final validation accuracy: {final_metrics['final_val_accuracy']:.4f}")
            print(f"Best validation accuracy: {final_metrics['best_val_accuracy']:.4f}")
    
    else:
        # Train without MLflow
        history1 = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        if fine_tune and model_type != "custom_cnn":
            base_model = model.layers[4] if model_type == "resnet50" else model.layers[1]
            base_model.trainable = True
            
            for layer in base_model.layers[:100]:
                layer.trainable = False
            
            model = compile_model(model, learning_rate=FINE_TUNE_LEARNING_RATE)
            
            history2 = model.fit(
                train_gen,
                epochs=fine_tune_epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                class_weight=class_weights
            )
            
            history = {
                'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
                'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
                'loss': history1.history['loss'] + history2.history['loss'],
                'val_loss': history1.history['val_loss'] + history2.history['val_loss']
            }
        else:
            history = history1.history
    
    # Save final model
    final_model_path = MODELS_DIR / f"plant_disease_{model_type}_final.keras"
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Save training history
    history_path = MODELS_DIR / f"plant_disease_{model_type}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save model metadata
    metadata = {
        "model_type": model_type,
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
        "class_indices": train_gen.class_indices,
        "trained_at": datetime.datetime.now().isoformat(),
        "final_accuracy": float(history['val_accuracy'][-1]),
        "best_accuracy": float(max(history['val_accuracy']))
    }
    
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model, history


def load_model(model_path: Optional[str] = None) -> Model:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the model file. If None, loads the best model.
        
    Returns:
        Loaded Keras model
    """
    if model_path is None:
        # Try to find the best model
        possible_paths = [
            MODELS_DIR / "plant_disease_resnet50_best.keras",
            MODELS_DIR / "plant_disease_mobilenetv2_best.keras",
            MODELS_DIR / "plant_disease_efficientnet_best.keras",
            MODELS_DIR / "plant_disease_resnet50_final.keras",
        ]
        
        for path in possible_paths:
            if path.exists():
                model_path = str(path)
                break
        
        if model_path is None:
            raise FileNotFoundError("No trained model found. Please train a model first.")
    
    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path)
    
    return model


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the currently saved model.
    """
    metadata_path = MODELS_DIR / "model_metadata.json"
    
    if not metadata_path.exists():
        return {"status": "no_model_found"}
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Check if model file exists
    model_files = list(MODELS_DIR.glob("*.keras"))
    metadata["available_models"] = [str(f.name) for f in model_files]
    
    return metadata


if __name__ == "__main__":
    print("PlantVillage Model Pipeline")
    print("=" * 50)
    
    # Train model
    model, history = train_model(
        model_type="resnet50",
        epochs=30,
        batch_size=32,
        learning_rate=0.001,
        use_class_weights=True,
        fine_tune=True,
        fine_tune_epochs=15,
        use_mlflow=True
    )
    
    print("\nTraining complete!")
    print(f"Final accuracy: {history['val_accuracy'][-1]:.4f}")
