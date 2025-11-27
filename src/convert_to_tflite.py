"""
Convert Keras model to TensorFlow Lite for low-memory deployment.
Run this locally, then upload the .tflite file to Hugging Face.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path

# Paths
MODEL_PATH = Path("models/plant_disease_resnet50_checkpoint_01.keras")
OUTPUT_PATH = Path("models/plant_disease_model.tflite")

print("Loading Keras model...")
model = tf.keras.models.load_model(str(MODEL_PATH), safe_mode=False, compile=False)

print("Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimize for size and speed
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # Use float16 for smaller size

# Convert
tflite_model = converter.convert()

# Save
OUTPUT_PATH.parent.mkdir(exist_ok=True)
with open(OUTPUT_PATH, 'wb') as f:
    f.write(tflite_model)

original_size = MODEL_PATH.stat().st_size / (1024 * 1024)
tflite_size = OUTPUT_PATH.stat().st_size / (1024 * 1024)

print(f"\nConversion complete!")
print(f"Original model: {original_size:.1f} MB")
print(f"TFLite model:   {tflite_size:.1f} MB")
print(f"Size reduction: {(1 - tflite_size/original_size) * 100:.1f}%")
print(f"\nSaved to: {OUTPUT_PATH}")
print("\nNext steps:")
print("1. Upload plant_disease_model.tflite to Hugging Face")
print("2. Update api.py to use TFLite interpreter")
