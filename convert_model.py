"""
Convert Keras model to H5 format for better compatibility.
"""
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')

# Load the checkpoint model
print("Loading model...")
model = tf.keras.models.load_model(
    'models/plant_disease_resnet50_checkpoint_01.keras', 
    safe_mode=False, 
    compile=False
)
print(f'Model loaded: {model.name}')
print(f'Input shape: {model.input_shape}')
print(f'Output shape: {model.output_shape}')

# Save in H5 format (more compatible across versions)
print("Saving as H5 format...")
model.save('models/plant_disease_model.h5')
print('Saved: models/plant_disease_model.h5')

# Check file sizes
import os
h5_size = os.path.getsize('models/plant_disease_model.h5') / (1024*1024)
print(f'H5 file size: {h5_size:.1f} MB')
print("\nUpload 'models/plant_disease_model.h5' to Hugging Face")
