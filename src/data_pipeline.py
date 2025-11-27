"""
Data Pipeline for PlantVillage Dataset
=======================================
This module handles data loading, preprocessing, augmentation, 
and train/val/test splitting for plant disease classification.
"""

import os
import shutil
import random
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
RETRAIN_DIR = DATA_DIR / "retrain_data"
SOURCE_DATASET_DIR = BASE_DIR / "plantvillagedataset"
MODELS_DIR = BASE_DIR / "models"


def create_directory_structure():
    """Create the required directory structure for the project."""
    directories = [
        DATA_DIR,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        RETRAIN_DIR,
        MODELS_DIR,
        BASE_DIR / "logs",
        BASE_DIR / "mlruns"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created/verified directory: {directory}")


def get_class_names(source_dir: Path = SOURCE_DATASET_DIR) -> List[str]:
    """Get all class names from the source dataset directory."""
    classes = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
    return classes


def get_dataset_statistics(source_dir: Path = SOURCE_DATASET_DIR) -> Dict:
    """
    Calculate dataset statistics including class distribution.
    
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "total_images": 0,
        "num_classes": 0,
        "class_distribution": {},
        "plant_types": {},
        "disease_types": {}
    }
    
    classes = get_class_names(source_dir)
    stats["num_classes"] = len(classes)
    
    for class_name in classes:
        class_dir = source_dir / class_name
        num_images = len(list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg")) + 
                        list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg")))
        stats["class_distribution"][class_name] = num_images
        stats["total_images"] += num_images
        
        # Parse plant and disease from class name
        parts = class_name.split("___")
        if len(parts) == 2:
            plant, disease = parts
            stats["plant_types"][plant] = stats["plant_types"].get(plant, 0) + num_images
            stats["disease_types"][disease] = stats["disease_types"].get(disease, 0) + num_images
    
    return stats


def split_dataset(
    source_dir: Path = SOURCE_DATASET_DIR,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = RANDOM_SEED,
    force_resplit: bool = False
) -> Dict[str, int]:
    """
    Split the PlantVillage dataset into train, validation, and test sets.
    
    Args:
        source_dir: Path to source dataset
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation  
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility
        force_resplit: If True, resplit even if data already exists
        
    Returns:
        Dictionary with split statistics
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Check if split already exists
    if not force_resplit and TRAIN_DIR.exists() and any(TRAIN_DIR.iterdir()):
        print("Dataset already split. Use force_resplit=True to resplit.")
        return {"status": "already_split"}
    
    # Create directories
    create_directory_structure()
    
    # Clear existing data if force resplit
    if force_resplit:
        for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
    
    random.seed(random_seed)
    split_stats = {"train": 0, "val": 0, "test": 0}
    
    classes = get_class_names(source_dir)
    print(f"Found {len(classes)} classes")
    
    for class_name in tqdm(classes, desc="Splitting dataset"):
        class_dir = source_dir / class_name
        
        # Get all image files
        images = []
        for ext in ["*.JPG", "*.jpg", "*.png", "*.jpeg", "*.JPEG"]:
            images.extend(list(class_dir.glob(ext)))
        
        if len(images) == 0:
            print(f"Warning: No images found in {class_name}")
            continue
            
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        n_train = int(len(images) * train_ratio)
        n_val = int(len(images) * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Create class directories and copy images
        for split_name, split_images in [("train", train_images), 
                                          ("val", val_images), 
                                          ("test", test_images)]:
            split_dir = DATA_DIR / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in split_images:
                dest_path = split_dir / img_path.name
                shutil.copy2(img_path, dest_path)
                split_stats[split_name] += 1
    
    # Save split info
    split_info = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_seed": random_seed,
        "split_stats": split_stats,
        "num_classes": len(classes),
        "class_names": classes
    }
    
    with open(DATA_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\nDataset split complete:")
    print(f"  Train: {split_stats['train']} images")
    print(f"  Validation: {split_stats['val']} images")
    print(f"  Test: {split_stats['test']} images")
    
    return split_stats


def create_data_generators(
    img_size: Tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    augment_train: bool = True
) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
    """
    Create data generators for train, validation, and test sets with augmentation.
    
    Args:
        img_size: Target image size (height, width)
        batch_size: Batch size for generators
        augment_train: Whether to apply augmentation to training data
        
    Returns:
        Tuple of (train_generator, val_generator, test_generator)
    """
    # Training data generator with augmentation
    if augment_train:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2]
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    # Validation and test generators - only rescale
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=RANDOM_SEED
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Save class indices mapping
    class_indices = train_generator.class_indices
    with open(DATA_DIR / "class_indices.json", "w") as f:
        json.dump(class_indices, f, indent=2)
    
    # Save reverse mapping (index to class name)
    index_to_class = {v: k for k, v in class_indices.items()}
    with open(DATA_DIR / "index_to_class.json", "w") as f:
        json.dump(index_to_class, f, indent=2)
    
    return train_generator, val_generator, test_generator


def create_tf_dataset(
    img_size: Tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    augment_train: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict]:
    """
    Create tf.data.Dataset objects for better performance.
    
    Returns:
        Tuple of (train_ds, val_ds, test_ds, class_info)
    """
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=RANDOM_SEED
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    class_info = {
        "class_names": class_names,
        "num_classes": num_classes
    }
    
    # Save class info
    with open(DATA_DIR / "class_info.json", "w") as f:
        json.dump(class_info, f, indent=2)
    
    # Normalization layer
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    # Data augmentation layer
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])
    
    def preprocess_train(image, label):
        image = normalization_layer(image)
        if augment_train:
            image = data_augmentation(image, training=True)
        return image, label
    
    def preprocess_val_test(image, label):
        image = normalization_layer(image)
        return image, label
    
    # Apply preprocessing and optimize performance
    train_ds = train_ds.map(preprocess_train, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    val_ds = val_ds.map(preprocess_val_test, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    test_ds = test_ds.map(preprocess_val_test, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_info


def preprocess_single_image(
    image_path: str,
    img_size: Tuple[int, int] = IMG_SIZE
) -> np.ndarray:
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
        img_size: Target image size
        
    Returns:
        Preprocessed image array ready for model prediction
    """
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=img_size
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def preprocess_image_bytes(
    image_bytes: bytes,
    img_size: Tuple[int, int] = IMG_SIZE
) -> np.ndarray:
    """
    Preprocess image from bytes (for API usage).
    
    Args:
        image_bytes: Image data as bytes
        img_size: Target image size
        
    Returns:
        Preprocessed image array ready for model prediction
    """
    img = tf.image.decode_image(image_bytes, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0  # Normalize
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img.numpy()


def visualize_dataset_distribution(save_path: Optional[str] = None):
    """
    Create visualizations of dataset distribution.
    
    Args:
        save_path: Path to save the visualization (optional)
    """
    stats = get_dataset_statistics()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Class distribution
    ax1 = axes[0, 0]
    classes = list(stats["class_distribution"].keys())
    counts = list(stats["class_distribution"].values())
    
    # Sort by count for better visualization
    sorted_indices = np.argsort(counts)[::-1]
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    
    ax1.barh(range(len(sorted_classes)), sorted_counts, color='steelblue')
    ax1.set_yticks(range(len(sorted_classes)))
    ax1.set_yticklabels(sorted_classes, fontsize=8)
    ax1.set_xlabel('Number of Images')
    ax1.set_title('Class Distribution')
    ax1.invert_yaxis()
    
    # 2. Plant type distribution
    ax2 = axes[0, 1]
    plants = list(stats["plant_types"].keys())
    plant_counts = list(stats["plant_types"].values())
    
    ax2.pie(plant_counts, labels=plants, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Distribution by Plant Type')
    
    # 3. Disease type distribution
    ax3 = axes[1, 0]
    diseases = list(stats["disease_types"].keys())
    disease_counts = list(stats["disease_types"].values())
    
    sorted_idx = np.argsort(disease_counts)[::-1]
    sorted_diseases = [diseases[i] for i in sorted_idx]
    sorted_disease_counts = [disease_counts[i] for i in sorted_idx]
    
    ax3.bar(range(len(sorted_diseases)), sorted_disease_counts, color='coral')
    ax3.set_xticks(range(len(sorted_diseases)))
    ax3.set_xticklabels(sorted_diseases, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Number of Images')
    ax3.set_title('Distribution by Disease Type')
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    Dataset Summary
    ===============
    
    Total Images: {stats['total_images']:,}
    Number of Classes: {stats['num_classes']}
    Number of Plant Types: {len(stats['plant_types'])}
    Number of Disease Types: {len(stats['disease_types'])}
    
    Avg Images per Class: {stats['total_images'] // stats['num_classes']:,}
    Max Images in a Class: {max(counts):,}
    Min Images in a Class: {min(counts):,}
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    return fig


def check_retrain_data() -> Dict:
    """
    Check for new data in the retrain_data folder.
    
    Returns:
        Dictionary with retrain data statistics
    """
    retrain_stats = {
        "has_new_data": False,
        "num_images": 0,
        "class_distribution": {}
    }
    
    if not RETRAIN_DIR.exists():
        RETRAIN_DIR.mkdir(parents=True, exist_ok=True)
        return retrain_stats
    
    for class_dir in RETRAIN_DIR.iterdir():
        if class_dir.is_dir():
            images = []
            for ext in ["*.JPG", "*.jpg", "*.png", "*.jpeg", "*.JPEG"]:
                images.extend(list(class_dir.glob(ext)))
            
            if images:
                retrain_stats["class_distribution"][class_dir.name] = len(images)
                retrain_stats["num_images"] += len(images)
    
    retrain_stats["has_new_data"] = retrain_stats["num_images"] > 0
    
    return retrain_stats


def add_retrain_data_to_training(clear_after: bool = True) -> bool:
    """
    Move retrain data to training set.
    
    Args:
        clear_after: Whether to clear retrain folder after moving
        
    Returns:
        True if data was added successfully
    """
    retrain_stats = check_retrain_data()
    
    if not retrain_stats["has_new_data"]:
        print("No new data to add for retraining")
        return False
    
    for class_name, count in retrain_stats["class_distribution"].items():
        src_dir = RETRAIN_DIR / class_name
        dest_dir = TRAIN_DIR / class_name
        
        # Create destination if it doesn't exist
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in src_dir.iterdir():
            if img_path.is_file():
                dest_path = dest_dir / img_path.name
                shutil.copy2(img_path, dest_path)
        
        if clear_after:
            shutil.rmtree(src_dir)
    
    print(f"Added {retrain_stats['num_images']} images to training set")
    return True


def load_class_mapping() -> Dict[int, str]:
    """Load the class index to name mapping."""
    mapping_file = DATA_DIR / "index_to_class.json"
    if mapping_file.exists():
        with open(mapping_file, "r") as f:
            mapping = json.load(f)
            # Convert string keys to integers
            return {int(k): v for k, v in mapping.items()}
    return {}


if __name__ == "__main__":
    # Example usage
    print("PlantVillage Data Pipeline")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    
    # Get dataset statistics
    print("\nDataset Statistics:")
    stats = get_dataset_statistics()
    print(f"Total images: {stats['total_images']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"Number of plant types: {len(stats['plant_types'])}")
    
    # Split dataset
    print("\nSplitting dataset...")
    split_stats = split_dataset(force_resplit=False)
    
    # Create data generators
    if split_stats.get("status") != "already_split" or True:
        print("\nCreating data generators...")
        train_gen, val_gen, test_gen = create_data_generators()
        print(f"Train samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Test samples: {test_gen.samples}")
        print(f"Number of classes: {train_gen.num_classes}")
    
    # Visualize dataset
    print("\nGenerating visualizations...")
    visualize_dataset_distribution(save_path=str(BASE_DIR / "dataset_distribution.png"))
