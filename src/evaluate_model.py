"""
Model Evaluation for PlantVillage Classification
=================================================
This module handles model evaluation, metrics calculation,
and visualization of results.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras

from data_pipeline import (
    create_data_generators,
    load_class_mapping,
    MODELS_DIR,
    DATA_DIR,
    BASE_DIR,
    IMG_SIZE,
    BATCH_SIZE
)
from model_pipeline import load_model


def evaluate_model(
    model: Optional[keras.Model] = None,
    model_path: Optional[str] = None,
    save_results: bool = True
) -> Dict:
    """
    Evaluate the model on the test set.
    
    Args:
        model: Trained Keras model (if None, loads from model_path)
        model_path: Path to saved model
        save_results: Whether to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load model if not provided
    if model is None:
        model = load_model(model_path)
    
    # Create test data generator
    _, _, test_gen = create_data_generators(batch_size=BATCH_SIZE)
    
    # Get class names
    class_indices = test_gen.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    num_classes = len(class_names)
    
    # Get predictions
    print("Generating predictions on test set...")
    test_gen.reset()
    
    # Get all predictions
    predictions = model.predict(test_gen, verbose=1)
    y_pred_proba = predictions
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = test_gen.classes
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        "num_samples": len(y_true),
        "num_classes": num_classes
    }
    
    # Calculate ROC-AUC for multi-class
    try:
        y_true_binarized = label_binarize(y_true, classes=list(range(num_classes)))
        metrics["roc_auc_macro"] = float(roc_auc_score(
            y_true_binarized, y_pred_proba, average='macro', multi_class='ovr'
        ))
        metrics["roc_auc_weighted"] = float(roc_auc_score(
            y_true_binarized, y_pred_proba, average='weighted', multi_class='ovr'
        ))
    except Exception as e:
        print(f"Warning: Could not calculate ROC-AUC: {e}")
        metrics["roc_auc_macro"] = None
        metrics["roc_auc_weighted"] = None
    
    # Per-class metrics
    class_report = classification_report(
        y_true, y_pred, 
        target_names=[class_names[i] for i in range(num_classes)],
        output_dict=True,
        zero_division=0
    )
    metrics["per_class_metrics"] = class_report
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
    if metrics["roc_auc_macro"]:
        print(f"ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")
    print("=" * 50)
    
    # Save results
    if save_results:
        results_path = MODELS_DIR / "evaluation_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_metrics = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
            json_metrics['confusion_matrix'] = metrics['confusion_matrix']
            json.dump(json_metrics, f, indent=2, default=str)
        print(f"\nResults saved to {results_path}")
    
    return metrics


def plot_confusion_matrix(
    confusion_mat: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 16),
    normalize: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        confusion_mat: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the figure
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
        
    Returns:
        Matplotlib figure
    """
    if normalize:
        confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
        confusion_mat = np.nan_to_num(confusion_mat)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        confusion_mat,
        annot=False,  # Too many classes for annotations
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
    
    # Rotate labels for readability
    plt.xticks(rotation=90, ha='center', fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    n_classes_to_plot: int = 10
) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        save_path: Path to save the figure
        n_classes_to_plot: Number of classes to plot (top N by sample count)
        
    Returns:
        Matplotlib figure
    """
    num_classes = len(class_names)
    y_true_binarized = label_binarize(y_true, classes=list(range(num_classes)))
    
    # Calculate ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculate macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot macro-average
    ax.plot(
        fpr["macro"], tpr["macro"],
        label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
        color='navy', linestyle='--', linewidth=2
    )
    
    # Plot top N classes by AUC
    sorted_classes = sorted(range(num_classes), key=lambda i: roc_auc[i], reverse=True)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes_to_plot))
    
    for idx, i in enumerate(sorted_classes[:n_classes_to_plot]):
        ax.plot(
            fpr[i], tpr[i],
            color=colors[idx],
            lw=1.5,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})'
        )
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (Top 10 Classes by AUC)', fontsize=14)
    ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    return fig


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history (accuracy and loss curves).
    
    Args:
        history: Training history dictionary
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Accuracy plot
    ax1 = axes[0]
    ax1.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2 = axes[1]
    ax2.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    return fig


def plot_class_performance(
    metrics: Dict,
    save_path: Optional[str] = None,
    top_n: int = 20
) -> plt.Figure:
    """
    Plot per-class performance metrics.
    
    Args:
        metrics: Evaluation metrics dictionary
        save_path: Path to save the figure
        top_n: Number of classes to show
        
    Returns:
        Matplotlib figure
    """
    per_class = metrics.get('per_class_metrics', {})
    
    # Extract class metrics (exclude aggregate metrics)
    class_data = {k: v for k, v in per_class.items() 
                  if k not in ['accuracy', 'macro avg', 'weighted avg']}
    
    if not class_data:
        print("No per-class metrics available")
        return None
    
    # Sort by F1-score
    sorted_classes = sorted(class_data.items(), key=lambda x: x[1].get('f1-score', 0), reverse=True)
    
    # Take top and bottom N/2
    top_classes = sorted_classes[:top_n//2]
    bottom_classes = sorted_classes[-(top_n//2):]
    selected_classes = top_classes + bottom_classes
    
    class_names = [c[0] for c in selected_classes]
    precision = [c[1].get('precision', 0) for c in selected_classes]
    recall = [c[1].get('recall', 0) for c in selected_classes]
    f1 = [c[1].get('f1-score', 0) for c in selected_classes]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax.barh(x - width, precision, width, label='Precision', color='steelblue')
    bars2 = ax.barh(x, recall, width, label='Recall', color='coral')
    bars3 = ax.barh(x + width, f1, width, label='F1-Score', color='seagreen')
    
    ax.set_xlabel('Score')
    ax.set_ylabel('Class')
    ax.set_title(f'Per-Class Performance (Top {top_n//2} & Bottom {top_n//2})')
    ax.set_yticks(x)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.legend()
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Class performance plot saved to {save_path}")
    
    return fig


def plot_prediction_confidence(
    y_pred_proba: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot prediction confidence distribution.
    
    Args:
        y_pred_proba: Predicted probabilities
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Get max confidence for each prediction
    max_confidence = np.max(y_pred_proba, axis=1)
    
    # Separate correct and incorrect predictions
    correct_mask = y_true == y_pred
    correct_confidence = max_confidence[correct_mask]
    incorrect_confidence = max_confidence[~correct_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confidence distribution
    ax1 = axes[0]
    ax1.hist(correct_confidence, bins=50, alpha=0.7, label='Correct', color='green', density=True)
    ax1.hist(incorrect_confidence, bins=50, alpha=0.7, label='Incorrect', color='red', density=True)
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Density')
    ax1.set_title('Prediction Confidence Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy by confidence threshold
    ax2 = axes[1]
    thresholds = np.linspace(0, 1, 50)
    accuracies = []
    coverage = []
    
    for thresh in thresholds:
        mask = max_confidence >= thresh
        if mask.sum() > 0:
            acc = (y_true[mask] == y_pred[mask]).mean()
            cov = mask.mean()
        else:
            acc = 0
            cov = 0
        accuracies.append(acc)
        coverage.append(cov)
    
    ax2.plot(thresholds, accuracies, 'b-', label='Accuracy', linewidth=2)
    ax2.plot(thresholds, coverage, 'r--', label='Coverage', linewidth=2)
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('Value')
    ax2.set_title('Accuracy vs Coverage at Different Confidence Thresholds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confidence plot saved to {save_path}")
    
    return fig


def generate_full_evaluation_report(
    model: Optional[keras.Model] = None,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Generate a complete evaluation report with all visualizations.
    
    Args:
        model: Trained Keras model
        model_path: Path to saved model
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary containing all evaluation results
    """
    if output_dir is None:
        output_dir = MODELS_DIR / "evaluation_report"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    if model is None:
        model = load_model(model_path)
    
    # Create test data generator
    _, _, test_gen = create_data_generators(batch_size=BATCH_SIZE)
    
    # Get class names
    class_indices = test_gen.class_indices
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
    
    # Get predictions
    print("Generating predictions...")
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred_proba = predictions
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # Calculate metrics
    metrics = evaluate_model(model, save_results=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        cm, class_names,
        save_path=str(output_dir / "confusion_matrix.png")
    )
    
    # 2. ROC curves
    plot_roc_curves(
        y_true, y_pred_proba, class_names,
        save_path=str(output_dir / "roc_curves.png")
    )
    
    # 3. Per-class performance
    plot_class_performance(
        metrics,
        save_path=str(output_dir / "class_performance.png")
    )
    
    # 4. Prediction confidence
    plot_prediction_confidence(
        y_pred_proba, y_true, y_pred,
        save_path=str(output_dir / "confidence_distribution.png")
    )
    
    # 5. Training history (if available)
    history_path = MODELS_DIR / "plant_disease_resnet50_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        plot_training_history(
            history,
            save_path=str(output_dir / "training_history.png")
        )
    
    # Generate text report
    report_text = f"""
===============================================================================
                     PLANT DISEASE CLASSIFICATION EVALUATION REPORT
===============================================================================

Model Evaluation Summary
------------------------
- Accuracy: {metrics['accuracy']:.4f}
- Precision (macro): {metrics['precision_macro']:.4f}
- Precision (weighted): {metrics['precision_weighted']:.4f}
- Recall (macro): {metrics['recall_macro']:.4f}
- Recall (weighted): {metrics['recall_weighted']:.4f}
- F1-Score (macro): {metrics['f1_macro']:.4f}
- F1-Score (weighted): {metrics['f1_weighted']:.4f}
- ROC-AUC (macro): {metrics.get('roc_auc_macro', 'N/A')}
- ROC-AUC (weighted): {metrics.get('roc_auc_weighted', 'N/A')}

Dataset Information
-------------------
- Number of test samples: {metrics['num_samples']}
- Number of classes: {metrics['num_classes']}

Generated Visualizations
------------------------
1. confusion_matrix.png - Confusion matrix heatmap
2. roc_curves.png - ROC curves for top performing classes
3. class_performance.png - Per-class precision, recall, F1-score
4. confidence_distribution.png - Prediction confidence analysis
5. training_history.png - Training/validation accuracy and loss curves

===============================================================================
"""
    
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\nEvaluation report generated at: {output_dir}")
    print(report_text)
    
    return metrics


if __name__ == "__main__":
    print("PlantVillage Model Evaluation")
    print("=" * 50)
    
    # Generate full evaluation report
    metrics = generate_full_evaluation_report()
