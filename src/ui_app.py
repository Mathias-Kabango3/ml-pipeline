"""
Streamlit UI Dashboard for Plant Disease Classification
========================================================
This module provides a web-based dashboard for monitoring,
prediction, and model retraining.
"""

import os
import io
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Configuration
API_URL = os.environ.get("API_URL", "https://ml-pipeline-production-be57.up.railway.app")
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Page configuration
st.set_page_config(
    page_title="Plant Disease Classification Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-degraded {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Helper functions
def get_api_status() -> Dict:
    """Get API status from the backend."""
    try:
        response = requests.get(f"{API_URL}/status", timeout=5)
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def predict_image(image_bytes: bytes) -> Dict:
    """Send image to API for prediction."""
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def trigger_retrain(epochs: int, learning_rate: float, model_type: str) -> Dict:
    """Trigger model retraining via API."""
    try:
        data = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "model_type": model_type
        }
        response = requests.post(f"{API_URL}/retrain", json=data, timeout=10)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def upload_retrain_images(files: List, class_name: str) -> Dict:
    """Upload images for retraining."""
    try:
        files_data = [("files", (f.name, f.read(), "image/jpeg")) for f in files]
        response = requests.post(
            f"{API_URL}/upload/retrain_data",
            files=files_data,
            params={"class_name": class_name},
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_dataset_stats() -> Dict:
    """Get dataset statistics from API."""
    try:
        response = requests.get(f"{API_URL}/dataset/stats", timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_model_metrics() -> Dict:
    """Get model metrics from API."""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def load_training_history() -> Optional[Dict]:
    """Load training history from local file."""
    history_files = list(MODELS_DIR.glob("*_history.json"))
    if history_files:
        with open(history_files[0], 'r') as f:
            return json.load(f)
    return None


def load_experiments_log() -> Optional[pd.DataFrame]:
    """Load experiments log from CSV file."""
    experiments_path = MODELS_DIR / "experiments_log.csv"
    if experiments_path.exists():
        df = pd.read_csv(experiments_path)
        return df
    return None


def load_local_metrics() -> Optional[Dict]:
    """Load evaluation results from local file."""
    metrics_path = MODELS_DIR / "evaluation_results.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


def load_class_distribution() -> Dict:
    """Load class distribution from data directory or split_info.json."""
    distribution = {}
    
    # First try to load from split_info.json (faster)
    split_info_path = DATA_DIR / "split_info.json"
    if split_info_path.exists():
        try:
            with open(split_info_path, 'r') as f:
                split_info = json.load(f)
                if "class_distribution" in split_info:
                    return split_info["class_distribution"]
                if "train_distribution" in split_info:
                    return split_info["train_distribution"]
        except Exception as e:
            pass
    
    # Try to load from class_indices.json
    class_indices_path = DATA_DIR / "class_indices.json"
    if class_indices_path.exists():
        try:
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
                # Create a dummy distribution showing class names exist
                return {name: 1 for name in class_indices.keys()}
        except Exception as e:
            pass
    
    # Fall back to scanning train directory
    train_dir = DATA_DIR / "train"
    if train_dir.exists():
        for class_dir in train_dir.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*")))
                distribution[class_dir.name] = count
    
    # Also try plantvillagedataset directory
    if not distribution:
        pvd_dir = BASE_DIR / "plantvillagedataset"
        if pvd_dir.exists():
            for class_dir in pvd_dir.iterdir():
                if class_dir.is_dir():
                    count = len(list(class_dir.glob("*")))
                    distribution[class_dir.name] = count
    
    return distribution


# Sidebar
st.sidebar.markdown("# Plant Disease AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predict", "Training Metrics", "Retrain", "Dataset"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### API Status")

# Check API status
api_status = get_api_status()
if api_status.get("status") == "error":
    st.sidebar.error("API Offline")
    st.sidebar.caption(f"Error: {api_status.get('error', 'Unknown')[:50]}...")
elif api_status.get("model_loaded"):
    st.sidebar.success("API Online")
    st.sidebar.caption("Model: Loaded")
    if api_status.get("uptime_seconds"):
        uptime = api_status["uptime_seconds"]
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        st.sidebar.caption(f"Uptime: {hours}h {minutes}m")
else:
    st.sidebar.warning("API Degraded")
    st.sidebar.caption("Model: Not loaded")


# Main content based on selected page
if page == "Dashboard":
    st.markdown('<h1 class="main-header">Plant Disease Classification Dashboard</h1>', unsafe_allow_html=True)
    
    # Load local data for fallback
    split_info_path = DATA_DIR / "split_info.json"
    local_num_classes = 0
    if split_info_path.exists():
        with open(split_info_path, 'r') as f:
            split_info = json.load(f)
            local_num_classes = split_info.get("num_classes", 0)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        api_online = api_status.get("model_loaded", False)
        st.metric(
            label="API Status",
            value="Online" if api_online else "Offline",
            delta="Healthy" if api_status.get("status") == "healthy" else ("Start API" if api_status.get("status") == "error" else None)
        )
    
    with col2:
        st.metric(
            label="Total Predictions",
            value=api_status.get("prediction_count", 0)
        )
    
    with col3:
        avg_time = api_status.get("avg_inference_time_ms", 0)
        st.metric(
            label="Avg Inference Time",
            value=f"{avg_time:.1f} ms" if avg_time > 0 else "N/A"
        )
    
    with col4:
        num_classes = api_status.get("num_classes", 0) or local_num_classes
        st.metric(
            label="Classes",
            value=num_classes
        )
    
    # Show API connection help if offline
    if api_status.get("status") == "error":
        st.warning("API is not running. Start it with: `python src/api.py`")
    
    st.markdown("---")
    
    # Training status
    if api_status.get("is_retraining"):
        st.warning("Retraining in progress...")
        progress = api_status.get("retrain_progress", 0)
        st.progress(progress)
        st.info(api_status.get("retrain_status", ""))
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Class Distribution")
        class_dist = load_class_distribution()
        
        # If no class distribution, try to load class names from split_info
        if not class_dist:
            split_info_path = DATA_DIR / "split_info.json"
            if split_info_path.exists():
                with open(split_info_path, 'r') as f:
                    split_info = json.load(f)
                    class_names = split_info.get("class_names", [])
                    if class_names:
                        # Show class names without counts
                        class_dist = {name: 1 for name in class_names}
        
        if class_dist:
            df = pd.DataFrame([
                {"Class": k, "Count": v} 
                for k, v in sorted(class_dist.items(), key=lambda x: x[1], reverse=True)[:15]
            ])
            fig = px.bar(df, x="Count", y="Class", orientation='h', 
                        title="Top 15 Classes by Sample Count")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No dataset loaded yet. Run data pipeline first.")
    
    with col2:
        st.subheader("Model Performance")
        
        # First try experiments log
        experiments_df = load_experiments_log()
        if experiments_df is not None and not experiments_df.empty:
            best_exp = experiments_df.loc[experiments_df['val_acc'].idxmax()]
            perf_data = {
                "Metric": ["Validation Accuracy", "Validation Loss"],
                "Value": [best_exp['val_acc'], 1 - best_exp['val_loss']]  # Invert loss for visualization
            }
            df = pd.DataFrame(perf_data)
            fig = px.bar(df, x="Metric", y="Value", color="Metric",
                        title=f"Best Model: {best_exp['model']}")
            fig.update_layout(height=400, showlegend=False)
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fall back to evaluation results
            metrics = load_local_metrics()
            if metrics:
                perf_data = {
                    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                    "Value": [
                        metrics.get("accuracy", 0),
                        metrics.get("precision_macro", 0),
                        metrics.get("recall_macro", 0),
                        metrics.get("f1_macro", 0)
                    ]
                }
                df = pd.DataFrame(perf_data)
                fig = px.bar(df, x="Metric", y="Value", color="Metric",
                            title="Model Metrics")
                fig.update_layout(height=400, showlegend=False)
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No evaluation results available. Train and evaluate a model first.")


elif page == "Predict":
    st.markdown('<h1 class="main-header">Plant Disease Prediction</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a plant leaf image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload an image of a plant leaf to classify"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("Prediction Results")
        
        if uploaded_file is not None:
            if st.button("Predict Disease", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Reset file position
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    
                    result = predict_image(image_bytes)
                    
                    if result.get("success"):
                        st.success(f"**Prediction:** {result['prediction']}")
                        st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                        st.info(f"Inference time: {result['inference_time_ms']:.1f} ms")
                        
                        # Show top predictions
                        st.subheader("Top Predictions")
                        predictions = result.get("all_predictions", {})
                        if predictions:
                            df = pd.DataFrame([
                                {"Disease": k, "Confidence": v*100}
                                for k, v in list(predictions.items())[:5]
                            ])
                            fig = px.bar(df, x="Confidence", y="Disease", 
                                        orientation='h', 
                                        title="Confidence Scores (%)")
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")
        else:
            st.info("Upload an image to get started")


elif page == "Training Metrics":
    st.markdown('<h1 class="main-header">Training Metrics & Visualizations</h1>', unsafe_allow_html=True)
    
    # Model Performance Images Section
    st.subheader("Model Performance (Training & Validation)")
    
    # Load experiments to get accuracy info
    experiments_df = load_experiments_log()
    
    # Display accuracy summary at the top
    if experiments_df is not None and not experiments_df.empty:
        best_exp = experiments_df.loc[experiments_df['val_acc'].idxmax()]
        best_acc = best_exp['val_acc'] * 100
        
        # Performance summary with color-coded status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Best Validation Accuracy",
                value=f"{best_acc:.2f}%",
                delta=f"{best_exp['model']} - {best_exp['variant']}"
            )
        
        with col2:
            # Performance rating based on accuracy
            if best_acc >= 98:
                rating = "Excellent"
                color = "green"
            elif best_acc >= 95:
                rating = "Very Good"
                color = "green"
            elif best_acc >= 90:
                rating = "Good"
                color = "orange"
            elif best_acc >= 80:
                rating = "Fair"
                color = "orange"
            else:
                rating = "Needs Improvement"
                color = "red"
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 10px; text-align: center;">
                <p style="margin: 0; font-size: 0.875rem; color: #666;">Model Rating</p>
                <p style="margin: 0; font-size: 1.5rem; font-weight: bold; color: {color};">{rating}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            val_loss = best_exp['val_loss']
            st.metric(
                label="Best Validation Loss",
                value=f"{val_loss:.4f}",
                delta=f"Train time: {best_exp['train_time_s']/60:.1f} min"
            )
        
        st.markdown("")
    
    # Check for performance images in modelperformance directory
    performance_dir = BASE_DIR / "modelperformance"
    accuracy_img_path = performance_dir / "accuracy.png"
    loss_img_path = performance_dir / "train_loss.png"
    
    if accuracy_img_path.exists() or loss_img_path.exists():
        col1, col2 = st.columns(2)
        
        with col1:
            if accuracy_img_path.exists():
                st.image(str(accuracy_img_path), caption="Training & Validation Accuracy", use_container_width=True)
            else:
                st.info("Accuracy plot not found")
        
        with col2:
            if loss_img_path.exists():
                st.image(str(loss_img_path), caption="Training & Validation Loss", use_container_width=True)
            else:
                st.info("Loss plot not found")
    else:
        st.info("No performance images found in `modelperformance/` directory. Add `accuracy.png` and `train_loss.png` to display training curves.")
    
    st.markdown("---")
    
    # Experiments Log from CSV
    st.subheader("Experiments Log")
    experiments_df = load_experiments_log()
    
    if experiments_df is not None and not experiments_df.empty:
        # Display summary metrics for best experiment
        best_exp = experiments_df.loc[experiments_df['val_acc'].idxmax()]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Model", best_exp['model'])
        with col2:
            st.metric("Best Val Accuracy", f"{best_exp['val_acc']*100:.2f}%")
        with col3:
            st.metric("Val Loss", f"{best_exp['val_loss']:.4f}")
        with col4:
            train_time_min = best_exp['train_time_s'] / 60
            st.metric("Train Time", f"{train_time_min:.1f} min")
        
        st.markdown("---")
        
        # Comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Validation Accuracy comparison
            fig = px.bar(
                experiments_df, 
                x='exp_id', 
                y='val_acc',
                color='model',
                title='Validation Accuracy by Experiment',
                labels={'val_acc': 'Validation Accuracy', 'exp_id': 'Experiment ID'}
            )
            fig.update_layout(xaxis_tickangle=-45, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Validation Loss comparison
            fig = px.bar(
                experiments_df, 
                x='exp_id', 
                y='val_loss',
                color='model',
                title='Validation Loss by Experiment',
                labels={'val_loss': 'Validation Loss', 'exp_id': 'Experiment ID'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Training time comparison
        st.subheader("Training Time Comparison")
        experiments_df['train_time_min'] = experiments_df['train_time_s'] / 60
        fig = px.bar(
            experiments_df,
            x='exp_id',
            y='train_time_min',
            color='model',
            title='Training Time by Experiment',
            labels={'train_time_min': 'Training Time (minutes)', 'exp_id': 'Experiment ID'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Full experiments table
        st.subheader("All Experiments")
        display_df = experiments_df[['exp_id', 'model', 'variant', 'val_acc', 'val_loss', 'train_time_s', 'notes']].copy()
        display_df['val_acc'] = display_df['val_acc'].apply(lambda x: f"{x*100:.2f}%")
        display_df['val_loss'] = display_df['val_loss'].apply(lambda x: f"{x:.4f}")
        display_df['train_time_s'] = display_df['train_time_s'].apply(lambda x: f"{x/60:.1f} min")
        display_df.columns = ['Experiment ID', 'Model', 'Variant', 'Val Accuracy', 'Val Loss', 'Train Time', 'Notes']
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No experiments log found. Add `experiments_log.csv` to the `models/` directory.")
    
    st.markdown("---")
    
    # Evaluation metrics
    st.subheader("Evaluation Metrics")
    metrics = load_local_metrics()
    
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
        with col2:
            st.metric("Precision (Macro)", f"{metrics.get('precision_macro', 0)*100:.2f}%")
        with col3:
            st.metric("Recall (Macro)", f"{metrics.get('recall_macro', 0)*100:.2f}%")
        with col4:
            st.metric("F1-Score (Macro)", f"{metrics.get('f1_macro', 0)*100:.2f}%")
        
        if metrics.get("roc_auc_macro"):
            st.metric("ROC-AUC (Macro)", f"{metrics.get('roc_auc_macro')*100:.2f}%")
        
        # Per-class performance
        st.subheader("Per-Class Performance")
        per_class = metrics.get("per_class_metrics", {})
        if per_class:
            class_data = []
            for class_name, class_metrics in per_class.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    class_data.append({
                        "Class": class_name,
                        "Precision": class_metrics.get("precision", 0),
                        "Recall": class_metrics.get("recall", 0),
                        "F1-Score": class_metrics.get("f1-score", 0),
                        "Support": class_metrics.get("support", 0)
                    })
            
            df = pd.DataFrame(class_data)
            df = df.sort_values("F1-Score", ascending=False)
            
            st.dataframe(df, use_container_width=True, height=400)
    else:
        st.info("No evaluation metrics available. Evaluate a model first.")


elif page == "Retrain":
    st.markdown('<h1 class="main-header">Model Retraining</h1>', unsafe_allow_html=True)
    
    # Retraining status
    if api_status.get("is_retraining"):
        st.warning("Retraining in progress...")
        progress = api_status.get("retrain_progress", 0)
        st.progress(progress)
        st.info(api_status.get("retrain_status", ""))
        st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Training Data")
        st.markdown("Upload new images for retraining. Images will be added to the training set.")
        
        # Get available classes
        try:
            response = requests.get(f"{API_URL}/classes", timeout=5)
            classes = response.json().get("classes", [])
        except:
            classes = []
        
        class_name = st.selectbox(
            "Select class",
            options=classes if classes else ["(No classes available - enter manually)"],
            help="Select the disease class for the uploaded images"
        )
        
        custom_class = st.text_input(
            "Or enter new class name",
            placeholder="e.g., Tomato___New_Disease"
        )
        
        final_class = custom_class if custom_class else class_name
        
        uploaded_files = st.file_uploader(
            "Upload images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Select multiple images for the selected class"
        )
        
        if uploaded_files and final_class:
            if st.button("Upload for Retraining", use_container_width=True):
                with st.spinner("Uploading images..."):
                    result = upload_retrain_images(uploaded_files, final_class)
                    if result.get("success"):
                        st.success(f"Successfully uploaded {len(uploaded_files)} images!")
                    else:
                        st.error(f"Upload failed: {result.get('error', 'Unknown error')}")
    
    with col2:
        st.subheader("Trigger Retraining")
        st.markdown("Configure and start model retraining with new data.")
        
        model_type = st.selectbox(
            "Model Type",
            options=["resnet50", "mobilenetv2", "efficientnet", "custom_cnn"],
            index=0,
            help="Select the model architecture"
        )
        
        epochs = st.slider(
            "Epochs",
            min_value=1,
            max_value=100,
            value=10,
            help="Number of training epochs"
        )
        
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.0001,
            help="Learning rate for training"
        )
        
        if st.button("Start Retraining", use_container_width=True, type="primary"):
            if api_status.get("is_retraining"):
                st.warning("Retraining already in progress!")
            else:
                with st.spinner("Initiating retraining..."):
                    result = trigger_retrain(epochs, learning_rate, model_type)
                    if result.get("success"):
                        st.success("Retraining started! Monitor progress on the dashboard.")
                        st.rerun()
                    else:
                        st.error(f"Failed to start retraining: {result.get('error', 'Unknown error')}")


elif page == "Dataset":
    st.markdown('<h1 class="main-header">Dataset Statistics</h1>', unsafe_allow_html=True)
    
    # Dataset overview
    st.subheader("Dataset Overview")
    
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"
    
    def count_images(directory: Path) -> int:
        if not directory.exists():
            return 0
        return sum(len(list(d.glob("*"))) for d in directory.iterdir() if d.is_dir())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_count = count_images(train_dir)
        st.metric("Training Images", train_count)
    
    with col2:
        val_count = count_images(val_dir)
        st.metric("Validation Images", val_count)
    
    with col3:
        test_count = count_images(test_dir)
        st.metric("Test Images", test_count)
    
    st.markdown("---")
    
    # Class distribution
    st.subheader("Class Distribution")
    class_dist = load_class_distribution()
    
    if class_dist:
        # Summary stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Classes", len(class_dist))
        with col2:
            st.metric("Avg Images/Class", int(sum(class_dist.values()) / len(class_dist)))
        with col3:
            st.metric("Max Images in Class", max(class_dist.values()))
        
        # Distribution chart
        df = pd.DataFrame([
            {"Class": k, "Count": v}
            for k, v in sorted(class_dist.items(), key=lambda x: x[1], reverse=True)
        ])
        
        fig = px.bar(df, x="Count", y="Class", orientation='h',
                    title="Images per Class")
        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)
        
        # Plant type distribution
        st.subheader("Distribution by Plant Type")
        plant_counts = {}
        for class_name, count in class_dist.items():
            parts = class_name.split("___")
            if len(parts) >= 1:
                plant = parts[0]
                plant_counts[plant] = plant_counts.get(plant, 0) + count
        
        if plant_counts:
            df_plants = pd.DataFrame([
                {"Plant": k, "Count": v}
                for k, v in sorted(plant_counts.items(), key=lambda x: x[1], reverse=True)
            ])
            
            fig = px.pie(df_plants, values="Count", names="Plant",
                        title="Distribution by Plant Type")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Disease distribution
        st.subheader("Distribution by Disease Type")
        disease_counts = {}
        for class_name, count in class_dist.items():
            parts = class_name.split("___")
            if len(parts) >= 2:
                disease = parts[1]
                disease_counts[disease] = disease_counts.get(disease, 0) + count
        
        if disease_counts:
            df_diseases = pd.DataFrame([
                {"Disease": k, "Count": v}
                for k, v in sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)
            ])
            
            fig = px.bar(df_diseases, x="Count", y="Disease", orientation='h',
                        title="Distribution by Disease Type")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No dataset available. Run the data pipeline to split the dataset first.")
        
        if st.button("Initialize Dataset"):
            st.info("Run the following command to initialize the dataset:")
            st.code("python src/data_pipeline.py", language="bash")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Links")
st.sidebar.markdown(f"- [API Docs]({API_URL}/docs)")
st.sidebar.markdown("- [GitHub Repository](#)")
st.sidebar.markdown("---")
st.sidebar.markdown("v1.0.0 | Plant Disease AI")
