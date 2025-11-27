# Plant Disease Classification Pipeline

A complete MLOps pipeline for plant disease classification using deep learning with TensorFlow/Keras. This project includes data preprocessing, model training, evaluation, API deployment, and a web-based dashboard for monitoring and retraining.

## Demo Video

[![Plant Disease Classification Demo](https://img.youtube.com/vi/xxB2_VbbX1M/0.jpg)](https://youtu.be/xxB2_VbbX1M)

**Watch the full demo:** [https://youtu.be/xxB2_VbbX1M](https://youtu.be/xxB2_VbbX1M)

## Live Deployment

- **API (Railway):** https://ml-pipeline-production-be57.up.railway.app
- **UI Dashboard (Streamlit Cloud):** https://ml-pipeline-mathias.streamlit.app
- **Model (Hugging Face):** https://huggingface.co/mathiaskabango/plantvillagev2

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Rubric Compliance](#rubric-compliance)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Load Testing](#load-testing)

## Features

- **Deep Learning Model**: Transfer learning with ResNet50 (pre-trained on ImageNet)
- **Data Pipeline**: Automated data loading, augmentation, and train/val/test splitting
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC
- **REST API**: FastAPI endpoints for prediction, retraining, and status monitoring
- **Web Dashboard**: Streamlit UI for visualization and model management
- **Automated Retraining**: Upload new images and fine-tune the model
- **MLOps Integration**: MLflow and TensorBoard for experiment tracking
- **Containerization**: Docker and Docker Compose for easy deployment
- **Load Testing**: Locust for performance testing

## Rubric Compliance

### 1. Retraining Process (10 pts)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Data File Uploading | Done | `POST /upload/retrain_data` - Upload images with class labels |
| Saving to Database | Done | Images saved to `data/retrain_data/{class_name}/` |
| Data Preprocessing | Done | ResNet50 preprocessing, image resizing (190x190), augmentation |
| Retraining with Pre-trained Model | Done | Fine-tunes existing model (freezes early layers, trains last 10) |

**Files:** `src/api.py` (endpoints), `src/retrain.py` (training logic), `src/ui_app.py` (UI)

### 2. Prediction Process (10 pts)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Insert Data Point for Prediction | Done | Upload image via UI or `POST /predict` API |
| Display Correct Prediction | Done | Returns class name with confidence score |

**Files:** `src/api.py` (`/predict` endpoint), `src/ui_app.py` (Prediction page)

### 3. Evaluation of Models (10 pts)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Clear Preprocessing Steps | Done | Data augmentation, normalization, train/val/test split |
| Optimization Techniques | Done | Early stopping, learning rate scheduling, pre-trained ResNet50 |
| Evaluation Metrics (4+) | Done | Accuracy, Loss, Precision, Recall, F1-Score, Confusion Matrix |

**Files:** `notebook/plant_disease_classification.ipynb`, `src/evaluate_model.py`

### 4. Deployment Package (10 pts)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| UI (Web App) | Done | Streamlit dashboard with full functionality |
| Public URL | Done | Railway API + Streamlit Cloud UI |
| Data Insights/Visualizations | Done | Class distribution, training metrics, confusion matrix |

**Live URLs:**
- API: https://ml-pipeline-production-be57.up.railway.app
- UI: https://ml-pipeline-mathias.streamlit.app

## Project Structure

```
ml-pipeline/
├── README.md
├── requirements.txt
├── Dockerfile.railway
├── docker-compose.yml
├── notebook/
│   └── plant_disease_classification.ipynb  # Model training notebook
├── src/
│   ├── data_pipeline.py       # Data loading and preprocessing
│   ├── model_pipeline.py      # Model creation and training
│   ├── evaluate_model.py      # Model evaluation and metrics
│   ├── api.py                 # FastAPI application
│   ├── ui_app.py              # Streamlit dashboard
│   ├── retrain.py             # Automated retraining script
│   └── locustfile.py          # Load testing
├── data/
│   ├── train/                 # Training images
│   ├── val/                   # Validation images
│   ├── test/                  # Test images
│   ├── retrain_data/          # New data for retraining
│   └── index_to_class.json    # Class label mapping
├── models/
│   └── plant_disease_model_v2.keras  # Trained model
├── logs/                      # TensorBoard logs
└── mlruns/                    # MLflow tracking
```

## Installation

### Prerequisites

- Python 3.10+
- pip or conda
- Docker (optional, for containerized deployment)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/Mathias-Kabango3/ml-pipeline.git
cd ml-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Installation

```bash
# Build Docker images
docker-compose build

# Start services
docker-compose up -d
```

## Quick Start

### 1. Prepare the Dataset

```bash
# Split the PlantVillage dataset into train/val/test
python src/data_pipeline.py
```

### 2. Train the Model

```bash
# Train with default settings (ResNet50)
python src/model_pipeline.py

# Or train with specific settings
python src/model_pipeline.py --model resnet50 --epochs 50 --batch-size 32
```

### 3. Evaluate the Model

```bash
python src/evaluate_model.py
```

### 4. Start the API

```bash
# Start FastAPI server
cd src
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Start the Dashboard

```bash
# Start Streamlit dashboard
streamlit run src/ui_app.py
```

## 📖 Usage

### Data Pipeline

```python
from src.data_pipeline import (
    split_dataset,
    create_data_generators,
    get_dataset_statistics,
    visualize_dataset_distribution
)

# Split dataset
split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

# Get statistics
stats = get_dataset_statistics()
print(f"Total images: {stats['total_images']}")
print(f"Number of classes: {stats['num_classes']}")

# Create data generators
train_gen, val_gen, test_gen = create_data_generators(batch_size=32)
```

### Model Training

```python
from src.model_pipeline import train_model, load_model

# Train model
model, history = train_model(
    model_type="resnet50",
    epochs=30,
    batch_size=32,
    learning_rate=0.001,
    use_class_weights=True,
    fine_tune=True
)

# Load saved model
model = load_model()
```

### Model Evaluation

```python
from src.evaluate_model import (
    evaluate_model,
    generate_full_evaluation_report
)

# Quick evaluation
metrics = evaluate_model()
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Full report with visualizations
generate_full_evaluation_report()
```

### API Prediction

```python
import requests

# Single image prediction
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

### Automated Retraining

```bash
# Manual retrain
python src/retrain.py --mode retrain --epochs 10 --lr 0.0001

# Watch folder for new data (auto-retrain)
python src/retrain.py --mode watch --threshold 50

# Check for new data
python src/retrain.py --mode check
```

## 📚 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | API health status |
| `/status` | GET | Detailed status including model uptime |
| `/predict` | POST | Predict disease from image |
| `/predict/batch` | POST | Batch prediction |
| `/retrain` | POST | Trigger model retraining |
| `/upload/retrain_data` | POST | Upload images for retraining |
| `/classes` | GET | Get all disease classes |
| `/dataset/stats` | GET | Dataset statistics |
| `/model/info` | GET | Model metadata |
| `/metrics` | GET | Model performance metrics |

### Interactive API Docs

After starting the API, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🐳 Deployment

### Docker Compose

```bash
# Start API and UI
docker-compose up -d api ui

# Start with monitoring (TensorBoard, MLflow)
docker-compose --profile monitoring up -d

# Start training service
docker-compose --profile training up training

# Run load tests
docker-compose --profile testing up locust
```

### Cloud Deployment

#### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t plant-disease-api --target api .
docker tag plant-disease-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/plant-disease-api:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/plant-disease-api:latest
```

#### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/<project>/plant-disease-api
gcloud run deploy plant-disease-api --image gcr.io/<project>/plant-disease-api --platform managed
```

#### Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: plant-disease-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: plant-disease-api
  template:
    metadata:
      labels:
        app: plant-disease-api
    spec:
      containers:
      - name: api
        image: plant-disease-api:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
```

## Load Testing

### Using Locust

```bash
# Start Locust with web UI
locust -f src/locustfile.py --host https://ml-pipeline-production-be57.up.railway.app

# Headless mode
locust -f src/locustfile.py --host https://ml-pipeline-production-be57.up.railway.app \
       --users 100 --spawn-rate 10 --run-time 5m \
       --headless --csv=results
```

### Load Test Results (Flood Request Simulation)

We performed load testing on the deployed Railway API to simulate high traffic scenarios.

**Test Configuration:**
- Target: Railway API (https://ml-pipeline-production-be57.up.railway.app)
- Users: 50 concurrent users
- Spawn Rate: 10 users/second
- Duration: 2 minutes

**Results Summary:**

| Metric | Value |
|--------|-------|
| Total Requests | 732 |
| Requests/sec (RPS) | 6.1 |
| Failure Rate | 0% |
| Median Response Time | 4,800 ms |
| 95th Percentile | 11,000 ms |
| Max Response Time | 19,000 ms |

**Endpoint Breakdown:**

| Endpoint | Requests | Failures | Median (ms) | 95% (ms) |
|----------|----------|----------|-------------|----------|
| GET /health | 366 | 0 | 240 | 510 |
| POST /predict | 366 | 0 | 9,100 | 17,000 |

**Analysis:**
- The API successfully handled all requests with **0% failure rate**
- Health endpoint responds quickly (~240ms median)
- Prediction endpoint is slower due to model inference on CPU (Railway free tier)
- No errors or crashes under sustained load
- System remained stable throughout the test

### Expected Performance

| Metric | Railway (CPU) | Local GPU |
|--------|---------------|-----------|
| Throughput | 6-10 req/s | 50-100 req/s |
| Latency (p50) | ~5,000ms | <200ms |
| Latency (p95) | ~11,000ms | <500ms |
| Memory | 512MB-1GB | 2-4 GB |

## 🛠️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | `http://localhost:8000` | API endpoint for UI |
| `API_HOST` | `http://localhost:8000` | API host for load tests |
| `TF_CPP_MIN_LOG_LEVEL` | `2` | TensorFlow log level |
| `PYTHONPATH` | `/app/src` | Python module path |

### Model Configuration

Edit `src/model_pipeline.py`:

```python
# Training configuration
EPOCHS = 50
LEARNING_RATE = 0.001
FINE_TUNE_EPOCHS = 20
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
```

## 📈 Experiment Tracking

### TensorBoard

```bash
tensorboard --logdir logs
```

### MLflow

```bash
mlflow ui --backend-store-uri mlruns
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) for the plant disease images
- TensorFlow team for the deep learning framework
- FastAPI and Streamlit communities for excellent web frameworks
