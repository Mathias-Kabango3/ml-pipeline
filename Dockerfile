# =========================================================
# Dockerfile for Plant Disease Classification Application
# =========================================================
# Multi-stage build for optimized image size

# Stage 1: Base image with Python and dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Stage 2: API Service
FROM base as api

# Copy source code
COPY src/ /app/src/
COPY models/ /app/models/
COPY data/ /app/data/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]


# Stage 3: Streamlit UI
FROM base as ui

# Copy source code
COPY src/ /app/src/
COPY models/ /app/models/
COPY data/ /app/data/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "src/ui_app.py", "--server.port=8501", "--server.address=0.0.0.0"]


# Stage 4: Training service
FROM base as training

# Install additional training dependencies
RUN pip install --no-cache-dir tensorboard

# Copy source code
COPY src/ /app/src/
COPY data/ /app/data/
COPY models/ /app/models/
COPY plantvillagedataset/ /app/plantvillagedataset/

# Default command - can be overridden
CMD ["python", "src/model_pipeline.py"]


# Stage 5: All-in-one development image
FROM base as development

# Install all dependencies including dev tools
RUN pip install --no-cache-dir \
    tensorboard \
    jupyter \
    ipykernel \
    pytest \
    pytest-cov \
    black \
    isort \
    flake8

# Copy all source code
COPY . /app/

# Expose all ports
EXPOSE 8000 8501 8089 6006

# Default to bash for development
CMD ["/bin/bash"]
