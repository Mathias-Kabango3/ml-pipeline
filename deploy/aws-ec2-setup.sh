#!/bin/bash
# AWS EC2 Deployment Script for Plant Disease Classification API
# Run this script on a fresh EC2 instance (Ubuntu 22.04 recommended)

set -e

echo "=== Updating system ==="
sudo apt update && sudo apt upgrade -y

echo "=== Installing Docker ==="
sudo apt install -y docker.io docker-compose git
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

echo "=== Cloning repository ==="
git clone https://github.com/Mathias-Kabango3/ml-pipeline.git
cd ml-pipeline

echo "=== Building Docker image ==="
sudo docker build -t plant-disease-api .

echo "=== Running container ==="
sudo docker run -d \
  --name plant-disease-api \
  --restart unless-stopped \
  -p 80:8000 \
  -e MODEL_DOWNLOAD_URL="https://huggingface.co/mathiaskabango/plantvillage/resolve/main/plant_disease_resnet50_checkpoint_01.keras" \
  plant-disease-api

echo "=== Setup complete! ==="
echo "API is now running on http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo ""
echo "Test with: curl http://YOUR_PUBLIC_IP/status"
