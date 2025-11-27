# AWS Deployment Guide for Plant Disease Classification API

## Option 1: AWS EC2 (Recommended for ML)

### Step 1: Launch EC2 Instance

1. Go to **AWS Console → EC2 → Launch Instance**
2. Choose settings:
   - **Name**: `plant-disease-api`
   - **AMI**: Ubuntu Server 22.04 LTS
   - **Instance Type**: `t3.medium` (2 vCPU, 4GB RAM) or `t3.large` for better performance
   - **Key pair**: Create or select existing
   - **Network settings**: 
     - Allow SSH (port 22)
     - Allow HTTP (port 80)
     - Allow HTTPS (port 443)
   - **Storage**: 30 GB gp3

3. Click **Launch Instance**

### Step 2: Connect to Instance

```bash
# Make key file secure
chmod 400 your-key.pem

# Connect via SSH
ssh -i your-key.pem ubuntu@YOUR_PUBLIC_IP
```

### Step 3: Deploy the API

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install -y docker.io docker-compose git
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Re-login to apply docker group
exit
# SSH back in

# Clone repository
git clone https://github.com/Mathias-Kabango3/ml-pipeline.git
cd ml-pipeline

# Build and run
sudo docker build -t plant-disease-api .
sudo docker run -d \
  --name plant-disease-api \
  --restart unless-stopped \
  -p 80:8000 \
  plant-disease-api

# Check status
curl http://localhost/status
```

### Step 4: Access Your API

Your API is now available at:
```
http://YOUR_EC2_PUBLIC_IP/status
http://YOUR_EC2_PUBLIC_IP/docs
```

---

## Option 2: AWS App Runner (Easiest)

### Step 1: Push Image to ECR

```bash
# Install AWS CLI and configure
aws configure

# Create ECR repository
aws ecr create-repository --repository-name plant-disease-api --region us-east-1

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t plant-disease-api .
docker tag plant-disease-api:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/plant-disease-api:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/plant-disease-api:latest
```

### Step 2: Create App Runner Service

1. Go to **AWS Console → App Runner → Create Service**
2. Select **Container registry → Amazon ECR**
3. Choose your image
4. Configure:
   - **Service name**: `plant-disease-api`
   - **CPU**: 1 vCPU
   - **Memory**: 2 GB
   - **Port**: 8000
5. Click **Create & Deploy**

---

## Option 3: AWS ECS with Fargate

### Step 1: Create Task Definition

Create `ecs-task-definition.json`:
```json
{
  "family": "plant-disease-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "plant-disease-api",
      "image": "YOUR_ECR_IMAGE_URI",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/plant-disease-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/ecsTaskExecutionRole"
}
```

### Step 2: Create ECS Cluster and Service

```bash
# Create cluster
aws ecs create-cluster --cluster-name ml-pipeline-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create service
aws ecs create-service \
  --cluster ml-pipeline-cluster \
  --service-name plant-disease-service \
  --task-definition plant-disease-api \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

---

## Cost Comparison

| Service | Monthly Cost (Estimate) | Pros | Cons |
|---------|------------------------|------|------|
| **EC2 t3.medium** | ~$30/month | Full control, persistent | Manual scaling |
| **App Runner** | ~$25-50/month | Easy, auto-scaling | Less control |
| **ECS Fargate** | ~$40-80/month | Serverless containers | More complex |
| **Lambda** | Pay per request | Cheapest for low traffic | Cold starts, size limits |

---

## Adding HTTPS with SSL

### Option A: Use AWS Certificate Manager + Load Balancer

```bash
# Create Application Load Balancer with SSL termination
# Point ALB to your EC2 instance
```

### Option B: Use Nginx + Let's Encrypt on EC2

```bash
# Install Nginx and Certbot
sudo apt install nginx certbot python3-certbot-nginx -y

# Get SSL certificate (requires domain)
sudo certbot --nginx -d your-domain.com
```

---

## Monitoring

### CloudWatch Logs

Add to your docker run command:
```bash
--log-driver=awslogs \
--log-opt awslogs-region=us-east-1 \
--log-opt awslogs-group=/ec2/plant-disease-api
```

### Health Check

Set up CloudWatch alarm on the `/status` endpoint.

---

## Quick Start Summary

**Fastest deployment (5 minutes):**

1. Launch Ubuntu EC2 (t3.medium)
2. Open ports 22, 80 in Security Group
3. SSH in and run:

```bash
sudo apt update && sudo apt install -y docker.io git
sudo systemctl start docker
git clone https://github.com/Mathias-Kabango3/ml-pipeline.git
cd ml-pipeline
sudo docker build -t plant-disease-api .
sudo docker run -d --restart unless-stopped -p 80:8000 plant-disease-api
```

4. Access at `http://YOUR_EC2_IP/docs`
