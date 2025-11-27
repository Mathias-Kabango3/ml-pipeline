"""
Locust Load Testing for Plant Disease Classification API
=========================================================
This module provides load testing capabilities using Locust
to simulate high-load scenarios and measure performance.
"""

import os
import io
import random
from pathlib import Path
from locust import HttpUser, task, between, events
from PIL import Image
import numpy as np


# Configuration
API_HOST = os.environ.get("API_HOST", "http://localhost:8000")
BASE_DIR = Path(__file__).parent.parent
TEST_IMAGES_DIR = BASE_DIR / "data" / "test"


def generate_random_image(size=(224, 224)):
    """Generate a random test image."""
    # Create a random RGB image
    img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.read()


def get_test_image():
    """Get a random test image from the test set."""
    if TEST_IMAGES_DIR.exists():
        # Get all image files
        image_files = []
        for class_dir in TEST_IMAGES_DIR.iterdir():
            if class_dir.is_dir():
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG"]:
                    image_files.extend(list(class_dir.glob(ext)))
        
        if image_files:
            # Pick a random image
            img_path = random.choice(image_files)
            with open(img_path, 'rb') as f:
                return f.read()
    
    # Fall back to random image
    return generate_random_image()


class PlantDiseaseAPIUser(HttpUser):
    """Simulates a user interacting with the Plant Disease API."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    host = API_HOST
    
    def on_start(self):
        """Called when a simulated user starts."""
        # Preload some test images for faster testing
        self.test_images = [get_test_image() for _ in range(10)]
        
    @task(10)
    def predict_image(self):
        """
        Test the /predict endpoint.
        Weight: 10 (most common operation)
        """
        image_data = random.choice(self.test_images)
        
        files = {
            'file': ('test_image.jpg', image_data, 'image/jpeg')
        }
        
        with self.client.post(
            "/predict",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    response.success()
                else:
                    response.failure(f"Prediction failed: {result.get('error')}")
            elif response.status_code == 503:
                response.failure("Model not loaded")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(3)
    def check_status(self):
        """
        Test the /status endpoint.
        Weight: 3 (common operation)
        """
        with self.client.get("/status", catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get("status") in ["healthy", "degraded"]:
                    response.success()
                else:
                    response.failure(f"Unexpected status: {result.get('status')}")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def health_check(self):
        """
        Test the /health endpoint.
        Weight: 2
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_classes(self):
        """
        Test the /classes endpoint.
        Weight: 1 (less common)
        """
        with self.client.get("/classes", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_metrics(self):
        """
        Test the /metrics endpoint.
        Weight: 1 (less common)
        """
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_dataset_stats(self):
        """
        Test the /dataset/stats endpoint.
        Weight: 1 (less common)
        """
        with self.client.get("/dataset/stats", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class BatchPredictionUser(HttpUser):
    """Simulates a user making batch predictions."""
    
    wait_time = between(5, 10)  # Longer wait for batch operations
    host = API_HOST
    weight = 1  # Lower weight - less common
    
    def on_start(self):
        """Called when a simulated user starts."""
        self.test_images = [get_test_image() for _ in range(20)]
    
    @task
    def batch_predict(self):
        """Test batch prediction endpoint."""
        # Select 3-5 random images
        num_images = random.randint(3, 5)
        images = random.sample(self.test_images, num_images)
        
        files = [
            ('files', (f'image_{i}.jpg', img, 'image/jpeg'))
            for i, img in enumerate(images)
        ]
        
        with self.client.post(
            "/predict/batch",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                result = response.json()
                results = result.get("results", [])
                successful = sum(1 for r in results if r.get("success"))
                if successful == len(images):
                    response.success()
                else:
                    response.failure(f"Only {successful}/{len(images)} predictions succeeded")
            else:
                response.failure(f"Status code: {response.status_code}")


class StressTestUser(HttpUser):
    """Aggressive user for stress testing - minimal wait time."""
    
    wait_time = between(0.1, 0.5)  # Very short wait
    host = API_HOST
    weight = 1  # Only use for stress testing
    
    def on_start(self):
        self.test_images = [get_test_image() for _ in range(5)]
    
    @task
    def rapid_predict(self):
        """Rapid-fire predictions."""
        image_data = random.choice(self.test_images)
        
        files = {'file': ('test.jpg', image_data, 'image/jpeg')}
        
        self.client.post("/predict", files=files)


# Event handlers for custom metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Log request metrics."""
    if exception:
        print(f"Request failed: {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print("=" * 50)
    print("Starting Load Test for Plant Disease API")
    print("=" * 50)
    print(f"Target host: {API_HOST}")
    print("=" * 50)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    print("\n" + "=" * 50)
    print("Load Test Completed")
    print("=" * 50)
    
    # Print summary
    stats = environment.stats
    print(f"\nTotal requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f} ms")
    print(f"Min response time: {stats.total.min_response_time:.2f} ms")
    print(f"Max response time: {stats.total.max_response_time:.2f} ms")
    print(f"Requests per second: {stats.total.current_rps:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    print("""
    Locust Load Testing for Plant Disease API
    ==========================================
    
    Usage:
    ------
    
    1. Start the API server:
       cd src && uvicorn api:app --host 0.0.0.0 --port 8000
    
    2. Run Locust with web UI:
       locust -f locustfile.py --host http://localhost:8000
       Then open http://localhost:8089 in your browser
    
    3. Run Locust headless (command line):
       locust -f locustfile.py --host http://localhost:8000 \\
              --users 100 --spawn-rate 10 --run-time 1m \\
              --headless --csv=results
    
    4. Run specific user class:
       locust -f locustfile.py --host http://localhost:8000 \\
              --class-picker  # For interactive selection
    
    Environment Variables:
    ----------------------
    API_HOST: API endpoint (default: http://localhost:8000)
    
    Results:
    --------
    - results_stats.csv: Overall statistics
    - results_stats_history.csv: Time-series data
    - results_failures.csv: Failed requests
    """)
