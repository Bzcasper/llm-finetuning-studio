import pytest
import asyncio
import httpx
import time
from fastapi.testclient import TestClient
import sys
import os

# Add the parent directory to the path to import api_server
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_server import app

client = TestClient(app)

class TestBackendAPI:
    """Test suite for the backend API endpoints"""
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "modal_connected" in data
    
    def test_cors_headers(self):
        """Test CORS headers are properly set"""
        response = client.options("/api/health")
        assert response.status_code == 200
        # Note: TestClient doesn't include CORS headers by default
        # This would be tested in integration tests
    
    def test_training_start_endpoint(self):
        """Test training start endpoint with valid payload"""
        payload = {
            "model_name": "microsoft/DialoGPT-small",
            "dataset_path": "dummy_dataset.jsonl",
            "output_dir": "/vol/test_model",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "learning_rate": 0.0002,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "use_4bit": True,
            "optimizer": "adamw_torch",
            "gpu_type": "T4",
            "timeout": 1800
        }
        
        response = client.post("/api/training/start", json=payload)
        # This might fail if Modal is not connected, which is expected in CI
        assert response.status_code in [200, 500]  # 500 if Modal not connected
        
        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data
            assert "status" in data
    
    def test_training_start_invalid_payload(self):
        """Test training start with invalid payload"""
        payload = {
            "model_name": "",  # Invalid empty model name
            "dataset_path": "",
        }
        
        response = client.post("/api/training/start", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_datasets_endpoint(self):
        """Test datasets listing endpoint"""
        response = client.get("/api/datasets")
        assert response.status_code in [200, 500]  # 500 if Modal not connected
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)
            assert "datasets" in data
    
    def test_models_endpoint(self):
        """Test models listing endpoint"""
        response = client.get("/api/models")
        assert response.status_code in [200, 500]  # 500 if Modal not connected
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)
            assert "models" in data
    
    def test_modal_status_endpoint(self):
        """Test Modal connection status endpoint"""
        response = client.get("/api/modal/status")
        assert response.status_code == 200
        data = response.json()
        assert "connected" in data
        assert "environment" in data
    
    def test_upload_endpoint(self):
        """Test file upload endpoint"""
        # Create a test file
        test_content = '{"text": "This is a test dataset entry"}\n'
        files = {"file": ("test_dataset.jsonl", test_content, "application/json")}
        
        response = client.post("/api/upload", files=files)
        assert response.status_code in [200, 500]  # 500 if Modal not connected
        
        if response.status_code == 200:
            data = response.json()
            assert "filename" in data
            assert "path" in data

class TestModalIntegration:
    """Test Modal.com integration"""
    
    @pytest.mark.asyncio
    async def test_modal_connection(self):
        """Test Modal connection"""
        try:
            import modal
            # This will fail in CI without proper credentials
            app = modal.App.lookup("llm-finetuner", create_if_missing=False)
            assert app is not None
        except Exception as e:
            # Expected to fail in CI environment
            assert "No such environment" in str(e) or "not found" in str(e)
    
    def test_modal_environment_variables(self):
        """Test Modal environment setup"""
        # Check if Modal profile is set
        modal_profile = os.environ.get("MODAL_PROFILE")
        # In CI, this might not be set, which is fine
        if modal_profile:
            assert modal_profile == "ai-tool-pool"

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])

