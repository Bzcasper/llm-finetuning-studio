#!/usr/bin/env python3
"""
API Integration Test Script
Tests the complete API workflow with real endpoints
"""

import requests
import time
import json
import sys
import os
from typing import Dict, Any

class APIIntegrationTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def test_health_check(self) -> bool:
        """Test the health check endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_modal_status(self) -> bool:
        """Test Modal connection status"""
        try:
            response = self.session.get(f"{self.base_url}/api/modal/status")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Modal status: {data}")
                return True
            else:
                print(f"❌ Modal status failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Modal status error: {e}")
            return False
    
    def test_datasets_endpoint(self) -> bool:
        """Test datasets listing"""
        try:
            response = self.session.get(f"{self.base_url}/api/datasets")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Datasets endpoint: Found {len(data.get('datasets', []))} datasets")
                return True
            else:
                print(f"❌ Datasets endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Datasets endpoint error: {e}")
            return False
    
    def test_models_endpoint(self) -> bool:
        """Test models listing"""
        try:
            response = self.session.get(f"{self.base_url}/api/models")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Models endpoint: Found {len(data.get('models', []))} models")
                return True
            else:
                print(f"❌ Models endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Models endpoint error: {e}")
            return False
    
    def test_file_upload(self) -> bool:
        """Test file upload functionality"""
        try:
            # Create a test dataset
            test_data = [
                {"text": "This is a test sentence for fine-tuning."},
                {"text": "Another example sentence for training."},
                {"text": "Machine learning is fascinating."}
            ]
            
            # Convert to JSONL format
            jsonl_content = "\n".join(json.dumps(item) for item in test_data)
            
            files = {
                'file': ('test_dataset.jsonl', jsonl_content, 'application/json')
            }
            
            response = self.session.post(f"{self.base_url}/api/upload", files=files)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ File upload successful: {data}")
                return True
            else:
                print(f"❌ File upload failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ File upload error: {e}")
            return False
    
    def test_training_workflow(self) -> bool:
        """Test the complete training workflow"""
        try:
            # Test training configuration
            training_config = {
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
            
            response = self.session.post(f"{self.base_url}/api/training/start", json=training_config)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Training start successful: {data}")
                
                # If we got a job ID, test status endpoint
                if "job_id" in data:
                    job_id = data["job_id"]
                    status_response = self.session.get(f"{self.base_url}/api/training/status/{job_id}")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        print(f"✅ Training status check: {status_data}")
                    else:
                        print(f"⚠️ Training status check failed: {status_response.status_code}")
                
                return True
            else:
                print(f"❌ Training start failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Training workflow error: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all integration tests"""
        print("🚀 Starting API Integration Tests...")
        print(f"Testing against: {self.base_url}")
        print("-" * 50)
        
        tests = {
            "Health Check": self.test_health_check,
            "Modal Status": self.test_modal_status,
            "Datasets Endpoint": self.test_datasets_endpoint,
            "Models Endpoint": self.test_models_endpoint,
            "File Upload": self.test_file_upload,
            "Training Workflow": self.test_training_workflow
        }
        
        results = {}
        for test_name, test_func in tests.items():
            print(f"\n🧪 Running: {test_name}")
            results[test_name] = test_func()
            time.sleep(1)  # Brief pause between tests
        
        print("\n" + "=" * 50)
        print("📊 Test Results Summary:")
        print("=" * 50)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed!")
            return True
        else:
            print("⚠️ Some tests failed. Check the logs above.")
            return False

def main():
    """Main function to run integration tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Integration Tester")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL for the API (default: http://localhost:8000)")
    parser.add_argument("--wait", type=int, default=0,
                       help="Wait time in seconds before starting tests")
    
    args = parser.parse_args()
    
    if args.wait > 0:
        print(f"⏳ Waiting {args.wait} seconds for server to start...")
        time.sleep(args.wait)
    
    tester = APIIntegrationTester(args.url)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

