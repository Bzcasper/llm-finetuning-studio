from http.server import BaseHTTPRequestHandler
import json
import os
import uuid
import time
from urllib.parse import urlparse, parse_qs

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/training/start':
            self.handle_training_start()
        else:
            self.send_error(404)
    
    def do_GET(self):
        if self.path.startswith('/api/training/status/'):
            job_id = self.path.split('/')[-1]
            self.handle_training_status(job_id)
        elif self.path.startswith('/api/training/logs/'):
            job_id = self.path.split('/')[-1]
            self.handle_training_logs(job_id)
        elif self.path == '/api/training/jobs':
            self.handle_list_jobs()
        else:
            self.send_error(404)
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def handle_training_start(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Validate required fields
            required_fields = ['model_name', 'dataset_path']
            for field in required_fields:
                if field not in data or not data[field]:
                    self.send_error(400, f"Missing required field: {field}")
                    return
            
            # Generate job ID
            job_id = str(uuid.uuid4())
            
            # In a real implementation, this would trigger Modal function
            # For now, we'll simulate the response
            try:
                # Try to connect to Modal (this will fail in Vercel without proper setup)
                modal_token_id = os.environ.get("MODAL_TOKEN_ID")
                modal_token_secret = os.environ.get("MODAL_TOKEN_SECRET")
                
                if not modal_token_id or not modal_token_secret:
                    raise Exception("Modal credentials not configured")
                
                # Import modal and attempt connection
                import modal
                
                # Set up Modal credentials
                os.environ["MODAL_TOKEN_ID"] = modal_token_id
                os.environ["MODAL_TOKEN_SECRET"] = modal_token_secret
                
                # Try to lookup the app
                app = modal.App.lookup("llm-finetuner", create_if_missing=False)
                
                # Call the fine-tuning function
                result = app.fine_tune_llm.remote(
                    model_name_or_path=data["model_name"],
                    dataset_path=data["dataset_path"],
                    output_dir=data.get("output_dir", "/vol/finetuned_model"),
                    num_train_epochs=data.get("num_train_epochs", 3),
                    per_device_train_batch_size=data.get("per_device_train_batch_size", 2),
                    learning_rate=data.get("learning_rate", 0.0002),
                    lora_r=data.get("lora_r", 8),
                    lora_alpha=data.get("lora_alpha", 16),
                    lora_dropout=data.get("lora_dropout", 0.05),
                    use_4bit=data.get("use_4bit", True),
                    optimizer=data.get("optimizer", "adamw_torch"),
                    gpu_type=data.get("gpu_type", "A100"),
                    timeout=data.get("timeout", 3600)
                )
                
                response = {
                    "job_id": job_id,
                    "status": "started",
                    "message": "Training job started successfully",
                    "modal_job_id": str(result),
                    "timestamp": time.time()
                }
                
            except Exception as e:
                # Fallback response when Modal is not available
                response = {
                    "job_id": job_id,
                    "status": "simulated",
                    "message": f"Training simulation started (Modal not connected: {str(e)})",
                    "timestamp": time.time(),
                    "config": data
                }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
        except Exception as e:
            self.send_error(500, str(e))
    
    def handle_training_status(self, job_id):
        # Simulate training status
        response = {
            "job_id": job_id,
            "status": "running",
            "progress": 0.5,
            "current_epoch": 2,
            "total_epochs": 3,
            "current_loss": 1.5,
            "timestamp": time.time()
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def handle_training_logs(self, job_id):
        # Simulate training logs
        logs = [
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training started for job {job_id}",
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading model and dataset...",
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting epoch 1/3",
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Current loss: 2.1"
        ]
        
        response = {
            "job_id": job_id,
            "logs": logs,
            "timestamp": time.time()
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def handle_list_jobs(self):
        # Simulate job list
        response = {
            "jobs": [
                {
                    "job_id": "example-job-1",
                    "status": "completed",
                    "model_name": "microsoft/DialoGPT-small",
                    "created_at": time.time() - 3600
                }
            ],
            "total": 1
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

