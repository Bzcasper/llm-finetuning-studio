import modal
import os
import json
import time
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import uvicorn

# Define the FastAPI app
api_app = FastAPI(title="LLM Fine-Tuning API", version="1.0.0")

# Add CORS middleware
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests
class TrainingConfig(BaseModel):
    model_name: str
    dataset_path: str
    output_dir: str = "/vol/finetuned_model"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 0.0002
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    optimizer_type: str = "adamw_torch"
    use_4bit_quantization: bool = False
    gpu_type: str = "A100"
    timeout: int = 3600

class TrainingStatus(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float
    current_epoch: int
    total_epochs: int
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    estimated_time_remaining: Optional[int] = None

# In-memory storage for training jobs (in production, use a database)
training_jobs: Dict[str, TrainingStatus] = {}
training_logs: Dict[str, List[str]] = {}

# Modal app reference for deployed functions
try:
    # Connect to the deployed Modal app
    deployed_app = modal.App.lookup("llm-finetuner", environment_name="ai-tool-pool")
    fine_tune_function = deployed_app.fine_tune_llm
    list_function = deployed_app.list_models_and_datasets
    print("Successfully connected to deployed Modal app")
except Exception as e:
    print(f"Warning: Could not connect to deployed Modal app: {e}")
    deployed_app = None
    fine_tune_function = None
    list_function = None

# Custom monitoring and logging functions
class TrainingMonitor:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.start_time = time.time()
        
    def log(self, message: str):
        if self.job_id not in training_logs:
            training_logs[self.job_id] = []
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        training_logs[self.job_id].append(f"[{timestamp}] {message}")
        
    def update_progress(self, epoch: int, total_epochs: int, loss: float = None, accuracy: float = None):
        progress = (epoch / total_epochs) * 100
        
        # Simulate GPU metrics (in real implementation, get from actual GPU monitoring)
        gpu_utilization = 85.0 + (epoch * 2)  # Simulate increasing utilization
        gpu_memory_used = 12.4 + (epoch * 0.5)  # Simulate memory usage
        
        # Estimate remaining time
        elapsed_time = time.time() - self.start_time
        if epoch > 0:
            time_per_epoch = elapsed_time / epoch
            remaining_epochs = total_epochs - epoch
            estimated_remaining = int(time_per_epoch * remaining_epochs)
        else:
            estimated_remaining = None
            
        training_jobs[self.job_id] = TrainingStatus(
            job_id=self.job_id,
            status="running",
            progress=progress,
            current_epoch=epoch,
            total_epochs=total_epochs,
            loss=loss,
            accuracy=accuracy,
            gpu_utilization=min(gpu_utilization, 100.0),
            gpu_memory_used=min(gpu_memory_used, 20.0),
            estimated_time_remaining=estimated_remaining
        )

# API endpoints
@api_app.post("/api/training/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start a new fine-tuning job"""
    job_id = f"job_{int(time.time())}"
    
    # Initialize job status
    training_jobs[job_id] = TrainingStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
        current_epoch=0,
        total_epochs=config.num_train_epochs
    )
    
    # Start training in background
    config_dict = config.dict()
    
    if fine_tune_function:
        # Use actual Modal function
        background_tasks.add_task(run_modal_training_real, config_dict, job_id)
    else:
        # Use simulation for demo
        background_tasks.add_task(run_modal_training_simulation, config_dict, job_id)
    
    return {"job_id": job_id, "status": "started"}

async def run_modal_training_real(config: dict, job_id: str):
    """Run the actual Modal training function"""
    try:
        monitor = TrainingMonitor(job_id)
        monitor.log("Starting Modal.com fine-tuning job...")
        
        # Call the deployed Modal function
        result = await fine_tune_function.remote.aio(
            model_name_or_path=config["model_name"],
            dataset_path=config["dataset_path"],
            output_dir=config["output_dir"],
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            learning_rate=config["learning_rate"],
            num_train_epochs=config["num_train_epochs"],
            per_device_train_batch_size=config["per_device_train_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            optimizer_type=config["optimizer_type"],
            use_4bit_quantization=config["use_4bit_quantization"],
            gpu_type=config["gpu_type"]
        )
        
        # Mark as completed
        training_jobs[job_id].status = "completed"
        training_jobs[job_id].progress = 100.0
        monitor.log("Modal.com fine-tuning completed successfully!")
        
    except Exception as e:
        if job_id in training_jobs:
            training_jobs[job_id].status = "failed"
        monitor.log(f"Modal.com training failed: {str(e)}")
        print(f"Training failed for job {job_id}: {e}")

async def run_modal_training_simulation(config: dict, job_id: str):
    """Simulate training progress for demo purposes"""
    monitor = TrainingMonitor(job_id)
    total_epochs = config["num_train_epochs"]
    
    monitor.log("Starting simulated training (Modal.com not connected)...")
    
    for epoch in range(1, total_epochs + 1):
        # Simulate epoch duration
        await asyncio.sleep(3)
        
        # Simulate loss and accuracy
        loss = 2.5 - (epoch * 0.3)
        accuracy = 0.65 + (epoch * 0.05)
        
        monitor.update_progress(epoch, total_epochs, loss, accuracy)
        monitor.log(f"Completed epoch {epoch}/{total_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.3f}")
    
    # Mark as completed
    training_jobs[job_id].status = "completed"
    training_jobs[job_id].progress = 100.0
    monitor.log("Simulated training completed successfully!")

@api_app.get("/api/training/status/{job_id}")
async def get_training_status(job_id: str):
    """Get the status of a training job"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return training_jobs[job_id]

@api_app.get("/api/training/logs/{job_id}")
async def get_training_logs(job_id: str):
    """Get the logs for a training job"""
    if job_id not in training_logs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"logs": training_logs[job_id]}

@api_app.get("/api/training/jobs")
async def list_training_jobs():
    """List all training jobs"""
    return {"jobs": list(training_jobs.values())}

@api_app.get("/api/datasets")
async def list_datasets():
    """List available datasets"""
    try:
        if list_function:
            # Use actual Modal function
            result = await list_function.remote.aio()
            return {"datasets": result.get("datasets", [])}
        else:
            # Return mock data
            return {
                "datasets": [
                    {"name": "dataset1.jsonl", "path": "/vol/dataset1.jsonl", "size": "1.2MB"},
                    {"name": "dataset2.csv", "path": "/vol/dataset2.csv", "size": "850KB"},
                    {"name": "custom_data.txt", "path": "/vol/custom_data.txt", "size": "2.1MB"}
                ]
            }
    except Exception as e:
        print(f"Error listing datasets: {e}")
        return {"datasets": []}

@api_app.get("/api/models")
async def list_models():
    """List available models"""
    try:
        if list_function:
            # Use actual Modal function
            result = await list_function.remote.aio()
            return {"models": result.get("models", [])}
        else:
            # Return mock data
            return {
                "models": [
                    {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "type": "huggingface"},
                    {"name": "microsoft/DialoGPT-medium", "type": "huggingface"},
                    {"name": "meta-llama/Llama-2-7b-chat-hf", "type": "huggingface"},
                    {"name": "finetuned_model_v1", "type": "volume", "path": "/vol/finetuned_model_v1"}
                ]
            }
    except Exception as e:
        print(f"Error listing models: {e}")
        return {"models": []}

@api_app.get("/api/modal/status")
async def get_modal_status():
    """Get Modal.com connection status"""
    return {
        "connected": deployed_app is not None,
        "app_name": "llm-finetuner" if deployed_app else None,
        "namespace": "ai-tool-pool" if deployed_app else None,
        "functions_available": {
            "fine_tune_llm": fine_tune_function is not None,
            "list_models_and_datasets": list_function is not None
        }
    }

@api_app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "modal_connected": deployed_app is not None
    }

# Test endpoint for Modal connection
@api_app.post("/api/test/modal")
async def test_modal_connection():
    """Test the Modal.com connection"""
    try:
        if not list_function:
            return {"success": False, "error": "Modal function not available"}
            
        # Test the list function
        result = await list_function.remote.aio()
        return {"success": True, "result": result}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(api_app, host="0.0.0.0", port=8000)

