import modal
import os
from typing import Optional
from minio import Minio
from minio.error import S3Error
import boto3
from botocore.exceptions import ClientError
import tempfile
import shutil

# Define a Modal App
app = modal.App(name="llm-finetuner")

# Define a shared volume for models and datasets
volume = modal.Volume.from_name("llm-finetuning-volume", create_if_missing=True)

# Image for fine-tuning, including all necessary libraries
finetune_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.35.0", 
        "datasets>=2.14.0",
        "peft>=0.6.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "trl>=0.7.0",
        "loralib>=0.1.2",
        "scipy>=1.11.0",
        "einops>=0.7.0",
        "wandb>=0.15.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tensorboard>=2.14.0",
        "evaluate>=0.4.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "safetensors>=0.3.0",
        "tokenizers>=0.14.0",
        "psutil>=5.9.0",
        "GPUtil>=1.4.0",
        "nvidia-ml-py3>=7.352.0",
        "minio>=7.2.0",
        "boto3>=1.34.0",
        "s3fs>=2023.12.0"
    ])
    .apt_install(["git", "wget", "curl", "htop"])
    .run_commands([
        "pip install flash-attn --no-build-isolation || echo 'Flash attention install failed, continuing...'",
    ])
)

class StorageManager:
    """Unified storage manager for Modal volumes and MinIO"""
    
    def __init__(self):
        self.minio_client = None
        self.s3_client = None
        self.setup_storage_clients()
    
    def setup_storage_clients(self):
        """Initialize MinIO and S3 clients"""
        try:
            # MinIO configuration
            minio_endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
            minio_access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
            minio_secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
            minio_secure = os.environ.get("MINIO_SECURE", "false").lower() == "true"
            
            self.minio_client = Minio(
                minio_endpoint,
                access_key=minio_access_key,
                secret_key=minio_secret_key,
                secure=minio_secure
            )
            
            # S3 configuration (for AWS S3 compatibility)
            aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
            aws_region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            
            if aws_access_key and aws_secret_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region
                )
            
            print("✅ Storage clients initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize storage clients: {e}")
    
    def ensure_bucket_exists(self, bucket_name: str):
        """Ensure MinIO bucket exists"""
        try:
            if self.minio_client and not self.minio_client.bucket_exists(bucket_name):
                self.minio_client.make_bucket(bucket_name)
                print(f"✅ Created MinIO bucket: {bucket_name}")
        except S3Error as e:
            print(f"⚠️ MinIO bucket error: {e}")
    
    def upload_to_minio(self, local_path: str, bucket_name: str, object_name: str):
        """Upload file to MinIO"""
        try:
            if not self.minio_client:
                return False
                
            self.ensure_bucket_exists(bucket_name)
            self.minio_client.fput_object(bucket_name, object_name, local_path)
            print(f"✅ Uploaded {local_path} to MinIO: {bucket_name}/{object_name}")
            return True
        except S3Error as e:
            print(f"❌ MinIO upload error: {e}")
            return False
    
    def download_from_minio(self, bucket_name: str, object_name: str, local_path: str):
        """Download file from MinIO"""
        try:
            if not self.minio_client:
                return False
                
            self.minio_client.fget_object(bucket_name, object_name, local_path)
            print(f"✅ Downloaded from MinIO: {bucket_name}/{object_name} to {local_path}")
            return True
        except S3Error as e:
            print(f"❌ MinIO download error: {e}")
            return False
    
    def upload_to_s3(self, local_path: str, bucket_name: str, object_name: str):
        """Upload file to AWS S3"""
        try:
            if not self.s3_client:
                return False
                
            self.s3_client.upload_file(local_path, bucket_name, object_name)
            print(f"✅ Uploaded {local_path} to S3: {bucket_name}/{object_name}")
            return True
        except ClientError as e:
            print(f"❌ S3 upload error: {e}")
            return False
    
    def download_from_s3(self, bucket_name: str, object_name: str, local_path: str):
        """Download file from AWS S3"""
        try:
            if not self.s3_client:
                return False
                
            self.s3_client.download_file(bucket_name, object_name, local_path)
            print(f"✅ Downloaded from S3: {bucket_name}/{object_name} to {local_path}")
            return True
        except ClientError as e:
            print(f"❌ S3 download error: {e}")
            return False
    
    def list_minio_objects(self, bucket_name: str, prefix: str = ""):
        """List objects in MinIO bucket"""
        try:
            if not self.minio_client:
                return []
                
            objects = self.minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            print(f"❌ MinIO list error: {e}")
            return []

@app.function(
    image=finetune_image,
    gpu="A100", # Configurable GPU type
    volumes={"/vol": volume},
    timeout=3600, # Configurable timeout
    secrets=[modal.Secret.from_name("huggingface-secret")] # For Hugging Face access
)
def fine_tune_llm(
    model_name_or_path: str,
    dataset_path: str,
    output_dir: str = "/vol/finetuned_model",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 0.0002,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    use_4bit: bool = True,
    optimizer: str = "adamw_torch",
    gpu_type: str = "A100",
    timeout: int = 3600,
    storage_backend: str = "volume",  # "volume", "minio", or "s3"
    minio_bucket: str = "llm-models",
    s3_bucket: str = "llm-models"
):
    """
    Fine-tune an LLM using LoRA/QLoRA with multiple storage backends
    
    Args:
        model_name_or_path: HuggingFace model name or path to model
        dataset_path: Path to training dataset
        output_dir: Directory to save the fine-tuned model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        learning_rate: Learning rate for training
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        use_4bit: Whether to use 4-bit quantization (QLoRA)
        optimizer: Optimizer to use
        gpu_type: GPU type for training
        timeout: Training timeout in seconds
        storage_backend: Storage backend ("volume", "minio", or "s3")
        minio_bucket: MinIO bucket name
        s3_bucket: S3 bucket name
    """
    import torch
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        TrainingArguments, 
        Trainer,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset
    import json
    import time
    import GPUtil
    import psutil
    
    print(f"🚀 Starting fine-tuning job")
    print(f"📊 Configuration:")
    print(f"   Model: {model_name_or_path}")
    print(f"   Dataset: {dataset_path}")
    print(f"   Output: {output_dir}")
    print(f"   Epochs: {num_train_epochs}")
    print(f"   Batch Size: {per_device_train_batch_size}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   LoRA r: {lora_r}, alpha: {lora_alpha}, dropout: {lora_dropout}")
    print(f"   4-bit: {use_4bit}")
    print(f"   Optimizer: {optimizer}")
    print(f"   Storage: {storage_backend}")
    
    # Initialize storage manager
    storage = StorageManager()
    
    # Setup quantization config for QLoRA
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("✅ 4-bit quantization enabled")
    else:
        bnb_config = None
        print("✅ Full precision training")
    
    # Load model and tokenizer
    print(f"📥 Loading model: {model_name_or_path}")
    
    # Check if model is in storage backends
    model_loaded_from_storage = False
    local_model_path = f"/tmp/model_{int(time.time())}"
    
    if storage_backend == "minio":
        # Try to load model from MinIO
        model_objects = storage.list_minio_objects(minio_bucket, f"models/{model_name_or_path}/")
        if model_objects:
            print(f"📥 Loading model from MinIO: {minio_bucket}")
            os.makedirs(local_model_path, exist_ok=True)
            for obj in model_objects:
                local_file = os.path.join(local_model_path, os.path.basename(obj))
                if storage.download_from_minio(minio_bucket, obj, local_file):
                    model_loaded_from_storage = True
    
    elif storage_backend == "s3":
        # Try to load model from S3
        try:
            response = storage.s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=f"models/{model_name_or_path}/")
            if 'Contents' in response:
                print(f"📥 Loading model from S3: {s3_bucket}")
                os.makedirs(local_model_path, exist_ok=True)
                for obj in response['Contents']:
                    local_file = os.path.join(local_model_path, os.path.basename(obj['Key']))
                    if storage.download_from_s3(s3_bucket, obj['Key'], local_file):
                        model_loaded_from_storage = True
        except Exception as e:
            print(f"⚠️ Could not check S3 for model: {e}")
    
    # Load model from HuggingFace or local storage
    if model_loaded_from_storage:
        model_path = local_model_path
        print(f"✅ Using model from storage: {model_path}")
    else:
        model_path = model_name_or_path
        print(f"✅ Using HuggingFace model: {model_path}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ Model and tokenizer loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    # Setup LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    print(f"📥 Loading dataset: {dataset_path}")
    
    # Check if dataset is in storage
    dataset_loaded_from_storage = False
    local_dataset_path = f"/tmp/dataset_{int(time.time())}.jsonl"
    
    if storage_backend == "minio":
        if storage.download_from_minio(minio_bucket, f"datasets/{dataset_path}", local_dataset_path):
            dataset_path = local_dataset_path
            dataset_loaded_from_storage = True
    elif storage_backend == "s3":
        if storage.download_from_s3(s3_bucket, f"datasets/{dataset_path}", local_dataset_path):
            dataset_path = local_dataset_path
            dataset_loaded_from_storage = True
    
    if not dataset_loaded_from_storage and not os.path.exists(dataset_path):
        # Try to load from Modal volume
        volume_dataset_path = f"/vol/{dataset_path}"
        if os.path.exists(volume_dataset_path):
            dataset_path = volume_dataset_path
            print(f"✅ Using dataset from Modal volume: {dataset_path}")
        else:
            print(f"❌ Dataset not found: {dataset_path}")
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    try:
        if dataset_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=dataset_path, split='train')
        elif dataset_path.endswith('.json'):
            dataset = load_dataset('json', data_files=dataset_path, split='train')
        else:
            # Try to load as HuggingFace dataset
            dataset = load_dataset(dataset_path, split='train')
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=4,
        optim=optimizer,
        save_steps=500,
        logging_steps=100,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        save_strategy="steps",
        evaluation_strategy="no",
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Monitor system resources
    def log_system_stats():
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"📊 GPU: {gpu.memoryUtil*100:.1f}% memory, {gpu.load*100:.1f}% utilization")
        
        memory = psutil.virtual_memory()
        print(f"📊 RAM: {memory.percent:.1f}% used ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
    
    # Start training
    print("🏋️ Starting training...")
    log_system_stats()
    
    start_time = time.time()
    
    try:
        trainer.train()
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    
    # Save the model
    print(f"💾 Saving model to: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Upload to storage backends
    if storage_backend == "minio":
        print(f"📤 Uploading model to MinIO: {minio_bucket}")
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, output_dir)
                object_name = f"models/finetuned/{model_name_or_path.replace('/', '_')}/{relative_path}"
                storage.upload_to_minio(local_file, minio_bucket, object_name)
    
    elif storage_backend == "s3":
        print(f"📤 Uploading model to S3: {s3_bucket}")
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, output_dir)
                object_name = f"models/finetuned/{model_name_or_path.replace('/', '_')}/{relative_path}"
                storage.upload_to_s3(local_file, s3_bucket, object_name)
    
    # Final system stats
    log_system_stats()
    
    # Cleanup temporary files
    if model_loaded_from_storage and os.path.exists(local_model_path):
        shutil.rmtree(local_model_path)
    if dataset_loaded_from_storage and os.path.exists(local_dataset_path):
        os.remove(local_dataset_path)
    
    result = {
        "status": "completed",
        "model_path": output_dir,
        "training_time": training_time,
        "storage_backend": storage_backend,
        "final_loss": trainer.state.log_history[-1].get("train_loss", 0) if trainer.state.log_history else 0,
        "total_steps": trainer.state.global_step,
        "message": "Fine-tuning completed successfully"
    }
    
    print(f"🎉 Fine-tuning job completed successfully!")
    print(f"📊 Final results: {result}")
    
    return result

@app.function(
    image=finetune_image,
    volumes={"/vol": volume},
)
def list_models_and_datasets():
    """List available models and datasets from all storage backends"""
    storage = StorageManager()
    
    models = []
    datasets = []
    
    # List from Modal volume
    volume_models_dir = "/vol/models"
    volume_datasets_dir = "/vol/datasets"
    
    if os.path.exists(volume_models_dir):
        for item in os.listdir(volume_models_dir):
            models.append({
                "name": item,
                "path": f"/vol/models/{item}",
                "source": "modal_volume",
                "size": "unknown"
            })
    
    if os.path.exists(volume_datasets_dir):
        for item in os.listdir(volume_datasets_dir):
            datasets.append({
                "name": item,
                "path": f"/vol/datasets/{item}",
                "source": "modal_volume",
                "size": "unknown"
            })
    
    # List from MinIO
    try:
        minio_models = storage.list_minio_objects("llm-models", "models/")
        for model in minio_models:
            models.append({
                "name": os.path.basename(model),
                "path": model,
                "source": "minio",
                "size": "unknown"
            })
        
        minio_datasets = storage.list_minio_objects("llm-models", "datasets/")
        for dataset in minio_datasets:
            datasets.append({
                "name": os.path.basename(dataset),
                "path": dataset,
                "source": "minio",
                "size": "unknown"
            })
    except Exception as e:
        print(f"⚠️ Could not list MinIO objects: {e}")
    
    # List from S3
    try:
        if storage.s3_client:
            response = storage.s3_client.list_objects_v2(Bucket="llm-models", Prefix="models/")
            if 'Contents' in response:
                for obj in response['Contents']:
                    models.append({
                        "name": os.path.basename(obj['Key']),
                        "path": obj['Key'],
                        "source": "s3",
                        "size": obj['Size']
                    })
            
            response = storage.s3_client.list_objects_v2(Bucket="llm-models", Prefix="datasets/")
            if 'Contents' in response:
                for obj in response['Contents']:
                    datasets.append({
                        "name": os.path.basename(obj['Key']),
                        "path": obj['Key'],
                        "source": "s3",
                        "size": obj['Size']
                    })
    except Exception as e:
        print(f"⚠️ Could not list S3 objects: {e}")
    
    return {
        "models": models,
        "datasets": datasets,
        "storage_backends": ["modal_volume", "minio", "s3"]
    }

@app.function(
    image=finetune_image,
    volumes={"/vol": volume},
)
def upload_file(file_bytes: bytes, file_name: str, storage_backend: str = "volume"):
    """Upload a file to the specified storage backend"""
    storage = StorageManager()
    
    # Save file temporarily
    temp_path = f"/tmp/{file_name}"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    
    success = False
    final_path = ""
    
    if storage_backend == "volume":
        # Save to Modal volume
        volume_path = f"/vol/{file_name}"
        shutil.copy2(temp_path, volume_path)
        success = True
        final_path = volume_path
        
    elif storage_backend == "minio":
        # Upload to MinIO
        bucket_name = "llm-models"
        object_name = f"datasets/{file_name}"
        success = storage.upload_to_minio(temp_path, bucket_name, object_name)
        final_path = f"minio://{bucket_name}/{object_name}"
        
    elif storage_backend == "s3":
        # Upload to S3
        bucket_name = "llm-models"
        object_name = f"datasets/{file_name}"
        success = storage.upload_to_s3(temp_path, bucket_name, object_name)
        final_path = f"s3://{bucket_name}/{object_name}"
    
    # Cleanup
    os.remove(temp_path)
    
    return {
        "success": success,
        "filename": file_name,
        "path": final_path,
        "storage_backend": storage_backend
    }

if __name__ == "__main__":
    # This function can be used to test the fine-tuning locally or via modal run
    print("🧪 Testing fine-tuning function...")
    
    # Example usage
    result = fine_tune_llm.remote(
        model_name_or_path="microsoft/DialoGPT-small",
        dataset_path="dummy_dataset.jsonl",
        output_dir="/vol/test_finetuned_model",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        storage_backend="volume"
    )
    
    print(f"✅ Test completed: {result}")

