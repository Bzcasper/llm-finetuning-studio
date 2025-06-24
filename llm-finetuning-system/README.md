# LLM Fine-Tuning Studio

A comprehensive, professional-grade fine-tuning platform powered by Modal.com for training Large Language Models with modern techniques like LoRA, QLoRA, and advanced optimizers.

## 🚀 Features

### Core Capabilities
- **Universal Model Support**: Fine-tune any HuggingFace model or custom models
- **Flexible Dataset Sources**: Support for HuggingFace datasets, file uploads, and Modal volume storage
- **Modern Training Techniques**: LoRA, QLoRA, 4-bit quantization, and multiple optimizers
- **GPU Acceleration**: Full Modal.com GPU support (T4, L4, A10G, A100, H100, H200, B200)
- **Real-time Monitoring**: Live training progress, loss/accuracy tracking, and GPU utilization
- **Professional UI**: Modern React-based interface with comprehensive configuration options

### Advanced Features
- **Dynamic Configuration**: Automatically adjust training parameters based on model and dataset
- **Built-in Monitoring**: Custom Weights & Biases-style visualization and logging
- **Secure Credential Management**: Safe storage and handling of API keys and tokens
- **Scalable Architecture**: Containerized deployment with Modal.com cloud infrastructure
- **Production Ready**: Full error handling, logging, and monitoring capabilities

## 📋 System Requirements

- Python 3.11+
- Node.js 20+
- Modal.com account with valid credentials
- HuggingFace account (optional, for private models)
- Weights & Biases account (optional, for experiment tracking)

## 🛠️ Installation & Setup

### 1. Clone and Setup Backend

```bash
# Install Python dependencies
pip install modal fastapi uvicorn transformers torch peft datasets accelerate bitsandbytes trl

# Set up Modal.com credentials
modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET --profile=ai-tool-pool

# Deploy the fine-tuning functions to Modal
export MODAL_PROFILE=ai-tool-pool
modal deploy finetune_llm.py
```

### 2. Setup Frontend

```bash
# Create React application (if not already created)
npx create-react-app llm-finetuning-gui
cd llm-finetuning-gui

# Install required dependencies
npm install recharts lucide-react

# Copy the provided App.jsx and other components
# Start development server
npm start
```

### 3. Start Backend API

```bash
# Start the FastAPI backend server
export MODAL_PROFILE=ai-tool-pool
python api_server.py
```

The system will be available at:
- Frontend: http://localhost:3000 (or 5173 for Vite)
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 🎯 Quick Start Guide

### Step 1: Configure Credentials
1. Navigate to the **Credentials** tab
2. Enter your Modal.com credentials:
   - Token ID
   - Token Secret
   - API Key
3. Optionally add HuggingFace and Weights & Biases tokens
4. Click "Save Credentials"

### Step 2: Select Model and Dataset
1. Go to **Configuration** tab
2. Choose your base model (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
3. Configure LoRA parameters:
   - Rank (r): 8-64 (higher = more parameters)
   - Alpha: 16-32 (scaling factor)
   - Dropout: 0.05-0.1 (regularization)
4. Set training parameters:
   - Learning rate: 0.0001-0.001
   - Epochs: 1-10
   - Batch size: 1-8 (depending on GPU memory)
5. Select GPU type based on your needs

### Step 3: Prepare Dataset
1. Navigate to **Dataset** tab
2. Choose one of three options:
   - **Upload**: Local files (JSON, JSONL, CSV, TXT)
   - **HuggingFace**: Use dataset ID (e.g., `tatsu-lab/alpaca`)
   - **Volume Storage**: Select from Modal volume

### Step 4: Start Training
1. Go to **Training** tab
2. Review your configuration summary
3. Click "Start Training"
4. Monitor progress in real-time:
   - Training logs
   - Loss and accuracy metrics
   - GPU utilization
   - Estimated completion time

### Step 5: Monitor and Visualize
1. Switch to **Monitoring** tab during training
2. View real-time charts:
   - Training loss progression
   - Accuracy improvement
   - GPU utilization and memory usage
3. Download trained models when complete

## 🔧 Configuration Options

### Model Configuration
- **Model Name/Path**: HuggingFace model ID or local path
- **Output Directory**: Where to save the fine-tuned model
- **GPU Type**: T4, L4, A10G, A100, H100, H200, B200

### LoRA Configuration
- **Rank (r)**: 4-128, controls the number of trainable parameters
- **Alpha**: Scaling factor, typically 2x the rank
- **Dropout**: Regularization, 0.05-0.1 recommended
- **4-bit Quantization**: Enable QLoRA for memory efficiency

### Training Parameters
- **Learning Rate**: 0.0001-0.001, start with 0.0002
- **Epochs**: Number of training iterations
- **Batch Size**: Samples per GPU, adjust based on memory
- **Gradient Accumulation**: Effective batch size multiplier
- **Optimizer**: AdamW (PyTorch/HuggingFace), SGD, Adafactor

### System Configuration
- **Timeout**: Maximum training time in seconds
- **GPU Memory**: Automatically managed based on model size

## 📊 Monitoring and Visualization

### Real-time Metrics
- **Training Progress**: Live progress bar with epoch information
- **Loss Tracking**: Real-time loss reduction visualization
- **Accuracy Monitoring**: Model performance improvement
- **GPU Metrics**: Utilization and memory consumption
- **Time Estimation**: Remaining training time prediction

### Built-in Dashboards
- **Training Loss Chart**: Line chart showing loss progression
- **Accuracy Chart**: Performance improvement over epochs
- **GPU Utilization**: Real-time resource monitoring
- **Memory Usage**: GPU memory consumption tracking

### Logging System
- **Structured Logs**: Timestamped training events
- **Error Handling**: Comprehensive error reporting
- **Progress Updates**: Detailed epoch-by-epoch information
- **Export Options**: Download logs and metrics

## 🚀 Deployment Options

### Local Development
```bash
# Backend
python api_server.py

# Frontend
npm start
```

### Production Deployment

#### Backend (Modal.com)
```bash
# Deploy functions to Modal
modal deploy finetune_llm.py

# Deploy API server (optional)
modal deploy api_server.py
```

#### Frontend (Vercel/Netlify)
```bash
# Build for production
npm run build

# Deploy to your preferred platform
# Update API_BASE_URL in the frontend code
```

### Docker Deployment
```dockerfile
# Backend Dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "api_server.py"]

# Frontend Dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
CMD ["npm", "start"]
```

## 🔐 Security Considerations

### Credential Management
- Store Modal.com credentials securely
- Use environment variables for production
- Implement proper secret rotation
- Limit API key permissions

### Network Security
- Use HTTPS in production
- Implement proper CORS policies
- Add authentication/authorization
- Monitor API usage

### Data Privacy
- Encrypt sensitive training data
- Implement data retention policies
- Ensure compliance with regulations
- Audit data access patterns

## 🐛 Troubleshooting

### Common Issues

#### Modal Connection Failed
```bash
# Check credentials
modal token list

# Verify deployment
modal app list

# Re-deploy if needed
modal deploy finetune_llm.py
```

#### GPU Memory Issues
- Reduce batch size
- Enable 4-bit quantization
- Use gradient accumulation
- Choose smaller model or LoRA rank

#### Training Failures
- Check dataset format
- Verify model compatibility
- Review error logs
- Adjust timeout settings

#### Frontend Connection Issues
- Verify backend is running
- Check CORS configuration
- Update API_BASE_URL
- Review browser console

### Performance Optimization

#### Training Speed
- Use appropriate GPU type
- Optimize batch size
- Enable mixed precision
- Use gradient checkpointing

#### Memory Efficiency
- Enable QLoRA (4-bit quantization)
- Reduce sequence length
- Use smaller LoRA rank
- Implement gradient accumulation

#### Cost Optimization
- Choose cost-effective GPU types
- Implement early stopping
- Use spot instances when available
- Monitor resource usage

## 📚 API Reference

### Training Endpoints
- `POST /api/training/start` - Start new training job
- `GET /api/training/status/{job_id}` - Get training status
- `GET /api/training/logs/{job_id}` - Get training logs
- `GET /api/training/jobs` - List all jobs

### Data Management
- `GET /api/datasets` - List available datasets
- `GET /api/models` - List available models
- `POST /api/upload` - Upload dataset file

### System Endpoints
- `GET /api/health` - Health check
- `GET /api/modal/status` - Modal connection status
- `POST /api/test/modal` - Test Modal connection

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Install dependencies
4. Make changes
5. Test thoroughly
6. Submit pull request

### Code Standards
- Follow PEP 8 for Python
- Use ESLint for JavaScript
- Add comprehensive tests
- Document new features
- Update README as needed

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Modal.com for cloud GPU infrastructure
- HuggingFace for model and dataset ecosystem
- React and FastAPI communities
- LoRA and QLoRA research teams

## 📞 Support

For support and questions:
- Create GitHub issues for bugs
- Join our Discord community
- Check documentation wiki
- Contact support team

---

**Built with ❤️ for the AI community**

