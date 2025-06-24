#!/bin/bash

# LLM Fine-Tuning System Deployment Script

echo "🚀 Starting LLM Fine-Tuning System Deployment..."

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Installing..."
    pip install modal
fi

# Check if Modal token is configured
if ! modal token list &> /dev/null; then
    echo "❌ Modal token not configured. Please run:"
    echo "modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET"
    exit 1
fi

# Install minimal local dependencies (only for API server)
echo "📦 Installing local dependencies for API server..."
pip install -r requirements.txt

# Deploy Modal functions (this will install all ML dependencies in the container)
echo "☁️ Deploying Modal functions with all ML dependencies..."
echo "   This will install torch, transformers, peft, and all other ML libraries in the Modal container..."
export MODAL_PROFILE=ai-tool-pool
modal deploy finetune_llm.py

# Install frontend dependencies
echo "🎨 Installing frontend dependencies..."
cd frontend
npm install

# Build frontend for production
echo "🏗️ Building frontend..."
npm run build

cd ..

echo "✅ Deployment complete!"
echo ""
echo "📋 Architecture Overview:"
echo "   • Local: Only Modal CLI, FastAPI, and frontend dependencies"
echo "   • Modal Container: All ML libraries (torch, transformers, peft, etc.)"
echo "   • This ensures optimal resource usage and faster local development"
echo ""
echo "To start the system:"
echo "1. Backend: python api_server.py"
echo "2. Frontend: cd frontend && npm start"
echo ""
echo "Access the application at:"
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"

