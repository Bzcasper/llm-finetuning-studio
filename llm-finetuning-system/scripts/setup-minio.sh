#!/bin/bash

# MinIO Deployment Script for Modal.com LLM Fine-tuning System

echo "🚀 Setting up MinIO for LLM Fine-tuning System..."

# Create MinIO deployment using Docker
echo "📦 Creating MinIO container..."

# Create data directory
mkdir -p ./minio-data

# Start MinIO container
docker run -d \
  --name llm-minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin123" \
  -v $(pwd)/minio-data:/data \
  minio/minio server /data --console-address ":9001"

echo "⏳ Waiting for MinIO to start..."
sleep 10

# Install MinIO client
echo "📥 Installing MinIO client..."
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/

# Configure MinIO client
echo "🔧 Configuring MinIO client..."
mc alias set local http://localhost:9000 minioadmin minioadmin123

# Create buckets
echo "🪣 Creating buckets..."
mc mb local/llm-models
mc mb local/llm-datasets
mc mb local/llm-checkpoints

# Set bucket policies (public read for models, private for datasets)
echo "🔒 Setting bucket policies..."
cat > /tmp/public-read-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": ["*"]
      },
      "Action": ["s3:GetObject"],
      "Resource": ["arn:aws:s3:::llm-models/*"]
    }
  ]
}
EOF

mc policy set-json /tmp/public-read-policy.json local/llm-models

echo "✅ MinIO setup completed!"
echo ""
echo "📊 MinIO Console: http://localhost:9001"
echo "🔑 Username: minioadmin"
echo "🔑 Password: minioadmin123"
echo ""
echo "🌐 MinIO API: http://localhost:9000"
echo ""
echo "📝 Environment variables to add:"
echo "MINIO_ENDPOINT=localhost:9000"
echo "MINIO_ACCESS_KEY=minioadmin"
echo "MINIO_SECRET_KEY=minioadmin123"
echo "MINIO_SECURE=false"

