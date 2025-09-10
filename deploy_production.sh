#!/bin/bash
# Persian Legal AI Training System - Production Deployment Script
# Generated: 2025-09-10

set -e

echo "🚀 Persian Legal AI Production Deployment"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "persian_legal_ai.db" ]; then
    echo "❌ Error: persian_legal_ai.db not found. Run from project root."
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed. Please log out and back in, then run this script again."
    exit 0
fi

echo "🔍 Pre-deployment checks..."

# Test local build
echo "📦 Building production Docker image..."
docker build -f Dockerfile.production -t persian-legal-ai-backend .

# Test the container locally
echo "🧪 Testing container locally..."
docker run -d --name test-container -p 8001:8000 persian-legal-ai-backend

# Wait for container to start
sleep 10

# Test health endpoint
if curl -f http://localhost:8001/api/system/health > /dev/null 2>&1; then
    echo "✅ Container test successful!"
else
    echo "❌ Container test failed!"
    docker logs test-container
    docker stop test-container
    docker rm test-container
    exit 1
fi

# Clean up test container
docker stop test-container
docker rm test-container

echo "✅ Production build ready!"
echo ""
echo "📋 Next steps for deployment:"
echo "   1. Railway: railway login && railway up"
echo "   2. Heroku: heroku create your-app-name && git push heroku main"
echo "   3. DigitalOcean App Platform: doctl apps create --spec railway.toml"
echo "   4. Google Cloud Run: gcloud run deploy --image gcr.io/PROJECT/persian-legal-ai"
echo ""
echo "🔧 Environment variables to set on your platform:"
echo "   - ENVIRONMENT=production"
echo "   - DATABASE_URL=sqlite:///persian_legal_ai.db"
echo "   - PERSIAN_BERT_MODEL=HooshvareLab/bert-fa-base-uncased"
echo "   - LOG_LEVEL=INFO"
echo ""
echo "🎯 Deployment complete! Your backend will be available at your platform's URL."