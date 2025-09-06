#!/bin/bash

# Persian Legal AI Training System - Deployment Script
# اسکریپت استقرار سیستم آموزش هوش مصنوعی حقوقی فارسی

echo "🚀 Starting Persian Legal AI Training System Deployment..."

# Build frontend
echo "📦 Building frontend..."
cd frontend
npm run build

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Frontend build successful!"
else
    echo "❌ Frontend build failed!"
    exit 1
fi

# Deploy to GitHub Pages
echo "🌐 Deploying to GitHub Pages..."
npm run deploy

if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo "🎉 Persian Legal AI Training System is now live!"
else
    echo "❌ Deployment failed!"
    exit 1
fi

echo "🔗 Your application should be available at:"
echo "   https://<USERNAME>.github.io/persian-legal-ai/"
echo ""
echo "📊 Backend API endpoints:"
echo "   - System Stats: /api/real/stats"
echo "   - Team Members: /api/real/team/members"
echo "   - Training Jobs: /api/real/models/training"
echo "   - System Metrics: /api/real/monitoring/system-metrics"
echo ""
echo "🔐 Demo credentials:"
echo "   - Admin: admin/admin123"
echo "   - Trainer: trainer/trainer123"
echo "   - Viewer: viewer/viewer123"