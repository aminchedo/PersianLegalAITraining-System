#!/bin/bash
set -e

echo "🧪 Testing Frontend Build and Deployment..."

# Clean previous builds
npm run clean

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Type checking
echo "🔍 Running TypeScript checks..."
npm run type-check

# Build project
echo "🏗️ Building project..."
npm run build

# Test build output
echo "✅ Testing build output..."
if [ -d ".next" ]; then
    echo "✅ Build successful - .next directory created"
else
    echo "❌ Build failed - .next directory not found"
    exit 1
fi

# Start development server for testing
echo "🚀 Starting development server..."
npm run dev &
DEV_PID=$!

# Wait for server to start
sleep 10

# Test if server responds
echo "🧪 Testing server response..."
if curl -f http://localhost:3000/ > /dev/null 2>&1; then
    echo "✅ Frontend server is responding"
else
    echo "❌ Frontend server is not responding"
    kill $DEV_PID
    exit 1
fi

# Cleanup
kill $DEV_PID
echo "✅ All frontend tests passed!"