#!/bin/bash
# check-bolt-endpoints.sh - Quick endpoint checker

echo "🔍 Checking Bolt backend endpoints..."

BASE_URL="http://localhost:8000"
ENDPOINTS=(
    "/api/bolt/health"
    "/api/bolt/documents" 
    "/api/bolt/analytics"
    "/api/training/status"
    "/api/models"
)

echo "Testing against: $BASE_URL"
echo "================================"

for endpoint in "${ENDPOINTS[@]}"; do
    echo -n "GET $endpoint ... "
    
    if command -v curl >/dev/null 2>&1; then
        if curl -s -f "$BASE_URL$endpoint" > /dev/null 2>&1; then
            echo "✅ Working"
        else
            echo "❌ Not implemented"
        fi
    else
        echo "⚠️  curl not available"
    fi
done

echo ""
echo "📖 See backend-implementation-guide.md for implementation details"
echo "🎯 Frontend integration is ready - implement backend endpoints to complete!"
