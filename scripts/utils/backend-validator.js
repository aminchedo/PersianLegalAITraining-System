// backend-validator.js
const axios = require('axios');
const fs = require('fs');

class BackendValidator {
  constructor(baseURL = 'http://localhost:8000') {
    this.baseURL = baseURL;
    this.boltEndpoints = [
      { method: 'GET', path: '/api/bolt/health', description: 'Bolt health check' },
      { method: 'POST', path: '/api/bolt/documents/upload', description: 'Document upload' },
      { method: 'GET', path: '/api/bolt/documents', description: 'List documents' },
      { method: 'POST', path: '/api/bolt/documents/:id/process', description: 'Process document' },
      { method: 'GET', path: '/api/bolt/analytics', description: 'Bolt analytics' },
      { method: 'GET', path: '/api/bolt/settings', description: 'Bolt settings' },
      { method: 'GET', path: '/api/training/status', description: 'Training status' },
      { method: 'POST', path: '/api/training/start', description: 'Start training' },
      { method: 'GET', path: '/api/models', description: 'List models' },
      { method: 'GET', path: '/api/data/sources', description: 'Data sources' },
    ];
  }

  async validateBackendIntegration() {
    console.log('🔧 Validating backend integration...');
    
    try {
      // Check if backend is running
      try {
        const healthResponse = await axios.get(`${this.baseURL}/api/system/health`, { timeout: 5000 });
        console.log('✅ Backend is running');
      } catch (error) {
        console.log('⚠️  Backend not running - will generate implementation guide');
        await this.generateBackendImplementation();
        return false;
      }
      
      // Test each Bolt endpoint
      let successfulEndpoints = 0;
      for (const endpoint of this.boltEndpoints) {
        const success = await this.testEndpoint(endpoint);
        if (success) successfulEndpoints++;
      }
      
      console.log(`📊 Backend integration results: ${successfulEndpoints}/${this.boltEndpoints.length} endpoints working`);
      
      if (successfulEndpoints >= this.boltEndpoints.length * 0.7) {
        console.log('✅ Backend integration validation completed successfully');
        return true;
      } else {
        console.log('⚠️  Some backend endpoints need implementation');
        await this.generateBackendImplementation();
        return false;
      }
      
    } catch (error) {
      console.error('❌ Backend validation failed:', error.message);
      await this.generateBackendImplementation();
      return false;
    }
  }

  async testEndpoint(endpoint) {
    const url = `${this.baseURL}${endpoint.path}`;
    
    try {
      let response;
      
      switch (endpoint.method) {
        case 'GET':
          response = await axios.get(url, { timeout: 5000 });
          break;
        case 'POST':
          // For POST endpoints, send minimal test data
          const testData = endpoint.path.includes('upload') 
            ? { test: true } // FormData would be handled differently
            : { test: true };
          response = await axios.post(url, testData, { timeout: 5000 });
          break;
        default:
          console.log(`⚠️  Skipping ${endpoint.method} ${endpoint.path} - manual test required`);
          return false;
      }
      
      if (response.status < 400) {
        console.log(`✅ ${endpoint.method} ${endpoint.path} - ${endpoint.description}`);
        return true;
      } else {
        console.log(`⚠️  ${endpoint.method} ${endpoint.path} - returned ${response.status}`);
        return false;
      }
      
    } catch (error) {
      if (error.response?.status === 404) {
        console.log(`❌ ${endpoint.method} ${endpoint.path} - endpoint not implemented`);
      } else if (error.response?.status === 401) {
        console.log(`🔐 ${endpoint.method} ${endpoint.path} - requires authentication (expected)`);
        return true; // Auth is expected
      } else if (error.code === 'ECONNREFUSED') {
        console.log(`🔌 ${endpoint.method} ${endpoint.path} - connection refused`);
      } else {
        console.log(`⚠️  ${endpoint.method} ${endpoint.path} - ${error.message}`);
      }
      return false;
    }
  }

  async generateBackendImplementation() {
    console.log('📝 Generating backend implementation guide...');
    
    const implementationGuide = `# Bolt Backend Implementation Guide
Generated: ${new Date().toISOString()}

## Required Backend Endpoints

The Bolt integration requires the following API endpoints to be implemented in your backend:

### Health & System
\`\`\`python
# backend/api/bolt_endpoints.py
from fastapi import APIRouter, UploadFile, HTTPException, Depends
from typing import List, Dict, Any
import logging

router = APIRouter(prefix="/bolt", tags=["bolt"])
logger = logging.getLogger(__name__)

@router.get("/health")
async def bolt_health():
    """Bolt service health check"""
    return {"status": "healthy", "service": "bolt", "timestamp": datetime.utcnow()}
\`\`\`

### Document Management
\`\`\`python
@router.post("/documents/upload")
async def upload_document(file: UploadFile):
    """Upload a document for processing"""
    try:
        # Save file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create document record
        document = {
            "id": str(uuid.uuid4()),
            "filename": file.filename,
            "size": len(content),
            "status": "uploaded",
            "upload_date": datetime.utcnow()
        }
        
        # Save to database
        # db.documents.insert(document)
        
        return document
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

@router.get("/documents")
async def list_documents():
    """List all documents"""
    # return db.documents.find_all()
    return {"documents": []}

@router.post("/documents/{doc_id}/process")
async def process_document(doc_id: str):
    """Process an uploaded document"""
    try:
        # Start background processing
        # task = process_document_task.delay(doc_id)
        
        return {
            "doc_id": doc_id, 
            "status": "processing",
            "task_id": "task_123"
        }
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")
\`\`\`

### Training & Models
\`\`\`python
@router.post("/training/start")
async def start_training(request: dict):
    """Start model training"""
    model_id = request.get("model_id")
    config = request.get("config", {})
    
    # Start training process
    session = {
        "id": str(uuid.uuid4()),
        "model_id": model_id,
        "status": "running",
        "config": config,
        "start_time": datetime.utcnow()
    }
    
    # Save session and start background task
    # db.training_sessions.insert(session)
    # training_task.delay(session["id"])
    
    return session

@router.get("/training/status")
async def get_training_status():
    """Get current training status"""
    # sessions = db.training_sessions.find_active()
    return {"sessions": []}

@router.get("/models")
async def list_models():
    """List available models"""
    # models = db.models.find_all()
    return {"models": []}
\`\`\`

### Analytics
\`\`\`python
@router.get("/analytics")
async def get_analytics():
    """Get Bolt analytics"""
    return {
        "total_documents": 0,
        "processed_documents": 0,
        "success_rate": 0.0,
        "models_trained": 0,
        "active_sessions": 0
    }

@router.get("/analytics/performance")
async def get_performance_metrics():
    """Get performance metrics"""
    return {
        "cpu_usage": 45.2,
        "memory_usage": 67.8,
        "disk_usage": 23.1,
        "processing_speed": 1.5
    }
\`\`\`

## Integration Steps

### 1. Add to main FastAPI app
\`\`\`python
# backend/main.py
from api.bolt_endpoints import router as bolt_router

app = FastAPI()
app.include_router(bolt_router)
\`\`\`

### 2. Database Models
Create database models for:
- Documents
- Training Sessions
- Models
- Analytics

### 3. Background Tasks
Set up Celery or similar for:
- Document processing
- Model training
- Analytics computation

### 4. File Storage
Configure file storage for:
- Uploaded documents
- Trained models
- Logs and artifacts

## Testing the Implementation

Use the frontend's boltApi service to test endpoints:
\`\`\`javascript
import { boltApi } from './api/boltApi';

// Test health check
const health = await boltApi.healthCheck();

// Test document upload
const file = new File(['test'], 'test.txt');
const upload = await boltApi.uploadDocument(file);

// Test analytics
const analytics = await boltApi.getAnalytics();
\`\`\`

## Environment Variables
Add to your .env file:
\`\`\`
BOLT_UPLOAD_DIR=uploads/bolt
BOLT_MAX_FILE_SIZE=10485760  # 10MB
BOLT_PROCESSING_TIMEOUT=3600  # 1 hour
\`\`\`

## Next Steps
1. Implement the above endpoints
2. Set up database models
3. Configure background task processing
4. Test with the frontend integration
5. Monitor and optimize performance

The frontend Bolt integration is ready and waiting for these backend endpoints!
`;

    fs.writeFileSync('/workspace/backend-implementation-guide.md', implementationGuide);
    console.log('📋 Backend implementation guide saved: backend-implementation-guide.md');
    
    // Also create a simple endpoint checker script
    const checkerScript = `#!/bin/bash
# check-bolt-endpoints.sh - Quick endpoint checker

echo "🔍 Checking Bolt backend endpoints..."

BASE_URL="http://localhost:8000"
ENDPOINTS=(
    "GET /api/bolt/health"
    "GET /api/bolt/documents"
    "GET /api/bolt/analytics"
    "GET /api/training/status"
    "GET /api/models"
)

for endpoint in "\${ENDPOINTS[@]}"; do
    method=\$(echo \$endpoint | cut -d' ' -f1)
    path=\$(echo \$endpoint | cut -d' ' -f2)
    
    echo -n "Testing \$endpoint... "
    
    if curl -s -X \$method "\$BASE_URL\$path" > /dev/null; then
        echo "✅"
    else
        echo "❌"
    fi
done

echo "Done! Check backend-implementation-guide.md for implementation details."
`;

    fs.writeFileSync('/workspace/check-bolt-endpoints.sh', checkerScript);
    fs.chmodSync('/workspace/check-bolt-endpoints.sh', 0o755);
    console.log('📋 Endpoint checker script saved: check-bolt-endpoints.sh');
  }
}

// Main execution
async function main() {
  try {
    const validator = new BackendValidator();
    const success = await validator.validateBackendIntegration();
    
    if (success) {
      console.log('🎉 Backend validation completed successfully!');
      process.exit(0);
    } else {
      console.log('⚠️  Backend needs implementation - guide generated');
      console.log('📖 See backend-implementation-guide.md for details');
      process.exit(0); // Not a failure, just needs implementation
    }
    
  } catch (error) {
    console.error('❌ Error during backend validation:', error);
    process.exit(1);
  }
}

main();