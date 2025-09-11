# Bolt Backend Implementation Guide
Generated: 2025-09-10T15:12:43.613Z

## 🎯 Overview
The Bolt frontend integration is complete and ready. This guide shows how to implement the required backend endpoints.

## 📋 Required Endpoints

### 1. Health Check
```python
# backend/api/bolt_endpoints.py
from fastapi import APIRouter, UploadFile, HTTPException, Depends
from typing import List, Dict, Any
import logging
from datetime import datetime
import uuid

router = APIRouter(prefix="/bolt", tags=["bolt"])
logger = logging.getLogger(__name__)

@router.get("/health")
async def bolt_health():
    """Bolt service health check"""
    return {
        "status": "healthy", 
        "service": "bolt", 
        "timestamp": datetime.utcnow().isoformat()
    }
```

### 2. Document Management
```python
@router.post("/documents/upload")
async def upload_document(file: UploadFile):
    """Upload a document for processing"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Save file
        file_id = str(uuid.uuid4())
        file_path = f"uploads/bolt/{file_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create document record
        document = {
            "id": file_id,
            "filename": file.filename,
            "size": len(content),
            "status": "uploaded",
            "upload_date": datetime.utcnow().isoformat(),
            "file_path": file_path
        }
        
        # TODO: Save to database
        # await db.documents.insert(document)
        
        return document
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/documents")
async def list_documents():
    """List all documents"""
    # TODO: Get from database
    # documents = await db.documents.find_all()
    return {
        "documents": [],
        "total": 0,
        "status": "ready"
    }

@router.post("/documents/{doc_id}/process")
async def process_document(doc_id: str):
    """Process an uploaded document"""
    try:
        # TODO: Start background processing
        # task = process_document_task.delay(doc_id)
        
        return {
            "doc_id": doc_id,
            "status": "processing",
            "message": "Document processing started"
        }
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")
```

### 3. Training & Models
```python
@router.post("/training/start")
async def start_training(request: dict):
    """Start model training"""
    model_id = request.get("model_id")
    config = request.get("config", {})
    
    session = {
        "id": str(uuid.uuid4()),
        "model_id": model_id,
        "status": "running",
        "config": config,
        "start_time": datetime.utcnow().isoformat(),
        "progress": 0
    }
    
    # TODO: Save session and start background task
    # await db.training_sessions.insert(session)
    # await start_training_task(session["id"])
    
    return session

@router.get("/training/status")
async def get_training_status():
    """Get current training status"""
    # TODO: Get from database
    # sessions = await db.training_sessions.find_active()
    return {
        "sessions": [],
        "active_count": 0
    }

@router.get("/models")
async def list_models():
    """List available models"""
    # TODO: Get from database
    # models = await db.models.find_all()
    return {
        "models": [],
        "total": 0
    }
```

### 4. Analytics
```python
@router.get("/analytics")
async def get_analytics():
    """Get Bolt analytics"""
    # TODO: Calculate real metrics
    return {
        "total_documents": 0,
        "processed_documents": 0,
        "success_rate": 0.0,
        "models_trained": 0,
        "active_sessions": 0,
        "last_updated": datetime.utcnow().isoformat()
    }

@router.get("/analytics/performance")
async def get_performance_metrics():
    """Get performance metrics"""
    # TODO: Get real system metrics
    return {
        "cpu_usage": 45.2,
        "memory_usage": 67.8,
        "disk_usage": 23.1,
        "processing_speed": 1.5,
        "timestamp": datetime.utcnow().isoformat()
    }
```

## 🔧 Integration Steps

### 1. Add Router to FastAPI App
```python
# backend/main.py
from fastapi import FastAPI
from api.bolt_endpoints import router as bolt_router

app = FastAPI(title="Persian Legal AI with Bolt")

# Include Bolt router
app.include_router(bolt_router)

# Enable CORS for frontend
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 2. Database Models
```python
# models/bolt_models.py
from sqlalchemy import Column, String, Integer, DateTime, JSON, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Document(Base):
    __tablename__ = "bolt_documents"
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    size = Column(Integer)
    status = Column(String, default="uploaded")
    upload_date = Column(DateTime)
    file_path = Column(String)

class TrainingSession(Base):
    __tablename__ = "bolt_training_sessions"
    
    id = Column(String, primary_key=True)
    model_id = Column(String)
    status = Column(String, default="running")
    config = Column(JSON)
    start_time = Column(DateTime)
    progress = Column(Float, default=0.0)

class Model(Base):
    __tablename__ = "bolt_models"
    
    id = Column(String, primary_key=True)
    name = Column(String)
    type = Column(String)
    status = Column(String, default="active")
    accuracy = Column(Float)
    created_at = Column(DateTime)
```

### 3. Environment Setup
```bash
# .env
BOLT_UPLOAD_DIR=uploads/bolt
BOLT_MAX_FILE_SIZE=10485760  # 10MB
BOLT_PROCESSING_TIMEOUT=3600  # 1 hour
DATABASE_URL=postgresql://user:pass@localhost/db
```

## 🧪 Testing

### Frontend Testing
The frontend is ready to test once backend is implemented:

```javascript
// Test in browser console
import { boltApi } from './api/boltApi';

// Test health
await boltApi.healthCheck();

// Test file upload
const file = new File(['test content'], 'test.txt');
await boltApi.uploadDocument(file);

// Test analytics
await boltApi.getAnalytics();
```

### Backend Testing
```bash
# Test endpoints with curl
curl http://localhost:8000/api/bolt/health
curl http://localhost:8000/api/bolt/documents
curl http://localhost:8000/api/bolt/analytics
```

## 🚀 Deployment Checklist

- [ ] Implement all Bolt endpoints
- [ ] Set up database models
- [ ] Configure file storage
- [ ] Add authentication/authorization
- [ ] Set up background task processing
- [ ] Configure logging
- [ ] Add monitoring and metrics
- [ ] Test all endpoints
- [ ] Deploy and verify integration

## 📊 Current Status

✅ **Frontend Integration**: Complete
- Bolt components migrated and integrated
- API client configured
- Routes and navigation added
- Error handling implemented
- TypeScript types defined

⚠️  **Backend Integration**: Needs Implementation
- API endpoints need to be created
- Database models need setup
- Background processing needs configuration

## 🎉 Next Steps

1. **Implement the backend endpoints** using the code above
2. **Test the integration** between frontend and backend
3. **Deploy** and monitor the system
4. **Iterate** based on user feedback

The frontend Bolt integration is production-ready and waiting for the backend implementation!
