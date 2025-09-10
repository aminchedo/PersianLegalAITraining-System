import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import uvicorn
from datetime import datetime

# Import local modules
from config.database import get_database, init_database, test_connection
from ai_classifier import classifier

# Initialize FastAPI app
app = FastAPI(
    title="Persian Legal AI System",
    description="Advanced Persian Legal Document Processing and AI Classification",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TextClassificationRequest(BaseModel):
    text: str
    include_confidence: bool = True

class SystemHealthResponse(BaseModel):
    status: str
    timestamp: datetime
    database_connected: bool
    ai_model_loaded: bool
    version: str

class DocumentStats(BaseModel):
    total_documents: int
    total_size: str
    last_updated: datetime

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and AI models on startup"""
    try:
        await init_database()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")

# Health check endpoint
@app.get("/api/system/health", response_model=SystemHealthResponse)
async def health_check():
    """System health check endpoint"""
    db_connected = await test_connection()
    ai_loaded = classifier.model is not None
    
    return SystemHealthResponse(
        status="healthy" if db_connected and ai_loaded else "degraded",
        timestamp=datetime.now(),
        database_connected=db_connected,
        ai_model_loaded=ai_loaded,
        version="2.0.0"
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Persian Legal AI System is running!",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs"
    }

# AI Classification endpoint
@app.post("/api/ai/classify")
async def classify_text(request: TextClassificationRequest):
    """Classify Persian legal text"""
    try:
        result = await classifier.classify_async(request.text)
        
        response = {
            "text": request.text[:100] + "..." if len(request.text) > 100 else request.text,
            "classification": result,
            "timestamp": datetime.now().isoformat()
        }
        
        if request.include_confidence:
            response["confidence"] = max(result.values())
            response["predicted_class"] = max(result, key=result.get)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# Document stats endpoint
@app.get("/api/documents/stats", response_model=DocumentStats)
async def document_statistics():
    """Get document statistics"""
    # Mock data - replace with actual database queries
    return DocumentStats(
        total_documents=1250,
        total_size="45.7 MB",
        last_updated=datetime.now()
    )

# Document upload endpoint
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload legal document for processing"""
    try:
        # Validate file type
        allowed_types = ['text/plain', 'application/pdf', 'application/msword']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file.content_type} not supported"
            )
        
        # Read file content
        content = await file.read()
        
        # Process document (implement actual processing logic)
        result = {
            "filename": file.filename,
            "size": len(content),
            "content_type": file.content_type,
            "status": "uploaded",
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Search endpoint
@app.get("/api/documents/search")
async def search_documents(q: str, limit: int = 10):
    """Search legal documents"""
    # Mock search results - replace with actual FTS5 search
    results = [
        {
            "id": i,
            "title": f"نتیجه جستجو {i} برای '{q}'",
            "snippet": f"این یک نمونه متن حقوقی است که شامل کلیدواژه '{q}' می‌باشد...",
            "score": 0.95 - (i * 0.1),
            "date": datetime.now().isoformat()
        }
        for i in range(1, min(limit + 1, 6))
    ]
    
    return {
        "query": q,
        "total_results": len(results),
        "results": results
    }

# Training endpoint
@app.post("/api/training/start")
async def start_training(dataset_name: str):
    """Start model training session"""
    # Mock training start - implement actual training logic
    return {
        "message": "Training session started",
        "dataset": dataset_name,
        "session_id": f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "status": "started",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )