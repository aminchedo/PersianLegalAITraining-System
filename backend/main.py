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
from config.database import get_database, test_connection
from database.migrations import run_migrations, get_database_health
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

# Initialize database and Redis on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database, Redis, and AI models on startup"""
    try:
        migration_success = await run_migrations()
        if migration_success:
            print("✅ Database migrations completed successfully")
        else:
            print("❌ Database migration failed")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
    
    try:
        from config.redis_config import init_redis
        redis_ok = await init_redis()
        if redis_ok:
            print("✅ Redis initialized successfully")
        else:
            print("⚠️ Redis initialization failed - continuing without cache")
    except Exception as e:
        print(f"⚠️ Redis initialization failed: {e} - continuing without cache")

# Health check endpoint
@app.get("/api/system/health", response_model=SystemHealthResponse)
async def health_check():
    """System health check endpoint"""
    db_health = await get_database_health()
    db_connected = db_health.get("status") == "healthy"
    ai_loaded = classifier.model is not None
    
    return SystemHealthResponse(
        status="healthy" if db_connected and ai_loaded else "degraded",
        timestamp=datetime.now(),
        database_connected=db_connected,
        ai_model_loaded=ai_loaded,
        version="2.0.0"
    )

# Database stats endpoint
@app.get("/api/system/database/stats")
async def database_stats():
    """Get database statistics"""
    try:
        from database.migrations import get_database_stats
        stats = await get_database_stats()
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database stats: {str(e)}")

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

# AI Classification endpoint with Redis caching
@app.post("/api/ai/classify")
async def classify_text(request: TextClassificationRequest):
    """Classify Persian legal text with Redis caching"""
    try:
        import hashlib
        
        # Create cache key from text hash
        text_hash = hashlib.md5(request.text.encode('utf-8')).hexdigest()
        
        # Try to get from cache first
        try:
            from config.redis_config import cache
            cached_result = await cache.get_classification_result(text_hash)
            if cached_result:
                print(f"✅ Cache hit for classification: {text_hash[:8]}...")
                cached_result["cached"] = True
                return cached_result
        except Exception as cache_error:
            print(f"⚠️ Cache read error: {cache_error}")
        
        # If not in cache, perform classification
        result = await classifier.classify_async(request.text)
        
        response = {
            "text": request.text[:100] + "..." if len(request.text) > 100 else request.text,
            "classification": result,
            "timestamp": datetime.now().isoformat(),
            "cached": False
        }
        
        if request.include_confidence:
            response["confidence"] = max(result.values())
            response["predicted_class"] = max(result, key=result.get)
        
        # Cache the result
        try:
            from config.redis_config import cache
            await cache.cache_classification_result(text_hash, response, ttl=3600)
            print(f"✅ Cached classification result: {text_hash[:8]}...")
        except Exception as cache_error:
            print(f"⚠️ Cache write error: {cache_error}")
        
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

# Shutdown event to cleanup resources
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    try:
        from config.redis_config import close_redis
        await close_redis()
        print("✅ Redis connection closed")
    except Exception as e:
        print(f"⚠️ Error closing Redis: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )