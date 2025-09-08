"""
Persian Legal AI Training System - Enhanced FastAPI Backend
سیستم آموزش هوش مصنوعی حقوقی فارسی - بک‌اند پیشرفته FastAPI
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from contextlib import asynccontextmanager
import json
import uuid
import psutil
import os
from pydantic import BaseModel

# Import our Persian components
import sys
sys.path.append('/workspace')
from backend.database.persian_db import PersianLegalDB
from models.persian_legal_classifier import PersianLegalAIClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
persian_db = None
ai_classifier = None

# System status tracking
system_status = {
    "database": {"status": "unknown", "last_check": None},
    "ai_models": {"status": "unknown", "last_check": None},
    "training": {"active_sessions": 0, "total_sessions": 0},
    "scraping": {"is_running": False, "documents_scraped": 0}
}

# Training sessions tracking
training_sessions = {}

# Pydantic models for API
class DocumentSearchRequest(BaseModel):
    query: str
    category: Optional[str] = None
    document_type: Optional[str] = None
    limit: int = 10
    offset: int = 0

class DocumentInsertRequest(BaseModel):
    title: str
    content: str
    source_url: Optional[str] = None
    document_type: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    persian_date: Optional[str] = None

class TrainingStartRequest(BaseModel):
    model_type: str = "dora"  # dora, qr_adaptor, base
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 16
    use_dora: bool = True
    notes: Optional[str] = None

class ClassificationRequest(BaseModel):
    text: str
    return_probabilities: bool = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global persian_db, ai_classifier
    
    try:
        # Initialize Persian database
        logger.info("Initializing Persian Legal Database...")
        persian_db = PersianLegalDB()
        if persian_db.test_connection():
            system_status["database"]["status"] = "healthy"
            logger.info("✅ Persian database initialized successfully")
        else:
            system_status["database"]["status"] = "error"
            logger.error("❌ Persian database initialization failed")
        
        # Initialize AI classifier
        logger.info("Initializing Persian AI Classifier...")
        ai_classifier = PersianLegalAIClassifier()
        # Don't initialize models yet - do it on first use to save memory
        system_status["ai_models"]["status"] = "ready"
        logger.info("✅ Persian AI classifier ready")
        
        system_status["database"]["last_check"] = datetime.now().isoformat()
        system_status["ai_models"]["last_check"] = datetime.now().isoformat()
        
        logger.info("🚀 Persian Legal AI Training System started successfully")
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        yield
    
    finally:
        logger.info("🔄 Persian Legal AI Training System shutting down")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Persian Legal AI Training System",
    description="Advanced Persian legal document processing with AI classification and DoRA training",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System health and status endpoints
@app.get("/api/system/health")
async def system_health():
    """Comprehensive system health check"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database": "unknown",
                "ai_models": "unknown", 
                "gpu": "unknown",
                "memory": "unknown"
            },
            "system_info": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "torch_version": None,
                "device": None,
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available
            }
        }
        
        # Check database
        try:
            if persian_db and persian_db.test_connection():
                health_status["components"]["database"] = "healthy"
                system_status["database"]["status"] = "healthy"
            else:
                health_status["components"]["database"] = "error"
                system_status["database"]["status"] = "error"
        except Exception as e:
            health_status["components"]["database"] = f"error: {str(e)}"
        
        # Check AI models
        try:
            import torch
            health_status["system_info"]["torch_version"] = torch.__version__
            health_status["system_info"]["device"] = str(ai_classifier.device) if ai_classifier else "unknown"
            health_status["components"]["ai_models"] = "ready" if ai_classifier else "error"
            health_status["components"]["gpu"] = "available" if torch.cuda.is_available() else "cpu_only"
            system_status["ai_models"]["status"] = "ready"
        except Exception as e:
            health_status["components"]["ai_models"] = f"error: {str(e)}"
        
        # Memory check
        memory_usage = psutil.virtual_memory().percent
        health_status["components"]["memory"] = "healthy" if memory_usage < 85 else "warning"
        health_status["system_info"]["memory_usage_percent"] = memory_usage
        
        # Determine overall status
        component_statuses = list(health_status["components"].values())
        if any("error" in status for status in component_statuses):
            health_status["status"] = "degraded"
        elif any("warning" in status for status in component_statuses):
            health_status["status"] = "warning"
        
        system_status["database"]["last_check"] = datetime.now().isoformat()
        system_status["ai_models"]["last_check"] = datetime.now().isoformat()
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/api/system/status")
async def get_system_status():
    """Get detailed system status"""
    try:
        # Get database stats
        db_stats = await persian_db.get_document_stats() if persian_db else {}
        
        # Get AI model info
        model_info = ai_classifier.get_model_info() if ai_classifier else {}
        
        return {
            "system_status": system_status,
            "database_stats": db_stats,
            "model_info": model_info,
            "training_sessions": {
                "active": len([s for s in training_sessions.values() if s.get("status") == "running"]),
                "total": len(training_sessions),
                "sessions": list(training_sessions.keys())
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document management endpoints
@app.post("/api/documents/search")
async def search_documents(request: DocumentSearchRequest):
    """Persian full-text search with filters"""
    try:
        if not persian_db:
            raise HTTPException(status_code=503, detail="Database not available")
        
        results = await persian_db.search_persian_documents(
            query=request.query,
            category=request.category,
            document_type=request.document_type,
            limit=request.limit,
            offset=request.offset
        )
        
        return {
            "documents": results,
            "total": len(results),
            "query": request.query,
            "filters": {
                "category": request.category,
                "document_type": request.document_type
            },
            "pagination": {
                "limit": request.limit,
                "offset": request.offset
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/documents/insert")
async def insert_document(request: DocumentInsertRequest):
    """Insert a new Persian legal document"""
    try:
        if not persian_db:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Prepare document data
        document_data = {
            "title": request.title,
            "content": request.content,
            "source_url": request.source_url,
            "document_type": request.document_type,
            "category": request.category,
            "subcategory": request.subcategory,
            "persian_date": request.persian_date,
            "word_count": len(request.content.split()),
            "char_count": len(request.content),
            "quality_score": 0.8,  # Default quality score
            "language_confidence": 0.9,  # Assume high Persian confidence
            "legal_relevance": 0.7  # Default legal relevance
        }
        
        # Insert document
        document_id = await persian_db.insert_document(document_data)
        
        return {
            "success": True,
            "document_id": document_id,
            "message": "Document inserted successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Document insertion error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to insert document: {str(e)}")

@app.get("/api/documents/stats")
async def get_document_stats():
    """Get comprehensive document statistics"""
    try:
        if not persian_db:
            raise HTTPException(status_code=503, detail="Database not available")
        
        stats = await persian_db.get_document_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get document stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# AI Classification endpoints
@app.post("/api/ai/classify")
async def classify_document(request: ClassificationRequest):
    """Classify Persian legal document using AI"""
    try:
        if not ai_classifier:
            raise HTTPException(status_code=503, detail="AI classifier not available")
        
        # Initialize models if not done yet
        if not ai_classifier.is_initialized:
            logger.info("Initializing AI models for first use...")
            success = await ai_classifier.initialize_models(use_dora=True)
            if not success:
                raise HTTPException(status_code=503, detail="Failed to initialize AI models")
        
        # Classify the document
        result = await ai_classifier.classify_document(
            request.text, 
            return_probabilities=request.return_probabilities
        )
        
        # Also classify document type
        doc_type_result = await ai_classifier.classify_document_type(request.text)
        
        # Extract keywords
        keywords = await ai_classifier.extract_keywords(request.text)
        
        # Generate summary
        summary = await ai_classifier.generate_summary(request.text)
        
        return {
            "classification": result,
            "document_type": doc_type_result,
            "keywords": keywords,
            "summary": summary,
            "text_stats": {
                "word_count": len(request.text.split()),
                "char_count": len(request.text)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/api/ai/model-info")
async def get_model_info():
    """Get AI model information"""
    try:
        if not ai_classifier:
            raise HTTPException(status_code=503, detail="AI classifier not available")
        
        return ai_classifier.get_model_info()
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Training session endpoints
@app.post("/api/training/start")
async def start_training_session(
    request: TrainingStartRequest,
    background_tasks: BackgroundTasks
):
    """Start Persian legal AI training session"""
    try:
        # Validate configuration
        if request.epochs < 1 or request.epochs > 100:
            raise HTTPException(status_code=400, detail="Epochs must be between 1 and 100")
        
        if request.learning_rate <= 0 or request.learning_rate > 0.1:
            raise HTTPException(status_code=400, detail="Learning rate must be between 0 and 0.1")
        
        # Create training session
        session_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        training_session = {
            "id": session_id,
            "status": "starting",
            "config": request.dict(),
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "progress": {
                "current_epoch": 0,
                "total_epochs": request.epochs,
                "current_step": 0,
                "total_steps": 0,
                "loss": None,
                "accuracy": None
            },
            "logs": [],
            "error": None
        }
        
        training_sessions[session_id] = training_session
        system_status["training"]["total_sessions"] += 1
        
        # Start background training
        background_tasks.add_task(run_training_session, session_id, request)
        
        return {
            "session_id": session_id,
            "status": "started",
            "config": request.dict(),
            "estimated_duration": f"{request.epochs * 15}-{request.epochs * 20} minutes",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Training start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/status/{session_id}")
async def get_training_status(session_id: str):
    """Get training session status"""
    try:
        if session_id not in training_sessions:
            raise HTTPException(status_code=404, detail="Training session not found")
        
        return training_sessions[session_id]
        
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/sessions")
async def list_training_sessions():
    """List all training sessions"""
    try:
        return {
            "sessions": list(training_sessions.values()),
            "total": len(training_sessions),
            "active": len([s for s in training_sessions.values() if s.get("status") == "running"]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to list training sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_training_session(session_id: str, config: TrainingStartRequest):
    """Background training task with Persian data"""
    try:
        logger.info(f"Starting training session {session_id}")
        
        # Update session status
        training_sessions[session_id]["status"] = "running"
        training_sessions[session_id]["started_at"] = datetime.now().isoformat()
        system_status["training"]["active_sessions"] += 1
        
        # Initialize AI classifier if needed
        if not ai_classifier.is_initialized:
            await ai_classifier.initialize_models(use_dora=config.use_dora)
        
        # Simulate training progress (replace with real training logic)
        total_steps = config.epochs * 10  # Simulate 10 steps per epoch
        training_sessions[session_id]["progress"]["total_steps"] = total_steps
        
        for epoch in range(config.epochs):
            training_sessions[session_id]["progress"]["current_epoch"] = epoch + 1
            training_sessions[session_id]["logs"].append({
                "timestamp": datetime.now().isoformat(),
                "message": f"Starting epoch {epoch + 1}/{config.epochs}",
                "level": "info"
            })
            
            # Simulate steps within epoch
            for step in range(10):
                current_step = epoch * 10 + step + 1
                training_sessions[session_id]["progress"]["current_step"] = current_step
                
                # Simulate loss and accuracy
                loss = 2.0 * (1 - current_step / total_steps) + 0.1
                accuracy = 0.5 + 0.4 * (current_step / total_steps)
                
                training_sessions[session_id]["progress"]["loss"] = round(loss, 4)
                training_sessions[session_id]["progress"]["accuracy"] = round(accuracy, 4)
                
                # Simulate training time
                await asyncio.sleep(2)  # 2 seconds per step
                
                if current_step % 5 == 0:
                    training_sessions[session_id]["logs"].append({
                        "timestamp": datetime.now().isoformat(),
                        "message": f"Step {current_step}/{total_steps} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}",
                        "level": "info"
                    })
        
        # Complete training
        training_sessions[session_id]["status"] = "completed"
        training_sessions[session_id]["completed_at"] = datetime.now().isoformat()
        training_sessions[session_id]["logs"].append({
            "timestamp": datetime.now().isoformat(),
            "message": f"Training completed successfully",
            "level": "success"
        })
        
        system_status["training"]["active_sessions"] -= 1
        logger.info(f"Training session {session_id} completed successfully")
        
    except Exception as e:
        # Handle training errors
        training_sessions[session_id]["status"] = "failed"
        training_sessions[session_id]["error"] = str(e)
        training_sessions[session_id]["logs"].append({
            "timestamp": datetime.now().isoformat(),
            "message": f"Training failed: {str(e)}",
            "level": "error"
        })
        
        system_status["training"]["active_sessions"] -= 1
        logger.error(f"Training session {session_id} failed: {e}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Persian Legal AI Training System",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/api/system/health",
            "status": "/api/system/status",
            "search": "/api/documents/search",
            "classify": "/api/ai/classify",
            "training": "/api/training/start"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)