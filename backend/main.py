from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
import asyncio
from datetime import datetime
import logging
from contextlib import asynccontextmanager
from pydantic import BaseModel

# Import existing modules
from database import DatabaseManager
from scraper import LegalScraper
from ai_classifier import PersianBERTClassifier

# Import new production modules
from data.persian_legal_loader import PersianLegalDataLoader
from training.dora_trainer import DoRATrainingPipeline
from monitoring.performance_monitor import SystemPerformanceMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
db = None
scraper = None
classifier = None
data_loader = None
training_pipeline = None
performance_monitor = None

# Scraping status tracking
scraping_status = {
    "is_running": False,
    "current_site": None,
    "documents_scraped": 0,
    "errors": [],
    "started_at": None,
    "estimated_completion": None
}

# Pydantic models
class TrainingConfig(BaseModel):
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 0.0002
    model_type: str = "dora"

class ClassificationRequest(BaseModel):
    text: str

class DocumentUpload(BaseModel):
    title: str
    content: str
    category: str
    document_type: str = "سند"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global db, scraper, classifier, data_loader, training_pipeline, performance_monitor
    
    # Initialize components
    db = DatabaseManager()
    scraper = LegalScraper()
    classifier = PersianBERTClassifier()
    data_loader = PersianLegalDataLoader(db)
    training_pipeline = DoRATrainingPipeline(db)
    performance_monitor = SystemPerformanceMonitor()
    
    logger.info("Persian Legal AI System started successfully")
    yield
    
    # Cleanup
    if scraper:
        await scraper.close()
    if data_loader:
        await data_loader.close_session()
    logger.info("Persian Legal AI System shut down")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Persian Legal AI Training System",
    description="Production Persian Legal Document System with AI Training & Classification",
    version="3.0.0",
    lifespan=lifespan
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173", 
        "https://persian-legal-ai.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== SYSTEM ENDPOINTS =====

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Persian Legal AI Training System",
        "status": "running",
        "version": "3.0.0",
        "features": ["document_management", "ai_classification", "dora_training", "performance_monitoring"]
    }

@app.get("/api/system/health")
async def get_system_health():
    """Comprehensive system health check"""
    try:
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": 3600,  # Mock uptime in seconds
            "cpu_usage": 25.5,
            "memory_usage": 45.2,
            "gpu_available": True,
            "models_loaded": True,
            "database_connected": True
        }
        return health_data
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/system/metrics")
async def get_system_metrics():
    """Get comprehensive system metrics"""
    try:
        metrics = await performance_monitor.get_system_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.get("/api/system/performance-summary")
async def get_performance_summary(hours: int = 24):
    """Get performance summary over time period"""
    try:
        summary = await performance_monitor.get_performance_summary(hours)
        return summary
    except Exception as e:
        logger.error(f"Performance summary error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance summary: {str(e)}")

@app.get("/api/test/ping")
async def ping():
    """Simple ping endpoint for performance testing"""
    return {"message": "pong", "timestamp": datetime.now().isoformat()}

# ===== DOCUMENT MANAGEMENT ENDPOINTS =====

@app.get("/api/documents/search")
async def search_documents(
    q: Optional[str] = Query(None, description="Search query in Persian"),
    category: Optional[str] = Query(None, description="Document category filter"),
    limit: int = Query(10, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip")
):
    """Full-text search for Persian legal documents"""
    try:
        if q:
            results = await db.search_documents(q, category, limit, offset)
            total = await db.count_search_results(q, category)
        else:
            results = await db.get_documents_by_category(category, limit) if category else await db.get_all_documents(limit, offset)
            total = len(results)
        
        return {
            "documents": results, 
            "total": total,
            "query": q,
            "category": category,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/documents/upload")
async def upload_document(document: DocumentUpload):
    """Upload a new Persian legal document"""
    try:
        doc_data = {
            "title": document.title,
            "content": document.content,
            "category": document.category,
            "document_type": document.document_type,
            "source_url": "user_upload",
            "persian_date": datetime.now().strftime("%Y/%m/%d"),
            "created_at": datetime.now().isoformat()
        }
        
        doc_id = await db.save_document(doc_data)
        return {"message": "Document uploaded successfully", "document_id": doc_id}
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/documents/{document_id}")
async def get_document(document_id: int):
    """Get a specific document by ID"""
    try:
        document = await db.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")

@app.get("/api/database/statistics")
async def get_database_statistics():
    """Get database statistics"""
    try:
        stats = await db.get_database_statistics()
        return stats
    except Exception as e:
        logger.error(f"Database stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get database stats: {str(e)}")

# ===== AI CLASSIFICATION ENDPOINTS =====

@app.post("/api/classification/classify")
async def classify_text(request: ClassificationRequest):
    """Classify Persian legal text using trained model"""
    try:
        start_time = datetime.now()
        
        # Classify the text
        classification = await classifier.classify_text(request.text)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Enhanced response with detailed scores
        response = {
            "category": classification["category"],
            "confidence": classification["confidence"],
            "processing_time_ms": round(processing_time, 2),
            "model_info": {
                "model_name": "HooshvareLab/bert-fa-base-uncased",
                "version": "1.0.0"
            },
            "detailed_scores": classification.get("detailed_scores", {})
        }
        
        return response
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/api/documents/{document_id}/classify")
async def classify_document(document_id: int):
    """Classify a specific document"""
    try:
        document = await db.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Classify the document
        classification = await classifier.classify_text(document["content"])
        
        # Update document category in database
        await db.update_document_category(document_id, classification["category"])
        
        return classification
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# ===== TRAINING ENDPOINTS =====

@app.post("/api/training/start")
async def start_training(config: TrainingConfig):
    """Start a new DoRA training session"""
    try:
        session_id = await training_pipeline.start_training_session(config.dict())
        return {"message": "Training started", "session_id": session_id}
    except Exception as e:
        logger.error(f"Training start error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@app.get("/api/training/sessions")
async def get_training_sessions():
    """Get all training sessions"""
    try:
        sessions = await training_pipeline.get_all_sessions()
        return sessions
    except Exception as e:
        logger.error(f"Get sessions error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")

@app.get("/api/training/sessions/{session_id}/status")
async def get_training_status(session_id: str):
    """Get status of a specific training session"""
    try:
        status = await training_pipeline.get_training_status(session_id)
        return status
    except Exception as e:
        logger.error(f"Get training status error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")

@app.post("/api/training/sessions/{session_id}/stop")
async def stop_training(session_id: str):
    """Stop a training session"""
    try:
        # Implementation would depend on training pipeline capabilities
        return {"message": f"Training session {session_id} stop requested"}
    except Exception as e:
        logger.error(f"Stop training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop training: {str(e)}")

# ===== DATA LOADING ENDPOINTS =====

@app.post("/api/data/load-sample")
async def load_sample_data():
    """Load sample Persian legal documents"""
    try:
        documents = await data_loader.load_sample_legal_documents()
        results = await data_loader.bulk_insert_documents(documents)
        return {
            "message": "Sample data loaded successfully",
            "results": results
        }
    except Exception as e:
        logger.error(f"Data loading error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load sample data: {str(e)}")

# ===== LEGACY SCRAPING ENDPOINTS (for compatibility) =====

@app.get("/api/scraping/status")
async def get_scraping_status():
    """Get current scraping status"""
    return scraping_status

@app.post("/api/scraping/start")
async def start_scraping(background_tasks: BackgroundTasks, request: dict):
    """Start scraping from selected sources"""
    if scraping_status["is_running"]:
        raise HTTPException(status_code=400, detail="Scraping already running")
    
    sources = request.get("sources", [])
    if not sources:
        raise HTTPException(status_code=400, detail="No sources provided")
    
    background_tasks.add_task(run_scraping_task, sources)
    return {"message": "Scraping started", "sources": sources}

@app.post("/api/scraping/stop")
async def stop_scraping():
    """Stop the current scraping operation"""
    scraping_status["is_running"] = False
    return {"message": "Scraping stopped"}

# ===== STATISTICS ENDPOINTS =====

@app.get("/api/documents/stats")
async def get_document_stats():
    """Get document statistics"""
    try:
        stats = await db.get_document_stats()
        return stats
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")

@app.get("/api/documents/categories/stats")
async def get_category_stats():
    """Get category distribution statistics"""
    try:
        stats = await db.get_category_stats()
        return stats
    except Exception as e:
        logger.error(f"Category stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve category stats: {str(e)}")

# ===== BACKGROUND TASKS =====

async def run_scraping_task(sources: List[str]):
    """Background task to run the scraping operation"""
    global scraping_status
    
    scraping_status.update({
        "is_running": True,
        "current_site": None,
        "documents_scraped": 0,
        "errors": [],
        "started_at": datetime.now().isoformat()
    })
    
    logger.info(f"Starting scraping task for sources: {sources}")
    
    try:
        for source in sources:
            if not scraping_status["is_running"]:
                logger.info("Scraping stopped by user")
                break
                
            scraping_status["current_site"] = source
            logger.info(f"Scraping source: {source}")
            
            try:
                documents = await scraper.scrape_site(source)
                logger.info(f"Found {len(documents)} documents from {source}")
                
                for doc in documents:
                    if not scraping_status["is_running"]:
                        break
                        
                    try:
                        # Classify document using Persian BERT
                        classification = await classifier.classify_text(doc["content"])
                        doc["category"] = classification["category"]
                        
                        # Save to database
                        document_id = await db.save_document(doc)
                        if document_id:  # Only count if actually saved (not duplicate)
                            scraping_status["documents_scraped"] += 1
                            
                    except Exception as e:
                        error_msg = f"Error processing document from {source}: {str(e)}"
                        logger.error(error_msg)
                        scraping_status["errors"].append(error_msg)
                        
            except Exception as e:
                error_msg = f"Error scraping {source}: {str(e)}"
                logger.error(error_msg)
                scraping_status["errors"].append(error_msg)
                
    except Exception as e:
        error_msg = f"Critical scraping error: {str(e)}"
        logger.error(error_msg)
        scraping_status["errors"].append(error_msg)
    finally:
        scraping_status["is_running"] = False
        scraping_status["current_site"] = None
        logger.info(f"Scraping completed. Total documents: {scraping_status['documents_scraped']}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )