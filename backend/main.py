from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import asyncio
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from database import DatabaseManager
from scraper import LegalScraper
from ai_classifier import PersianBERTClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
db = None
scraper = None
classifier = None

# Scraping status tracking
scraping_status = {
    "is_running": False,
    "current_site": None,
    "documents_scraped": 0,
    "errors": [],
    "started_at": None,
    "estimated_completion": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global db, scraper, classifier
    
    # Initialize components
    db = DatabaseManager()
    scraper = LegalScraper()
    classifier = PersianBERTClassifier()
    
    logger.info("Iranian Legal Archive API started successfully")
    yield
    
    # Cleanup
    if scraper:
        await scraper.close()
    logger.info("Iranian Legal Archive API shut down")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Iranian Legal Archive API",
    description="Persian Legal Document Archive System with AI Classification",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173", 
        "https://your-domain.github.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Iranian Legal Archive API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/api/documents/search")
async def search_documents(
    query: str = Query(..., description="Search query in Persian"),
    category: Optional[str] = Query(None, description="Document category filter"),
    limit: int = Query(10, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip")
):
    """Full-text search for Persian legal documents"""
    try:
        results = await db.search_documents(query, category, limit, offset)
        total = await db.count_search_results(query, category)
        
        return {
            "documents": results, 
            "total": total,
            "query": query,
            "category": category,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

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

@app.get("/api/documents/category/{category}")
async def get_documents_by_category(
    category: str,
    limit: int = Query(10, ge=1, le=100)
):
    """Get documents by category"""
    try:
        documents = await db.get_documents_by_category(category, limit)
        return documents
    except Exception as e:
        logger.error(f"Get documents by category error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve documents: {str(e)}")

@app.post("/api/documents/{document_id}/classify")
async def classify_document(document_id: int):
    """Classify a document using Persian BERT"""
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
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

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