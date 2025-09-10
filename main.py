import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Persian Legal AI API",
    description="Persian Legal AI Backend System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Persian Legal AI Backend is running!",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/api/system/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "operational",
            "ai_model": "ready",
            "api": "operational"
        }
    }

@app.get("/api/documents/stats")
async def document_stats():
    return {
        "total_documents": 0,
        "processed_documents": 0,
        "pending_documents": 0,
        "last_updated": datetime.now().isoformat()
    }

@app.post("/api/ai/classify")
async def classify_text(request: dict):
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Mock classification response
    return {
        "text": text,
        "classification": "legal_document",
        "confidence": 0.95,
        "categories": ["contract", "legal"],
        "language": "persian"
    }

@app.get("/api/system/info")
async def system_info():
    return {
        "system": "Persian Legal AI",
        "version": "1.0.0",
        "python_version": sys.version,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "database": "SQLite",
        "features": [
            "Document Classification",
            "Persian Text Processing",
            "Legal Document Analysis",
            "Web Scraping",
            "Training Management"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)