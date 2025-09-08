"""
Simple Test API for Persian Legal AI System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Persian Legal AI Training System",
    description="سیستم آموزش هوش مصنوعی حقوقی فارسی",
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

# Pydantic models
class ClassificationRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    text: str
    category: str
    confidence: float
    method: str
    processing_time_ms: float

# Sample data
SAMPLE_DOCUMENTS = [
    {
        "id": 1,
        "title": "قرارداد خرید و فروش ملک",
        "content": "این قرارداد فی‌مابین آقای احمد محمدی به عنوان فروشنده و آقای رضا رضایی به عنوان خریدار منعقد گردید.",
        "category": "قرارداد",
        "date": "1402/09/08"
    },
    {
        "id": 2,
        "title": "دادنامه شماره ۱۴۰۲۱۲۳۴",
        "content": "دادگاه با بررسی اوراق پرونده و استماع اظهارات طرفین، حکم به محکومیت خوانده صادر می‌نماید.",
        "category": "دادنامه",
        "date": "1402/09/07"
    },
    {
        "id": 3,
        "title": "شکایت کیفری",
        "content": "اینجانب علی علوی از آقای حسن حسنی به اتهام کلاهبرداری شکایت دارم.",
        "category": "شکایت",
        "date": "1402/09/06"
    }
]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Persian Legal AI Training System API",
        "message_fa": "سیستم API آموزش هوش مصنوعی حقوقی فارسی",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/system/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Persian Legal AI Backend",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": 0
    }

@app.get("/api/models/status")
async def models_status():
    """Get models status"""
    return {
        "is_loaded": True,
        "persian_bert_available": True,
        "model_type": "simulated",
        "device": "cpu",
        "memory_usage_mb": 512,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/models/classify", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    """Classify Persian legal text"""
    start_time = datetime.now()
    
    # Simple keyword-based classification for testing
    text_lower = request.text.lower()
    
    if any(word in request.text for word in ["قرارداد", "خرید", "فروش", "اجاره"]):
        category = "قرارداد"
        confidence = 0.85
    elif any(word in request.text for word in ["دادنامه", "حکم", "دادگاه", "قاضی"]):
        category = "دادنامه"
        confidence = 0.90
    elif any(word in request.text for word in ["شکایت", "شاکی", "متهم"]):
        category = "شکایت"
        confidence = 0.88
    elif any(word in request.text for word in ["لایحه", "دفاعیه"]):
        category = "لایحه"
        confidence = 0.82
    else:
        category = "سایر"
        confidence = 0.60
    
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return ClassificationResponse(
        text=request.text[:100] + "..." if len(request.text) > 100 else request.text,
        category=category,
        confidence=confidence,
        method="keyword_based",
        processing_time_ms=processing_time
    )

@app.get("/api/documents")
async def get_documents(category: Optional[str] = None):
    """Get sample documents"""
    if category:
        filtered = [doc for doc in SAMPLE_DOCUMENTS if doc["category"] == category]
        return {"documents": filtered, "total": len(filtered)}
    return {"documents": SAMPLE_DOCUMENTS, "total": len(SAMPLE_DOCUMENTS)}

@app.get("/api/documents/stats")
async def get_document_stats():
    """Get document statistics"""
    categories = {}
    for doc in SAMPLE_DOCUMENTS:
        cat = doc["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        "total_documents": len(SAMPLE_DOCUMENTS),
        "categories": categories,
        "last_updated": datetime.now().isoformat()
    }

@app.post("/api/training/start")
async def start_training(epochs: int = 3, batch_size: int = 8):
    """Start training simulation"""
    return {
        "status": "success",
        "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 0.0002
        },
        "message": "Training started (simulated)"
    }

@app.get("/api/system/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "cpu_percent": 25.5,
        "memory_percent": 45.2,
        "disk_percent": 60.8,
        "active_connections": 2,
        "requests_per_minute": 15,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Persian Legal AI Test API on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)