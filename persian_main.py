import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Persian Legal AI - Persian Module",
    description="Persian-specific Legal AI Processing",
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
        "message": "Persian Legal AI - Persian Module is running!",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "language": "persian",
        "status": "operational"
    }

@app.post("/api/persian/analyze")
async def analyze_persian_text(request: dict):
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Persian text is required")
    
    # Basic Persian text analysis
    words = text.split()
    persian_pattern = re.compile(r'[\u0600-\u06FF]+')
    persian_words = [word for word in words if persian_pattern.search(word)]
    
    # Mock legal terms (in Persian)
    legal_terms = ["قانون", "ماده", "تبصره", "مقررات", "آئین‌نامه", "دستورالعمل"]
    detected_legal_terms = [term for term in legal_terms if term in text]
    
    return {
        "text": text,
        "language": "persian",
        "word_count": len(words),
        "persian_word_count": len(persian_words),
        "char_count": len(text),
        "analysis": {
            "sentiment": "neutral",
            "complexity": "medium",
            "legal_terms_detected": detected_legal_terms,
            "persian_ratio": len(persian_words) / len(words) if words else 0
        }
    }

@app.post("/api/persian/normalize")
async def normalize_persian_text(request: dict):
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Persian text is required")
    
    # Basic Persian text normalization
    # Replace Arabic/Urdu characters with Persian equivalents
    persian_map = {
        'ي': 'ی',
        'ك': 'ک',
        '٠': '۰',
        '١': '۱',
        '٢': '۲',
        '٣': '۳',
        '٤': '۴',
        '٥': '۵',
        '٦': '۶',
        '٧': '۷',
        '٨': '۸',
        '٩': '۹',
    }
    
    normalized_text = text
    for arabic, persian in persian_map.items():
        normalized_text = normalized_text.replace(arabic, persian)
    
    # Remove extra whitespaces
    normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
    
    return {
        "original_text": text,
        "normalized_text": normalized_text,
        "changes_made": len([1 for a, p in persian_map.items() if a in text]),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/persian/legal-terms")
async def get_legal_terms():
    """Get common Persian legal terms"""
    legal_terms = {
        "basic": ["قانون", "ماده", "تبصره", "فصل", "بخش"],
        "contract": ["قرارداد", "طرفین", "متعهد", "التزام", "حق"],
        "court": ["دادگاه", "قاضی", "دادرس", "حکم", "رای"],
        "civil": ["مدنی", "شخصی", "اموال", "ملکیت", "حقوق"],
        "criminal": ["جزایی", "جرم", "مجازات", "متهم", "شاکی"]
    }
    
    return {
        "legal_terms": legal_terms,
        "total_terms": sum(len(terms) for terms in legal_terms.values()),
        "categories": list(legal_terms.keys())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)