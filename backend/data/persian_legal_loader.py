"""
Persian Legal Data Loader Module
Minimal implementation for system startup
"""

import json
import os
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PersianLegalDataLoader:
    """Minimal Persian Legal Data Loader for testing"""
    
    def __init__(self, data_dir: str = "data/samples"):
        self.data_dir = Path(data_dir)
        self.documents = []
        self.categories = [
            "قرارداد",  # Contract
            "دادنامه",  # Verdict
            "شکایت",   # Complaint
            "لایحه",    # Brief
            "استشهادیه" # Testimony
        ]
        logger.info(f"Persian Legal Data Loader initialized with data_dir: {self.data_dir}")
    
    def load_sample_data(self) -> Dict:
        """Load sample Persian legal documents"""
        sample_documents = [
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
        
        self.documents = sample_documents
        return {
            "status": "success",
            "documents_loaded": len(sample_documents),
            "categories": self.categories
        }
    
    def get_documents(self, category: Optional[str] = None) -> List[Dict]:
        """Get documents, optionally filtered by category"""
        if category:
            return [doc for doc in self.documents if doc.get("category") == category]
        return self.documents
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded documents"""
        stats = {
            "total_documents": len(self.documents),
            "categories": {}
        }
        
        for category in self.categories:
            count = len([doc for doc in self.documents if doc.get("category") == category])
            stats["categories"][category] = count
        
        return stats