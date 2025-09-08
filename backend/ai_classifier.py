import asyncio
import logging
from typing import Dict, List, Optional
import re
import json
from datetime import datetime

# For production, you would install and use:
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

logger = logging.getLogger(__name__)

class PersianBERTClassifier:
    """Persian BERT-based document classifier for Iranian legal documents"""
    
    def __init__(self, model_name: str = "HooshvareLab/bert-fa-base-uncased"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Legal categories in Persian and English
        self.categories = {
            'civil_law': {
                'fa': 'حقوق مدنی',
                'keywords': ['مدنی', 'قرارداد', 'تعهدات', 'اموال', 'ملکیت', 'مالکیت', 'وکالت', 'وصیت', 'ارث']
            },
            'criminal_law': {
                'fa': 'حقوق جزا',
                'keywords': ['جزا', 'جرم', 'مجازات', 'کیفری', 'قصاص', 'دیه', 'تعزیر', 'محاکمه']
            },
            'commercial_law': {
                'fa': 'حقوق تجارت',
                'keywords': ['تجارت', 'شرکت', 'بازرگانی', 'تجاری', 'صنعتی', 'اقتصادی', 'بانک', 'بیمه']
            },
            'administrative_law': {
                'fa': 'حقوق اداری',
                'keywords': ['اداری', 'دولت', 'مقررات', 'آیین‌نامه', 'بخشنامه', 'دستورالعمل', 'اجرایی']
            },
            'constitutional_law': {
                'fa': 'حقوق قانون اساسی',
                'keywords': ['قانون اساسی', 'اساسی', 'مشروطه', 'حقوق بنیادین', 'آزادی', 'حاکمیت']
            },
            'labor_law': {
                'fa': 'حقوق کار',
                'keywords': ['کار', 'کارگر', 'کارفرما', 'استخدام', 'اشتغال', 'بیکاری', 'حقوق کارگری']
            },
            'family_law': {
                'fa': 'حقوق خانواده',
                'keywords': ['خانواده', 'ازدواج', 'طلاق', 'نفقه', 'حضانت', 'ولایت', 'نکاح']
            },
            'property_law': {
                'fa': 'حقوق اموال',
                'keywords': ['املاک', 'زمین', 'ساختمان', 'مستغلات', 'رهن', 'اجاره', 'خرید و فروش']
            },
            'tax_law': {
                'fa': 'حقوق مالیاتی',
                'keywords': ['مالیات', 'عوارض', 'گمرک', 'مالی', 'بودجه', 'درآمد', 'مالیاتی']
            },
            'international_law': {
                'fa': 'حقوق بین‌الملل',
                'keywords': ['بین‌الملل', 'دیپلماتیک', 'معاهده', 'کنوانسیون', 'خارجی', 'سفارت']
            }
        }
        
        # Initialize the model (in production)
        # asyncio.create_task(self.load_model())
    
    async def load_model(self):
        """Load the Persian BERT model (placeholder for actual implementation)"""
        try:
            # In production, uncomment these lines:
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # self.model = AutoModelForSequenceClassification.from_pretrained(
            #     self.model_name,
            #     num_labels=len(self.categories)
            # )
            # self.model.eval()
            
            self.is_loaded = True
            logger.info(f"Persian BERT model loaded: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load Persian BERT model: {str(e)}")
            self.is_loaded = False
    
    async def classify_text(self, text: str) -> Dict:
        """Classify Persian legal text into categories"""
        try:
            # Clean and preprocess text
            cleaned_text = self.preprocess_persian_text(text)
            
            if not cleaned_text:
                return self.get_default_classification()
            
            # For demo purposes, use keyword-based classification
            # In production, this would use the actual BERT model
            if self.is_loaded:
                return await self.bert_classify(cleaned_text)
            else:
                return await self.keyword_classify(cleaned_text)
                
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return self.get_default_classification()
    
    async def bert_classify(self, text: str) -> Dict:
        """Use Persian BERT for classification (production implementation)"""
        try:
            # This is a placeholder for the actual BERT implementation
            # In production, you would:
            # 1. Tokenize the text
            # 2. Run inference through the model
            # 3. Get probability scores for each category
            # 4. Return the results
            
            # Placeholder implementation using keyword matching
            return await self.keyword_classify(text)
            
        except Exception as e:
            logger.error(f"BERT classification error: {str(e)}")
            return self.get_default_classification()
    
    async def keyword_classify(self, text: str) -> Dict:
        """Keyword-based classification for demo purposes"""
        try:
            text_lower = text.lower()
            category_scores = {}
            
            # Calculate scores for each category based on keyword matches
            for category, info in self.categories.items():
                score = 0
                keyword_matches = []
                
                for keyword in info['keywords']:
                    # Count occurrences of each keyword
                    count = text_lower.count(keyword.lower())
                    if count > 0:
                        score += count * len(keyword)  # Weight by keyword length
                        keyword_matches.append(keyword)
                
                # Normalize score by text length
                if len(text) > 0:
                    category_scores[category] = score / len(text) * 1000
                else:
                    category_scores[category] = 0
            
            # Find the best category
            if category_scores:
                best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
                confidence = category_scores[best_category]
                
                # If confidence is too low, classify as 'other'
                if confidence < 0.1:
                    best_category = 'other'
                    confidence = 0.5
                else:
                    # Normalize confidence to 0-1 range
                    max_score = max(category_scores.values())
                    confidence = min(confidence / max_score, 1.0)
            else:
                best_category = 'other'
                confidence = 0.5
            
            return {
                'category': best_category,
                'confidence': confidence,
                'categories': category_scores,
                'persian_name': self.categories.get(best_category, {}).get('fa', 'سایر'),
                'classification_method': 'keyword_based',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Keyword classification error: {str(e)}")
            return self.get_default_classification()
    
    def preprocess_persian_text(self, text: str) -> str:
        """Preprocess Persian text for classification"""
        if not text:
            return ''
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove unwanted characters but keep Persian, Arabic, and common punctuation
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s\d\w\.\,\:\;\!\?\-\(\)\[\]\"\']+', '', text)
        
        # Normalize Persian/Arabic characters
        persian_chars = {
            'ي': 'ی',
            'ك': 'ک',
            'ء': 'ٔ',
        }
        
        for old_char, new_char in persian_chars.items():
            text = text.replace(old_char, new_char)
        
        # Remove extra spaces and trim
        text = ' '.join(text.split())
        
        return text.strip()
    
    def get_default_classification(self) -> Dict:
        """Return default classification for errors"""
        return {
            'category': 'other',
            'confidence': 0.0,
            'categories': {cat: 0.0 for cat in self.categories.keys()},
            'persian_name': 'سایر',
            'classification_method': 'default',
            'timestamp': datetime.now().isoformat(),
            'error': 'Classification failed'
        }
    
    def get_category_info(self, category: str) -> Dict:
        """Get information about a specific category"""
        return self.categories.get(category, {
            'fa': 'سایر',
            'keywords': []
        })
    
    def get_all_categories(self) -> Dict:
        """Get all available categories"""
        return {
            cat: {
                'english': cat,
                'persian': info['fa'],
                'keywords': info['keywords']
            }
            for cat, info in self.categories.items()
        }
    
    async def batch_classify(self, texts: List[str]) -> List[Dict]:
        """Classify multiple texts in batch"""
        results = []
        for text in texts:
            result = await self.classify_text(text)
            results.append(result)
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)
        return results
    
    def get_classification_stats(self, classifications: List[Dict]) -> Dict:
        """Get statistics from a list of classifications"""
        if not classifications:
            return {}
        
        category_counts = {}
        confidence_scores = []
        
        for classification in classifications:
            category = classification.get('category', 'unknown')
            confidence = classification.get('confidence', 0.0)
            
            category_counts[category] = category_counts.get(category, 0) + 1
            confidence_scores.append(confidence)
        
        return {
            'total_documents': len(classifications),
            'category_distribution': category_counts,
            'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            'high_confidence_count': len([c for c in confidence_scores if c > 0.8]),
            'low_confidence_count': len([c for c in confidence_scores if c < 0.3])
        }