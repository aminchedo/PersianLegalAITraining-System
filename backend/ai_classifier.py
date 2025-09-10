import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Optional
import asyncio
from functools import lru_cache

# Try to import Persian NLP tools
try:
    import hazm
    HAZM_AVAILABLE = True
except ImportError:
    HAZM_AVAILABLE = False
    print("⚠️ Hazm not available, using fallback text processing")

class PersianBERTClassifier:
    """Enhanced Persian BERT classifier with caching and error handling"""
    
    def __init__(self, model_name: str = "HooshvareLab/bert-fa-base-uncased"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        
        # Initialize Persian NLP tools if available
        if HAZM_AVAILABLE:
            self.normalizer = hazm.Normalizer()
            self.word_tokenizer = hazm.WordTokenizer()
        else:
            self.normalizer = None
            self.word_tokenizer = None
            
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the BERT model and tokenizer"""
        try:
            print(f"🤖 Loading Persian BERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=5  # Adjust based on your classification needs
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Persian BERT model loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load Persian BERT model: {e}")
            print("📋 Falling back to keyword-based classification")
            # Fallback to keyword-based classification
            self.model = None
            self.tokenizer = None
    
    @lru_cache(maxsize=1000)
    def preprocess_text(self, text: str) -> str:
        """Preprocess Persian text with caching"""
        if HAZM_AVAILABLE and self.normalizer:
            # Normalize Persian text using Hazm
            normalized = self.normalizer.normalize(text)
        else:
            # Basic text normalization
            normalized = text.strip()
            # Replace common Persian character variants
            normalized = normalized.replace('ي', 'ی').replace('ك', 'ک')
            
        return normalized
    
    async def classify_async(self, text: str) -> Dict[str, float]:
        """Asynchronous text classification"""
        if self.model is None or self.tokenizer is None:
            return await self._fallback_classification(text)
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
            
            # Convert to dictionary
            labels = ['legal', 'contract', 'regulation', 'court_decision', 'other']
            result = {
                labels[i]: float(probabilities[0][i]) 
                for i in range(len(labels))
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return await self._fallback_classification(text)
    
    async def _fallback_classification(self, text: str) -> Dict[str, float]:
        """Fallback keyword-based classification"""
        text_lower = text.lower()
        keywords = {
            'legal': ['قانون', 'حقوقی', 'قضایی', 'دادگاه', 'حکم', 'رأی'],
            'contract': ['قرارداد', 'توافق', 'عقد', 'پیمان', 'تعهد'],
            'regulation': ['آیین‌نامه', 'مقررات', 'دستورالعمل', 'بخشنامه'],
            'court_decision': ['رأی', 'حکم', 'قضاوت', 'دادگاه', 'محکمه'],
            'other': ['سایر', 'متفرقه']
        }
        
        scores = {}
        total_score = 0
        
        for category, words in keywords.items():
            score = 0
            for word in words:
                if word in text_lower:
                    score += 1
            scores[category] = score
            total_score += score
        
        # Normalize scores
        if total_score > 0:
            scores = {k: v/total_score for k, v in scores.items()}
        else:
            # Default equal distribution
            scores = {k: 0.2 for k in keywords.keys()}
        
        return scores

    def classify_sync(self, text: str) -> Dict[str, float]:
        """Synchronous wrapper for classification"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.classify_async(text))
        except RuntimeError:
            # If no event loop is running, create a new one
            return asyncio.run(self.classify_async(text))

# Global classifier instance
classifier = PersianBERTClassifier()

# Convenience functions
async def classify_text_async(text: str) -> Dict[str, float]:
    """Async function to classify Persian legal text"""
    return await classifier.classify_async(text)

def classify_text(text: str) -> Dict[str, float]:
    """Sync function to classify Persian legal text"""
    return classifier.classify_sync(text)