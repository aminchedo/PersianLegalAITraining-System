"""
Persian Legal AI Classifier with DoRA and QR-Adaptor Support
طبقه‌بند هوش مصنوعی حقوقی فارسی با پشتیبانی DoRA و QR-Adaptor
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class PersianLegalAIClassifier:
    """Advanced Persian Legal Document Classifier with DoRA support"""
    
    def __init__(self, model_name: str = "HooshvareLab/bert-base-parsbert-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_initialized = False
        
        # Persian legal categories mapping
        self.legal_categories = {
            0: {"fa": "حقوق مدنی", "en": "civil_law"},
            1: {"fa": "حقوق کیفری", "en": "criminal_law"},
            2: {"fa": "حقوق اداری", "en": "administrative_law"},
            3: {"fa": "حقوق تجاری", "en": "commercial_law"},
            4: {"fa": "حقوق کار", "en": "labor_law"},
            5: {"fa": "حقوق قضایی", "en": "judicial_law"},
        }
        
        # Document type mapping
        self.document_types = {
            0: {"fa": "قانون", "en": "law"},
            1: {"fa": "آیین‌نامه", "en": "regulation"},
            2: {"fa": "رأی", "en": "verdict"},
            3: {"fa": "مصوبه", "en": "resolution"},
            4: {"fa": "بخشنامه", "en": "circular"},
        }
        
        logger.info(f"Persian Legal AI Classifier initialized for device: {self.device}")
    
    async def initialize_models(self, use_dora: bool = True, force_cpu: bool = False) -> bool:
        """Initialize Persian BERT models with optional DoRA configuration"""
        try:
            if force_cpu:
                self.device = torch.device("cpu")
                logger.info("Forcing CPU usage for model initialization")
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load base model for classification
            logger.info(f"Loading base model from {self.model_name}")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.legal_categories),
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            if use_dora:
                # Configure DoRA (Weight-Decomposed Low-Rank Adaptation)
                logger.info("Configuring DoRA adaptation")
                dora_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=8,                    # Low-rank dimension
                    lora_alpha=16,          # Scaling parameter
                    lora_dropout=0.1,       # Dropout probability
                    target_modules=["query", "value", "key", "dense"],
                    use_dora=True,          # Enable DoRA decomposition
                    bias="none"
                )
                
                # Apply DoRA to model
                self.model = get_peft_model(base_model, dora_config)
                logger.info("✅ DoRA configuration applied successfully")
            else:
                self.model = base_model
                logger.info("✅ Using base model without DoRA")
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Test model with a simple forward pass
            test_input = self.tokenizer(
                "تست مدل فارسی",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                _ = self.model(**test_input)
            
            self.is_initialized = True
            logger.info(f"✅ Persian Legal AI initialized successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            # Fallback to CPU if CUDA fails
            if not force_cpu and self.device.type == "cuda":
                logger.info("Retrying initialization with CPU...")
                return await self.initialize_models(use_dora=use_dora, force_cpu=True)
            return False
    
    async def classify_document(self, text: str, return_probabilities: bool = True) -> Dict:
        """Classify Persian legal document with confidence scores"""
        if not self.is_initialized:
            logger.warning("Model not initialized, attempting to initialize...")
            if not await self.initialize_models():
                return self._error_response("Model initialization failed")
        
        try:
            # Preprocess Persian text
            cleaned_text = self._preprocess_persian_text(text)
            
            # Tokenize input
            inputs = self.tokenizer(
                cleaned_text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Prepare result
            result = {
                "category_fa": self.legal_categories[predicted_class]["fa"],
                "category_en": self.legal_categories[predicted_class]["en"],
                "confidence": round(confidence, 4),
                "predicted_class": predicted_class,
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "name": self.model_name,
                    "device": str(self.device),
                    "dora_enabled": hasattr(self.model, 'peft_config')
                }
            }
            
            if return_probabilities:
                result["all_probabilities"] = {
                    self.legal_categories[i]["fa"]: round(prob.item(), 4) 
                    for i, prob in enumerate(probabilities[0])
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return self._error_response(str(e))
    
    async def classify_document_type(self, text: str) -> Dict:
        """Classify document type (قانون، آیین‌نامه، رأی، etc.)"""
        try:
            # Simple heuristic-based classification for document types
            # In production, this would use a separate trained model
            text_lower = text.lower()
            
            type_keywords = {
                "قانون": ["قانون", "ماده", "تبصره", "فصل"],
                "آیین‌نامه": ["آیین‌نامه", "مقررات", "ضوابط"],
                "رأی": ["رأی", "دادگاه", "قاضی", "حکم"],
                "مصوبه": ["مصوبه", "شورا", "تصویب"],
                "بخشنامه": ["بخشنامه", "دستورالعمل", "راهنما"]
            }
            
            scores = {}
            for doc_type, keywords in type_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text)
                scores[doc_type] = score
            
            # Find the type with highest score
            predicted_type = max(scores, key=scores.get) if max(scores.values()) > 0 else "نامشخص"
            confidence = max(scores.values()) / len(text.split()) if text.split() else 0.0
            
            return {
                "document_type_fa": predicted_type,
                "document_type_en": next(
                    (v["en"] for k, v in self.document_types.items() if v["fa"] == predicted_type),
                    "unknown"
                ),
                "confidence": min(confidence, 1.0),
                "scores": scores,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Document type classification failed: {e}")
            return self._error_response(str(e))
    
    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract Persian keywords from legal text"""
        try:
            # Simple keyword extraction based on frequency and legal terms
            # In production, this would use more sophisticated NLP techniques
            
            # Common Persian legal terms
            legal_terms = [
                "قانون", "ماده", "تبصره", "فصل", "بخش", "کتاب",
                "حقوق", "قرارداد", "تعهد", "مسئولیت", "جرم", "مجازات",
                "دادگاه", "قاضی", "حکم", "رأی", "شهادت", "مدرک",
                "اشخاص", "اموال", "میراث", "ازدواج", "طلاق"
            ]
            
            words = text.split()
            word_freq = {}
            
            for word in words:
                cleaned_word = word.strip("،؛:.!؟()[]{}\"'")
                if len(cleaned_word) > 2 and cleaned_word in legal_terms:
                    word_freq[cleaned_word] = word_freq.get(cleaned_word, 0) + 1
            
            # Sort by frequency and return top keywords
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [keyword[0] for keyword in keywords[:max_keywords]]
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    async def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate Persian summary of legal text"""
        try:
            # Simple extractive summarization
            # In production, this would use a dedicated summarization model
            
            sentences = text.split('.')
            if len(sentences) <= 2:
                return text[:max_length]
            
            # Take first and last sentences as summary
            summary_sentences = [sentences[0].strip()]
            if len(sentences) > 1:
                summary_sentences.append(sentences[-1].strip())
            
            summary = '. '.join(summary_sentences)
            return summary[:max_length] + "..." if len(summary) > max_length else summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def _preprocess_persian_text(self, text: str) -> str:
        """Preprocess Persian text for better classification"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Normalize Persian characters
        persian_char_map = {
            'ي': 'ی',
            'ك': 'ک',
            'ء': 'ٔ'
        }
        
        for old_char, new_char in persian_char_map.items():
            text = text.replace(old_char, new_char)
        
        return text
    
    def _error_response(self, error_message: str) -> Dict:
        """Generate standardized error response"""
        return {
            "category_fa": "نامشخص",
            "category_en": "unknown",
            "confidence": 0.0,
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "is_initialized": self.is_initialized,
            "dora_enabled": hasattr(self.model, 'peft_config') if self.model else False,
            "num_categories": len(self.legal_categories),
            "num_document_types": len(self.document_types),
            "categories": {k: v["fa"] for k, v in self.legal_categories.items()},
            "document_types": {k: v["fa"] for k, v in self.document_types.items()},
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

# Global classifier instance
persian_classifier = PersianLegalAIClassifier()

# Test function
async def test_classifier():
    """Test the Persian legal classifier"""
    print("🧪 Testing Persian Legal AI Classifier...")
    
    # Initialize the classifier
    success = await persian_classifier.initialize_models(use_dora=True)
    if not success:
        print("❌ Failed to initialize classifier")
        return
    
    # Test classification
    test_text = "این قانون در راستای تنظیم روابط مدنی میان اشخاص حقیقی و حقوقی وضع شده است."
    
    result = await persian_classifier.classify_document(test_text)
    print(f"✅ Classification result: {result}")
    
    # Test document type classification
    doc_type_result = await persian_classifier.classify_document_type(test_text)
    print(f"✅ Document type result: {doc_type_result}")
    
    # Test keyword extraction
    keywords = await persian_classifier.extract_keywords(test_text)
    print(f"✅ Keywords: {keywords}")
    
    # Test summary generation
    summary = await persian_classifier.generate_summary(test_text)
    print(f"✅ Summary: {summary}")
    
    # Get model info
    model_info = persian_classifier.get_model_info()
    print(f"✅ Model info: {json.dumps(model_info, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    asyncio.run(test_classifier())