"""
Mock Enhanced Model Service (For Testing Without Heavy Dependencies)
==================================================================
Provides mock Persian BERT functionality for testing integration
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import time
import random

logger = logging.getLogger(__name__)

class MockEnhancedModelService:
    """Mock enhanced model service for testing purposes"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.enhanced_dir = self.project_root / "ai_models" / "enhanced"
        self.bert_path = self.enhanced_dir / "persian_bert"
        
        self.mock_model_loaded = False
        self.mock_tokenizer_loaded = False
        self.is_mock_mode = True
        
        # Try to load existing model service if available
        self.existing_service = self._load_existing_service()
        
        logger.info(f"Mock enhanced service initialized - Enhanced dir: {self.enhanced_dir}")
    
    def _load_existing_service(self):
        """Try to load existing model service without breaking"""
        try:
            # Try to import existing service
            import sys
            backend_path = self.project_root / "backend"
            if str(backend_path) not in sys.path:
                sys.path.insert(0, str(backend_path))
            
            # Look for existing model service
            try:
                from services.real_model_service import real_model_service
                logger.info("✅ Found existing real_model_service")
                return real_model_service
            except ImportError:
                pass
            
            try:
                from services.model_service import model_service
                logger.info("✅ Found existing model_service")
                return model_service
            except ImportError:
                pass
            
            logger.info("ℹ️ No existing model service found")
            return None
            
        except Exception as e:
            logger.warning(f"Could not load existing service: {e}")
            return None
    
    def load_mock_model(self) -> bool:
        """Load mock Persian BERT model"""
        if self.mock_model_loaded:
            return True
        
        if not self.bert_path.exists():
            logger.error(f"Mock Persian BERT not found at {self.bert_path}")
            logger.info("Run: python3 scripts/mock_enhanced_setup.py")
            return False
        
        try:
            logger.info("Loading mock Persian BERT...")
            start_time = time.time()
            
            # Simulate model loading time
            time.sleep(0.5)
            
            # Check if mock files exist
            config_file = self.bert_path / "config.json"
            vocab_file = self.bert_path / "vocab.txt"
            
            if config_file.exists() and vocab_file.exists():
                # Load mock config
                with open(config_file) as f:
                    self.mock_config = json.load(f)
                
                # Load mock vocab
                with open(vocab_file, encoding="utf-8") as f:
                    self.mock_vocab = [line.strip() for line in f.readlines()]
                
                load_time = time.time() - start_time
                self.mock_model_loaded = True
                self.mock_tokenizer_loaded = True
                logger.info(f"✅ Mock Persian BERT loaded in {load_time:.2f}s")
                return True
            else:
                logger.error("Mock model files not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load mock model: {e}")
            return False
    
    def classify_enhanced(self, text: str) -> Dict[str, Any]:
        """Enhanced classification using mock Persian BERT"""
        # Try mock model first
        if self.mock_model_loaded or self.load_mock_model():
            return self._classify_with_mock_bert(text)
        
        # Fallback to existing service if available
        if self.existing_service:
            try:
                if hasattr(self.existing_service, 'classify_document_real'):
                    return self.existing_service.classify_document_real(text)
                elif hasattr(self.existing_service, 'classify_document'):
                    return self.existing_service.classify_document(text)
            except Exception as e:
                logger.warning(f"Existing service failed: {e}")
        
        # Ultimate fallback
        return self._fallback_classification(text)
    
    def _classify_with_mock_bert(self, text: str) -> Dict[str, Any]:
        """Classification using mock Persian BERT"""
        start_time = time.time()
        
        try:
            # Simulate tokenization
            mock_tokens = self._mock_tokenize(text)
            
            # Simulate embedding generation (random but consistent)
            mock_embedding_size = 768
            random.seed(hash(text) % 2147483647)  # Consistent results for same text
            mock_embeddings = [random.uniform(-1, 1) for _ in range(mock_embedding_size)]
            
            # Enhanced classification logic with mock embeddings
            classification, confidence, all_scores = self._enhanced_classification_logic(text, mock_embeddings)
            
            processing_time = time.time() - start_time
            
            return {
                "classification": classification,
                "confidence": float(confidence),
                "all_scores": {k: float(v) for k, v in all_scores.items()},
                "processing_time": processing_time,
                "method": "mock_persian_bert",
                "embedding_shape": [1, len(mock_embeddings)],
                "model_source": "mock_enhanced",
                "token_count": len(mock_tokens),
                "is_mock": True
            }
            
        except Exception as e:
            logger.error(f"Mock classification failed: {e}")
            return self._fallback_classification(text)
    
    def _mock_tokenize(self, text: str) -> list:
        """Mock tokenization of Persian text"""
        # Simple mock tokenization
        tokens = ["[CLS]"]
        
        # Split by spaces and add to tokens
        words = text.split()
        for word in words[:50]:  # Limit to 50 words
            # Check if word is in mock vocab
            if hasattr(self, 'mock_vocab') and word in self.mock_vocab:
                tokens.append(word)
            else:
                # Split into subwords (mock BPE)
                if len(word) > 3:
                    tokens.extend([word[:2] + "##", "##" + word[2:]])
                else:
                    tokens.append(word)
        
        tokens.append("[SEP]")
        return tokens
    
    def _enhanced_classification_logic(self, text: str, embeddings: list) -> tuple:
        """Enhanced classification logic using mock embeddings and patterns"""
        text_lower = text.lower()
        
        # Enhanced patterns with weights
        enhanced_patterns = {
            "قرارداد": {
                "indicators": ["قرارداد", "طرف اول", "طرف دوم", "متعهد", "ماده", "پرداخت"],
                "weight": 1.0
            },
            "دادنامه": {
                "indicators": ["دادنامه", "دادگاه", "قاضی", "حکم", "رای", "محکومیت"],
                "weight": 1.0
            },
            "شکایت": {
                "indicators": ["شکایت", "شاکی", "متشاکی منه", "اتهام", "تقاضا", "درخواست"],
                "weight": 1.0
            },
            "وکالت‌نامه": {
                "indicators": ["وکالت", "موکل", "وکیل", "نمایندگی", "اختیار"],
                "weight": 0.9
            }
        }
        
        scores = {}
        
        for doc_type, pattern_info in enhanced_patterns.items():
            indicators = pattern_info["indicators"]
            weight = pattern_info["weight"]
            
            # Pattern matching score
            pattern_matches = sum(1 for indicator in indicators if indicator in text_lower)
            pattern_score = (pattern_matches / len(indicators)) * weight
            
            # Add mock embedding influence (simulate semantic understanding)
            embedding_influence = 1.0 + (sum(embeddings[:10]) / 10) * 0.1  # Use first 10 dimensions
            embedding_influence = max(0.8, min(1.2, embedding_influence))  # Clamp between 0.8 and 1.2
            
            final_score = pattern_score * embedding_influence
            scores[doc_type] = final_score
        
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            classification = best_type[0]
            confidence = min(best_type[1], 1.0)
        else:
            classification = "نامشخص"
            confidence = 0.0
        
        return classification, confidence, scores
    
    def _fallback_classification(self, text: str) -> Dict[str, Any]:
        """Fallback classification when mock model fails"""
        logger.warning("Using fallback classification")
        
        text_lower = text.lower()
        simple_patterns = {
            "قرارداد": ["قرارداد", "طرف اول", "طرف دوم"],
            "دادنامه": ["دادنامه", "دادگاه", "حکم"],
            "شکایت": ["شکایت", "شاکی", "اتهام"]
        }
        
        scores = {}
        for doc_type, indicators in simple_patterns.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            scores[doc_type] = score / len(indicators)
        
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            classification = best_type[0]
            confidence = best_type[1]
        else:
            classification = "نامشخص"
            confidence = 0.0
        
        return {
            "classification": classification,
            "confidence": float(confidence),
            "all_scores": {k: float(v) for k, v in scores.items()},
            "method": "fallback_patterns",
            "model_source": "fallback",
            "is_mock": True
        }
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get mock enhanced service status"""
        return {
            "enhanced_loaded": self.mock_model_loaded,
            "enhanced_path": str(self.bert_path),
            "enhanced_available": self.bert_path.exists(),
            "existing_service_available": self.existing_service is not None,
            "service_type": "mock_enhanced_preserving_existing",
            "is_mock_mode": self.is_mock_mode,
            "mock_tokenizer_loaded": self.mock_tokenizer_loaded
        }

# Global mock enhanced service instance
mock_enhanced_model_service = MockEnhancedModelService()