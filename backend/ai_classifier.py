import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Optional
import asyncio
from functools import lru_cache
import logging

# Import hardware detection service
from .services.hardware_detector import HardwareDetector

logger = logging.getLogger(__name__)

# Try to import Persian NLP tools
try:
    import hazm
    HAZM_AVAILABLE = True
except ImportError:
    HAZM_AVAILABLE = False
    print("⚠️ Hazm not available, using fallback text processing")

class PersianBERTClassifier:
    """Enhanced Persian BERT classifier with dynamic hardware detection and optimization"""
    
    def __init__(self, model_name: str = None):
        # Initialize hardware detection for dynamic model selection
        self.hardware_detector = HardwareDetector()
        self.model_config = self.hardware_detector.select_optimal_model_config()
        
        # Use dynamic model selection or fallback to provided model
        self.model_name = model_name or self.model_config['model_name']
        self.device = torch.device(self.model_config['device'])
        self.batch_size = self.model_config['batch_size']
        self.max_length = self.model_config['max_length']
        self.quantization = self.model_config['quantization']
        self.memory_efficient = self.model_config['memory_efficient']
        
        self.tokenizer = None
        self.model = None
        
        logger.info(f"Initializing Persian BERT Classifier with hardware-optimized config: {self.hardware_detector.get_hardware_summary()}")
        
        # Initialize Persian NLP tools if available
        if HAZM_AVAILABLE:
            self.normalizer = hazm.Normalizer()
            self.word_tokenizer = hazm.WordTokenizer()
        else:
            self.normalizer = None
            self.word_tokenizer = None
            
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the BERT model and tokenizer with hardware optimization"""
        try:
            logger.info(f"🤖 Loading Persian BERT model: {self.model_name}")
            logger.info(f"📊 Hardware config: {self.model_config['hardware_profile']}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Configure model loading based on hardware capabilities
            model_kwargs = {
                'num_labels': 5,  # Adjust based on your classification needs
            }
            
            # Add quantization if enabled
            if self.quantization:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                    model_kwargs['quantization_config'] = quantization_config
                    logger.info("✅ 8-bit quantization enabled")
                except ImportError:
                    logger.warning("BitsAndBytesConfig not available, skipping quantization")
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to appropriate device
            self.model.to(self.device)
            self.model.eval()
            
            # Enable memory efficient attention if supported
            if self.memory_efficient and hasattr(self.model.config, 'use_memory_efficient_attention'):
                self.model.config.use_memory_efficient_attention = True
            
            # Enable gradient checkpointing for memory efficiency
            if self.memory_efficient and hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            logger.info(f"✅ Persian BERT model loaded successfully on {self.device}")
            logger.info(f"🔧 Optimizations: quantization={self.quantization}, memory_efficient={self.memory_efficient}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Persian BERT model: {e}")
            logger.info("📋 Falling back to keyword-based classification")
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
            
            # Tokenize with dynamic max_length based on hardware
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference with hardware-optimized settings
            with torch.no_grad():
                # Use autocast for mixed precision if available and beneficial
                if self.device.type == 'cuda' and self.model_config.get('precision') == 'fp16':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**inputs)
                else:
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
    
    def get_system_info(self) -> Dict[str, any]:
        """Get hardware and model configuration info"""
        return {
            'hardware_summary': self.hardware_detector.get_hardware_summary(),
            'hardware_info': self.hardware_detector.hardware_info,
            'model_config': self.model_config,
            'model_name': self.model_name,
            'device': str(self.device),
            'optimization_config': self.hardware_detector.optimization_config,
            'performance_recommendations': self.hardware_detector.get_performance_recommendations(),
            'production_ready': self.hardware_detector.is_production_ready(),
            'optimal_for_platform': True
        }
    
    def get_hardware_score(self) -> float:
        """Get hardware capability score"""
        return self.hardware_detector._calculate_hardware_score()

# Global classifier instance
classifier = PersianBERTClassifier()

# Convenience functions
async def classify_text_async(text: str) -> Dict[str, float]:
    """Async function to classify Persian legal text"""
    return await classifier.classify_async(text)

def classify_text(text: str) -> Dict[str, float]:
    """Sync function to classify Persian legal text"""
    return classifier.classify_sync(text)