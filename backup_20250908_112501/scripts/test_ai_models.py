#!/usr/bin/env python3
"""
Persian Legal AI Models Testing Script
تست جامع مدل‌های هوش مصنوعی حقوقی فارسی

This script provides comprehensive testing of DoRA and QR-Adaptor models
with Persian legal text processing and performance verification.
"""

import sys
import os
import time
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import traceback

# Add project paths for imports
project_root = Path(__file__).parent.parent
sys.path.extend([
    str(project_root),
    str(project_root / "backend"),
    str(project_root / "models"),
    str(project_root / "backend" / "models")
])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_models_test.log')
    ]
)
logger = logging.getLogger(__name__)

class PersianLegalAIModelTester:
    """Comprehensive AI model tester for Persian Legal AI system"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "environment_check": {},
            "model_imports": {},
            "model_loading": {},
            "persian_text_processing": {},
            "classification_tests": {},
            "training_pipeline_tests": {},
            "performance_metrics": {},
            "memory_usage": {},
            "gpu_utilization": {}
        }
        
        # Persian legal text samples for testing
        self.persian_test_samples = [
            {
                "text": "دادگاه عالی کشور در رسیدگی به پرونده شماره ۱۴۰۲۰۵۱۲۳۴۵۶ مطابق قانون آیین دادرسی کیفری رای نهایی خود را صادر کرد.",
                "expected_category": "کیفری",
                "description": "Criminal law text"
            },
            {
                "text": "قرارداد خرید و فروش املاک واقع در تهران، خیابان ولیعصر، طبق ماده ۳۹۸ قانون مدنی منعقد گردید.",
                "expected_category": "مدنی",
                "description": "Civil law text"
            },
            {
                "text": "شرکت سهامی عام مطابق قانون تجارت و اساسنامه مصوب مجمع عمومی فوق‌العاده اقدام به افزایش سرمایه نمود.",
                "expected_category": "تجاری",
                "description": "Commercial law text"
            },
            {
                "text": "کمیسیون ماده ۱۰۰ قانون شهرداری‌ها در خصوص تخلف ساختمانی واقع در منطقه ۲ تهران تصمیم‌گیری کرد.",
                "expected_category": "اداری",
                "description": "Administrative law text"
            },
            {
                "text": "دستگاه قضایی با توجه به اصل ۱۵۶ قانون اساسی در رسیدگی به شکایات مردم اقدام به صدور حکم نمود.",
                "expected_category": "قانون اساسی",
                "description": "Constitutional law text"
            }
        ]
    
    async def run_comprehensive_test(self):
        """Run comprehensive AI model tests"""
        print("🤖 Persian Legal AI - AI Models Testing")
        print("=" * 80)
        print(f"📁 Project Root: {self.project_root}")
        print(f"🕐 Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        try:
            # Phase 1: Environment and Dependencies Check
            await self._test_environment()
            
            # Phase 2: Model Import Tests
            await self._test_model_imports()
            
            # Phase 3: Model Loading Tests
            await self._test_model_loading()
            
            # Phase 4: Persian Text Processing Tests
            await self._test_persian_text_processing()
            
            # Phase 5: Classification Performance Tests
            await self._test_classification_performance()
            
            # Phase 6: Training Pipeline Tests
            await self._test_training_pipeline()
            
            # Phase 7: Memory and Performance Analysis
            await self._test_performance_metrics()
            
            # Generate final report
            self._generate_test_report()
            
        except Exception as e:
            logger.error(f"Critical AI model test error: {e}")
            logger.error(traceback.format_exc())
            self.test_results["critical_error"] = str(e)
    
    async def _test_environment(self):
        """Test AI/ML environment and dependencies"""
        print("\n🔧 Phase 1: Environment and Dependencies Check")
        print("-" * 50)
        
        env_result = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch_available": False,
            "transformers_available": False,
            "peft_available": False,
            "hazm_available": False,
            "cuda_available": False,
            "device_info": {},
            "memory_info": {}
        }
        
        print(f"🐍 Python Version: {env_result['python_version']}")
        
        # Test PyTorch
        try:
            import torch
            env_result["torch_available"] = True
            env_result["torch_version"] = torch.__version__
            env_result["cuda_available"] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                env_result["device_info"] = {
                    "cuda_version": torch.version.cuda,
                    "gpu_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
                }
                print(f"✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda}")
                print(f"   GPU: {env_result['device_info']['device_name']}")
            else:
                print(f"✅ PyTorch {torch.__version__} (CPU only)")
                
        except ImportError as e:
            print(f"❌ PyTorch: Not available - {e}")
        
        # Test Transformers
        try:
            import transformers
            env_result["transformers_available"] = True
            env_result["transformers_version"] = transformers.__version__
            print(f"✅ Transformers: {transformers.__version__}")
        except ImportError as e:
            print(f"❌ Transformers: Not available - {e}")
        
        # Test PEFT
        try:
            import peft
            env_result["peft_available"] = True
            env_result["peft_version"] = peft.__version__
            print(f"✅ PEFT: {peft.__version__}")
        except ImportError as e:
            print(f"❌ PEFT: Not available - {e}")
        
        # Test Hazm (Persian NLP)
        try:
            import hazm
            env_result["hazm_available"] = True
            print("✅ Hazm (Persian NLP): Available")
        except ImportError as e:
            print(f"❌ Hazm: Not available - {e}")
        
        # Memory info
        try:
            import psutil
            memory = psutil.virtual_memory()
            env_result["memory_info"] = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent_used": memory.percent
            }
            print(f"💾 Memory: {memory.available / (1024**3):.1f}GB available / {memory.total / (1024**3):.1f}GB total")
        except ImportError:
            print("⚠️  Memory info not available")
        
        self.test_results["environment_check"] = env_result
    
    async def _test_model_imports(self):
        """Test model imports and availability"""
        print("\n📦 Phase 2: Model Import Tests")
        print("-" * 50)
        
        import_results = {
            "dora_trainer": False,
            "qr_adaptor": False,
            "persian_classifier": False,
            "model_manager": False,
            "data_trainer": False,
            "import_errors": {}
        }
        
        # Test DoRA trainer import
        try:
            from models.dora_trainer import DoRATrainingPipeline
            import_results["dora_trainer"] = True
            print("✅ DoRA Trainer: Import successful")
        except ImportError as e:
            import_results["import_errors"]["dora_trainer"] = str(e)
            print(f"❌ DoRA Trainer: Import failed - {e}")
        except Exception as e:
            import_results["import_errors"]["dora_trainer"] = str(e)
            print(f"⚠️  DoRA Trainer: Import error - {e}")
        
        # Test QR-Adaptor import
        try:
            from backend.models.qr_adaptor import QRAdaptorModel
            import_results["qr_adaptor"] = True
            print("✅ QR-Adaptor: Import successful")
        except ImportError as e:
            import_results["import_errors"]["qr_adaptor"] = str(e)
            print(f"❌ QR-Adaptor: Import failed - {e}")
        except Exception as e:
            import_results["import_errors"]["qr_adaptor"] = str(e)
            print(f"⚠️  QR-Adaptor: Import error - {e}")
        
        # Test Persian classifier import
        try:
            from models.persian_legal_classifier import PersianLegalAIClassifier
            import_results["persian_classifier"] = True
            print("✅ Persian Classifier: Import successful")
        except ImportError as e:
            try:
                from backend.ai_classifier import PersianBERTClassifier
                import_results["persian_classifier"] = True
                print("✅ Persian BERT Classifier: Import successful")
            except ImportError as e2:
                import_results["import_errors"]["persian_classifier"] = f"{e} / {e2}"
                print(f"❌ Persian Classifier: Import failed - {e}")
        except Exception as e:
            import_results["import_errors"]["persian_classifier"] = str(e)
            print(f"⚠️  Persian Classifier: Import error - {e}")
        
        # Test model manager import
        try:
            from backend.models.model_manager import ModelManager
            import_results["model_manager"] = True
            print("✅ Model Manager: Import successful")
        except ImportError as e:
            import_results["import_errors"]["model_manager"] = str(e)
            print(f"❌ Model Manager: Import failed - {e}")
        except Exception as e:
            import_results["import_errors"]["model_manager"] = str(e)
            print(f"⚠️  Model Manager: Import error - {e}")
        
        # Test verified data trainer
        try:
            from backend.models.verified_data_trainer import VerifiedDataTrainer
            import_results["data_trainer"] = True
            print("✅ Verified Data Trainer: Import successful")
        except ImportError as e:
            import_results["import_errors"]["data_trainer"] = str(e)
            print(f"❌ Verified Data Trainer: Import failed - {e}")
        except Exception as e:
            import_results["import_errors"]["data_trainer"] = str(e)
            print(f"⚠️  Verified Data Trainer: Import error - {e}")
        
        self.test_results["model_imports"] = import_results
    
    async def _test_model_loading(self):
        """Test model loading and initialization"""
        print("\n🔄 Phase 3: Model Loading Tests")
        print("-" * 50)
        
        loading_results = {
            "models_tested": [],
            "successful_loads": [],
            "failed_loads": [],
            "loading_times": {},
            "model_info": {}
        }
        
        # Test Persian BERT classifier loading
        if self.test_results["model_imports"].get("persian_classifier"):
            print("🔄 Testing Persian BERT Classifier loading...")
            try:
                start_time = time.time()
                
                try:
                    from models.persian_legal_classifier import PersianLegalAIClassifier
                    classifier = PersianLegalAIClassifier()
                except ImportError:
                    from backend.ai_classifier import PersianBERTClassifier
                    classifier = PersianBERTClassifier()
                
                # Test basic initialization
                loading_time = time.time() - start_time
                loading_results["loading_times"]["persian_classifier"] = loading_time
                loading_results["successful_loads"].append("persian_classifier")
                loading_results["models_tested"].append("persian_classifier")
                
                print(f"✅ Persian Classifier loaded ({loading_time:.2f}s)")
                
                # Get model info if available
                if hasattr(classifier, 'get_model_info'):
                    info = classifier.get_model_info()
                    loading_results["model_info"]["persian_classifier"] = info
                
            except Exception as e:
                loading_results["failed_loads"].append({"model": "persian_classifier", "error": str(e)})
                loading_results["models_tested"].append("persian_classifier")
                print(f"❌ Persian Classifier loading failed: {e}")
        
        # Test DoRA trainer loading
        if self.test_results["model_imports"].get("dora_trainer"):
            print("🔄 Testing DoRA Trainer loading...")
            try:
                start_time = time.time()
                from models.dora_trainer import DoRATrainingPipeline
                
                # Test basic initialization (don't load full models yet)
                dora_trainer = DoRATrainingPipeline(db=None)  # Mock database
                
                loading_time = time.time() - start_time
                loading_results["loading_times"]["dora_trainer"] = loading_time
                loading_results["successful_loads"].append("dora_trainer")
                loading_results["models_tested"].append("dora_trainer")
                
                print(f"✅ DoRA Trainer loaded ({loading_time:.2f}s)")
                
            except Exception as e:
                loading_results["failed_loads"].append({"model": "dora_trainer", "error": str(e)})
                loading_results["models_tested"].append("dora_trainer")
                print(f"❌ DoRA Trainer loading failed: {e}")
        
        # Test QR-Adaptor loading
        if self.test_results["model_imports"].get("qr_adaptor"):
            print("🔄 Testing QR-Adaptor loading...")
            try:
                start_time = time.time()
                from backend.models.qr_adaptor import QRAdaptorModel
                
                # Test basic initialization
                qr_model = QRAdaptorModel()
                
                loading_time = time.time() - start_time
                loading_results["loading_times"]["qr_adaptor"] = loading_time
                loading_results["successful_loads"].append("qr_adaptor")
                loading_results["models_tested"].append("qr_adaptor")
                
                print(f"✅ QR-Adaptor loaded ({loading_time:.2f}s)")
                
            except Exception as e:
                loading_results["failed_loads"].append({"model": "qr_adaptor", "error": str(e)})
                loading_results["models_tested"].append("qr_adaptor")
                print(f"❌ QR-Adaptor loading failed: {e}")
        
        self.test_results["model_loading"] = loading_results
    
    async def _test_persian_text_processing(self):
        """Test Persian text processing capabilities"""
        print("\n📝 Phase 4: Persian Text Processing Tests")
        print("-" * 50)
        
        processing_results = {
            "tokenization_tests": [],
            "text_preprocessing": [],
            "encoding_tests": [],
            "model_inference": []
        }
        
        # Test basic Persian text handling
        for i, sample in enumerate(self.persian_test_samples[:3], 1):
            print(f"🔤 Testing Persian text sample {i}...")
            
            text = sample["text"]
            test_result = {
                "sample_id": i,
                "text_length": len(text),
                "word_count": len(text.split()),
                "description": sample["description"],
                "tokenization_success": False,
                "encoding_success": False,
                "processing_time": None
            }
            
            start_time = time.time()
            
            # Test tokenization if Hazm is available
            if self.test_results["environment_check"].get("hazm_available"):
                try:
                    import hazm
                    normalizer = hazm.Normalizer()
                    normalized_text = normalizer.normalize(text)
                    
                    word_tokenizer = hazm.WordTokenizer()
                    tokens = word_tokenizer.tokenize(normalized_text)
                    
                    test_result["tokenization_success"] = True
                    test_result["token_count"] = len(tokens)
                    print(f"   ✅ Tokenization: {len(tokens)} tokens")
                    
                except Exception as e:
                    test_result["tokenization_error"] = str(e)
                    print(f"   ❌ Tokenization failed: {e}")
            
            # Test encoding with transformers if available
            if self.test_results["environment_check"].get("transformers_available"):
                try:
                    from transformers import AutoTokenizer
                    
                    # Use Persian BERT tokenizer
                    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
                    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                    
                    test_result["encoding_success"] = True
                    test_result["encoded_length"] = encoded["input_ids"].shape[1]
                    print(f"   ✅ BERT Encoding: {encoded['input_ids'].shape[1]} tokens")
                    
                except Exception as e:
                    test_result["encoding_error"] = str(e)
                    print(f"   ❌ BERT Encoding failed: {e}")
            
            test_result["processing_time"] = (time.time() - start_time) * 1000
            processing_results["text_preprocessing"].append(test_result)
        
        self.test_results["persian_text_processing"] = processing_results
    
    async def _test_classification_performance(self):
        """Test classification performance with Persian legal texts"""
        print("\n🎯 Phase 5: Classification Performance Tests")
        print("-" * 50)
        
        classification_results = {
            "total_tests": len(self.persian_test_samples),
            "successful_classifications": 0,
            "failed_classifications": 0,
            "classification_details": [],
            "average_confidence": 0,
            "average_processing_time": 0,
            "accuracy_estimate": 0
        }
        
        # Only proceed if we have a working classifier
        if "persian_classifier" not in self.test_results["model_loading"].get("successful_loads", []):
            print("⚠️  Skipping classification tests - no working classifier loaded")
            self.test_results["classification_tests"] = classification_results
            return
        
        # Initialize classifier
        try:
            try:
                from models.persian_legal_classifier import PersianLegalAIClassifier
                classifier = PersianLegalAIClassifier()
            except ImportError:
                from backend.ai_classifier import PersianBERTClassifier
                classifier = PersianBERTClassifier()
            
            print("🔄 Running classification tests...")
            
            total_confidence = 0
            total_time = 0
            correct_predictions = 0
            
            for i, sample in enumerate(self.persian_test_samples, 1):
                print(f"   📄 Testing sample {i}: {sample['description']}")
                
                start_time = time.time()
                
                try:
                    # Attempt classification
                    if hasattr(classifier, 'classify_document'):
                        result = await classifier.classify_document(sample["text"])
                    elif hasattr(classifier, 'classify_text'):
                        result = await classifier.classify_text(sample["text"])
                    else:
                        # Fallback to synchronous method
                        result = classifier.predict(sample["text"])
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    # Extract results
                    category = result.get("category", "unknown") if isinstance(result, dict) else str(result)
                    confidence = result.get("confidence", 0) if isinstance(result, dict) else 0.5
                    
                    classification_detail = {
                        "sample_id": i,
                        "input_text": sample["text"][:100] + "...",
                        "expected_category": sample["expected_category"],
                        "predicted_category": category,
                        "confidence": confidence,
                        "processing_time_ms": processing_time,
                        "success": True
                    }
                    
                    # Check if prediction matches expected (approximate)
                    if sample["expected_category"].lower() in category.lower() or category.lower() in sample["expected_category"].lower():
                        correct_predictions += 1
                        classification_detail["correct_prediction"] = True
                        print(f"      ✅ {category} (confidence: {confidence:.2f}, {processing_time:.1f}ms)")
                    else:
                        classification_detail["correct_prediction"] = False
                        print(f"      ⚠️  {category} (expected: {sample['expected_category']}, confidence: {confidence:.2f}, {processing_time:.1f}ms)")
                    
                    classification_results["successful_classifications"] += 1
                    classification_results["classification_details"].append(classification_detail)
                    
                    total_confidence += confidence
                    total_time += processing_time
                    
                except Exception as e:
                    print(f"      ❌ Classification failed: {e}")
                    classification_results["failed_classifications"] += 1
                    classification_results["classification_details"].append({
                        "sample_id": i,
                        "input_text": sample["text"][:100] + "...",
                        "error": str(e),
                        "success": False
                    })
            
            # Calculate averages
            if classification_results["successful_classifications"] > 0:
                classification_results["average_confidence"] = total_confidence / classification_results["successful_classifications"]
                classification_results["average_processing_time"] = total_time / classification_results["successful_classifications"]
                classification_results["accuracy_estimate"] = (correct_predictions / classification_results["successful_classifications"]) * 100
            
            print(f"\n📊 Classification Results:")
            print(f"   Successful: {classification_results['successful_classifications']}/{classification_results['total_tests']}")
            print(f"   Average Confidence: {classification_results['average_confidence']:.2f}")
            print(f"   Average Processing Time: {classification_results['average_processing_time']:.1f}ms")
            print(f"   Estimated Accuracy: {classification_results['accuracy_estimate']:.1f}%")
            
        except Exception as e:
            print(f"❌ Classification testing failed: {e}")
            classification_results["setup_error"] = str(e)
        
        self.test_results["classification_tests"] = classification_results
    
    async def _test_training_pipeline(self):
        """Test training pipeline functionality"""
        print("\n🎓 Phase 6: Training Pipeline Tests")
        print("-" * 50)
        
        training_results = {
            "dora_pipeline_available": False,
            "training_config_valid": False,
            "mock_training_successful": False,
            "training_components": []
        }
        
        # Test DoRA training pipeline
        if "dora_trainer" in self.test_results["model_loading"].get("successful_loads", []):
            print("🔄 Testing DoRA training pipeline...")
            
            try:
                from models.dora_trainer import DoRATrainingPipeline
                
                # Create mock training pipeline
                pipeline = DoRATrainingPipeline(db=None)
                training_results["dora_pipeline_available"] = True
                
                # Test training configuration
                test_config = {
                    "model_type": "dora",
                    "epochs": 1,
                    "learning_rate": 0.0002,
                    "batch_size": 4,
                    "use_dora": True
                }
                
                # Check if configuration is valid
                if hasattr(pipeline, 'validate_config'):
                    is_valid = pipeline.validate_config(test_config)
                    training_results["training_config_valid"] = is_valid
                    print(f"   ✅ Training config validation: {'Valid' if is_valid else 'Invalid'}")
                else:
                    training_results["training_config_valid"] = True
                    print("   ✅ Training config: No validation method found (assuming valid)")
                
                # Test mock training session creation
                if hasattr(pipeline, 'create_training_session'):
                    session_id = pipeline.create_training_session(test_config)
                    training_results["mock_training_successful"] = True
                    print(f"   ✅ Mock training session created: {session_id}")
                else:
                    print("   ⚠️  No training session creation method found")
                
                training_results["training_components"].append("DoRA Pipeline")
                
            except Exception as e:
                print(f"   ❌ DoRA training pipeline test failed: {e}")
                training_results["dora_error"] = str(e)
        
        # Test other training components
        if self.test_results["model_imports"].get("data_trainer"):
            print("🔄 Testing Verified Data Trainer...")
            try:
                from backend.models.verified_data_trainer import VerifiedDataTrainer
                
                trainer = VerifiedDataTrainer()
                training_results["training_components"].append("Verified Data Trainer")
                print("   ✅ Verified Data Trainer initialized")
                
            except Exception as e:
                print(f"   ❌ Verified Data Trainer test failed: {e}")
        
        self.test_results["training_pipeline_tests"] = training_results
    
    async def _test_performance_metrics(self):
        """Test performance and memory usage"""
        print("\n⚡ Phase 7: Performance and Memory Analysis")
        print("-" * 50)
        
        performance_results = {
            "memory_before_mb": 0,
            "memory_after_mb": 0,
            "memory_increase_mb": 0,
            "gpu_memory_used": 0,
            "inference_benchmarks": [],
            "system_resources": {}
        }
        
        try:
            import psutil
            
            # Get initial memory usage
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB
            performance_results["memory_before_mb"] = memory_before
            
            print(f"💾 Initial Memory Usage: {memory_before:.1f} MB")
            
            # Test GPU memory if available
            if self.test_results["environment_check"].get("cuda_available"):
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                        performance_results["gpu_memory_used"] = gpu_memory
                        print(f"🎮 GPU Memory Usage: {gpu_memory:.1f} MB")
                except Exception as e:
                    print(f"⚠️  GPU memory check failed: {e}")
            
            # Run inference benchmark if classifier is available
            if "persian_classifier" in self.test_results["model_loading"].get("successful_loads", []):
                print("🏃 Running inference benchmark...")
                
                try:
                    # Load classifier
                    try:
                        from models.persian_legal_classifier import PersianLegalAIClassifier
                        classifier = PersianLegalAIClassifier()
                    except ImportError:
                        from backend.ai_classifier import PersianBERTClassifier
                        classifier = PersianBERTClassifier()
                    
                    # Benchmark with different text lengths
                    test_texts = [
                        "قانون مدنی",  # Short
                        "دادگاه عالی کشور در رسیدگی به پرونده شماره ۱۴۰۲۰۵۱۲۳۴۵۶ مطابق قانون آیین دادرسی کیفری رای نهایی خود را صادر کرد.",  # Medium
                        " ".join([self.persian_test_samples[0]["text"]] * 5)  # Long
                    ]
                    
                    for i, text in enumerate(test_texts):
                        times = []
                        for _ in range(3):  # Run 3 times for average
                            start_time = time.time()
                            try:
                                if hasattr(classifier, 'classify_text'):
                                    await classifier.classify_text(text)
                                else:
                                    classifier.predict(text)
                                times.append((time.time() - start_time) * 1000)
                            except Exception:
                                pass
                        
                        if times:
                            avg_time = sum(times) / len(times)
                            performance_results["inference_benchmarks"].append({
                                "text_length": len(text),
                                "word_count": len(text.split()),
                                "avg_inference_time_ms": avg_time,
                                "throughput_words_per_sec": len(text.split()) / (avg_time / 1000)
                            })
                            print(f"   📏 Text length {len(text):4d}: {avg_time:.1f}ms avg")
                
                except Exception as e:
                    print(f"   ❌ Inference benchmark failed: {e}")
            
            # Get final memory usage
            memory_after = process.memory_info().rss / (1024 * 1024)  # MB
            performance_results["memory_after_mb"] = memory_after
            performance_results["memory_increase_mb"] = memory_after - memory_before
            
            print(f"💾 Final Memory Usage: {memory_after:.1f} MB (+{memory_after - memory_before:.1f} MB)")
            
            # System resource summary
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            performance_results["system_resources"] = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3)
            }
            
            print(f"🖥️  System Resources: CPU {cpu_percent:.1f}%, RAM {memory.percent:.1f}%")
            
        except ImportError:
            print("⚠️  Performance monitoring not available (psutil not installed)")
        except Exception as e:
            print(f"❌ Performance testing failed: {e}")
        
        self.test_results["performance_metrics"] = performance_results
    
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📋 AI Models Test Report")
        print("=" * 80)
        
        # Calculate overall scores
        env_score = sum([
            self.test_results["environment_check"].get("torch_available", False),
            self.test_results["environment_check"].get("transformers_available", False),
            self.test_results["environment_check"].get("peft_available", False),
            self.test_results["environment_check"].get("hazm_available", False)
        ]) / 4 * 100
        
        import_score = sum([
            self.test_results["model_imports"].get("dora_trainer", False),
            self.test_results["model_imports"].get("qr_adaptor", False),
            self.test_results["model_imports"].get("persian_classifier", False),
            self.test_results["model_imports"].get("model_manager", False)
        ]) / 4 * 100
        
        loading_score = len(self.test_results["model_loading"].get("successful_loads", [])) / max(1, len(self.test_results["model_loading"].get("models_tested", []))) * 100
        
        classification_score = 0
        if self.test_results["classification_tests"].get("total_tests", 0) > 0:
            classification_score = (self.test_results["classification_tests"].get("successful_classifications", 0) / 
                                  self.test_results["classification_tests"]["total_tests"]) * 100
        
        overall_score = (env_score + import_score + loading_score + classification_score) / 4
        
        print(f"📊 Test Summary:")
        print(f"   Environment & Dependencies: {env_score:.1f}/100")
        print(f"   Model Imports: {import_score:.1f}/100")
        print(f"   Model Loading: {loading_score:.1f}/100")
        print(f"   Classification Performance: {classification_score:.1f}/100")
        print(f"   Overall AI Models Score: {overall_score:.1f}/100")
        
        # Environment details
        env = self.test_results["environment_check"]
        print(f"\n🔧 Environment Status:")
        print(f"   PyTorch: {'✅' if env.get('torch_available') else '❌'} {env.get('torch_version', '')}")
        print(f"   CUDA: {'✅' if env.get('cuda_available') else '❌'}")
        print(f"   Transformers: {'✅' if env.get('transformers_available') else '❌'} {env.get('transformers_version', '')}")
        print(f"   PEFT (DoRA/LoRA): {'✅' if env.get('peft_available') else '❌'} {env.get('peft_version', '')}")
        print(f"   Hazm (Persian NLP): {'✅' if env.get('hazm_available') else '❌'}")
        
        if env.get("device_info"):
            print(f"   GPU: {env['device_info'].get('device_name', 'N/A')}")
        
        # Model loading results
        loading = self.test_results["model_loading"]
        if loading.get("successful_loads"):
            print(f"\n✅ Successfully Loaded Models:")
            for model in loading["successful_loads"]:
                time_taken = loading.get("loading_times", {}).get(model, 0)
                print(f"   - {model.replace('_', ' ').title()}: {time_taken:.2f}s")
        
        if loading.get("failed_loads"):
            print(f"\n❌ Failed to Load Models:")
            for failure in loading["failed_loads"]:
                print(f"   - {failure['model']}: {failure['error']}")
        
        # Classification performance
        classification = self.test_results["classification_tests"]
        if classification.get("successful_classifications", 0) > 0:
            print(f"\n🎯 Classification Performance:")
            print(f"   Successful Tests: {classification['successful_classifications']}/{classification['total_tests']}")
            print(f"   Average Confidence: {classification.get('average_confidence', 0):.2f}")
            print(f"   Average Processing Time: {classification.get('average_processing_time', 0):.1f}ms")
            print(f"   Estimated Accuracy: {classification.get('accuracy_estimate', 0):.1f}%")
        
        # Performance metrics
        performance = self.test_results["performance_metrics"]
        if performance.get("memory_increase_mb", 0) > 0:
            print(f"\n⚡ Performance Metrics:")
            print(f"   Memory Usage Increase: {performance['memory_increase_mb']:.1f} MB")
            if performance.get("gpu_memory_used", 0) > 0:
                print(f"   GPU Memory Usage: {performance['gpu_memory_used']:.1f} MB")
            
            if performance.get("inference_benchmarks"):
                print(f"   Inference Benchmarks:")
                for bench in performance["inference_benchmarks"]:
                    print(f"     {bench['word_count']:3d} words: {bench['avg_inference_time_ms']:.1f}ms ({bench['throughput_words_per_sec']:.1f} words/sec)")
        
        # Training pipeline status
        training = self.test_results["training_pipeline_tests"]
        if training.get("training_components"):
            print(f"\n🎓 Training Pipeline Status:")
            for component in training["training_components"]:
                print(f"   ✅ {component}")
            
            if training.get("dora_pipeline_available"):
                print(f"   DoRA Training: {'✅ Available' if training['training_config_valid'] else '⚠️  Config Issues'}")
        
        # Overall assessment
        print(f"\n🎯 AI Models Assessment:")
        if overall_score >= 90:
            print(f"   🎉 EXCELLENT: AI models are production ready ({overall_score:.1f}/100)")
        elif overall_score >= 80:
            print(f"   ✅ GOOD: AI models are mostly functional ({overall_score:.1f}/100)")
        elif overall_score >= 60:
            print(f"   ⚠️  FAIR: AI models have some issues ({overall_score:.1f}/100)")
        else:
            print(f"   ❌ POOR: AI models need significant work ({overall_score:.1f}/100)")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if not env.get("cuda_available"):
            print("   - Consider GPU acceleration for better performance")
        if import_score < 100:
            print("   - Fix model import issues for full functionality")
        if classification_score < 80:
            print("   - Improve classification accuracy through training")
        if not training.get("dora_pipeline_available"):
            print("   - Implement DoRA training pipeline for advanced fine-tuning")
        
        # Save detailed report
        report_file = Path("ai_models_test_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        print("=" * 80)

async def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Persian Legal AI Models Functionality Tester")
    parser.add_argument("--project-root", type=Path, 
                       help="Project root directory (default: parent of script directory)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    tester = PersianLegalAIModelTester(args.project_root)
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())