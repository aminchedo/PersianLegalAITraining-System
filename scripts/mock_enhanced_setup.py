#!/usr/bin/env python3
"""
Mock Enhanced Persian BERT Setup (For Testing Without Heavy Dependencies)
========================================================================
Creates mock Persian BERT setup for testing integration without downloading large models
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import time

class MockEnhancedModelSetup:
    """Mock enhanced model setup for testing purposes"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "ai_models"
        self.cache_dir = self.models_dir / "cache"
        self.enhanced_dir = self.models_dir / "enhanced"
        
        # Create directories safely
        for directory in [self.models_dir, self.cache_dir, self.enhanced_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"Mock enhanced setup initialized - Models dir: {self.models_dir}")
    
    def check_existing_setup(self) -> dict:
        """Check what's already working"""
        status = {
            "existing_models": [],
            "existing_data": [],
            "existing_scripts": []
        }
        
        # Check existing models
        if self.cache_dir.exists():
            existing_models = list(self.cache_dir.iterdir())
            status["existing_models"] = [m.name for m in existing_models if m.is_dir()]
        
        # Check existing datasets
        datasets_dir = self.models_dir / "datasets"
        if datasets_dir.exists():
            existing_data = list(datasets_dir.glob("*.json"))
            status["existing_data"] = [d.name for d in existing_data]
        
        print(f"✅ Found existing models: {status['existing_models']}")
        print(f"✅ Found existing data: {status['existing_data']}")
        
        return status
    
    def mock_install_dependencies(self) -> bool:
        """Mock dependency installation for testing"""
        print("📦 Mock dependency check (no actual installation)...")
        
        # Simulate dependency check without actually installing
        required_packages = ['torch', 'transformers', 'huggingface_hub']
        
        print("✅ Mock dependencies satisfied for testing")
        return True
    
    def create_mock_persian_bert(self) -> bool:
        """Create mock Persian BERT structure for testing"""
        bert_path = self.enhanced_dir / "persian_bert"
        
        if bert_path.exists():
            print(f"✅ Mock Persian BERT already exists at: {bert_path}")
            return True
        
        print("📥 Creating mock Persian BERT structure...")
        
        try:
            # Create mock model directory structure
            bert_path.mkdir(exist_ok=True)
            
            # Create mock config file
            mock_config = {
                "model_type": "bert",
                "vocab_size": 50000,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
                "max_position_embeddings": 512,
                "type_vocab_size": 2,
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-12,
                "pad_token_id": 0,
                "position_embedding_type": "absolute",
                "use_cache": True,
                "classifier_dropout": None,
                "architectures": ["BertModel"],
                "model_name": "HooshvareLab/bert-base-parsbert-uncased"
            }
            
            with open(bert_path / "config.json", "w") as f:
                json.dump(mock_config, f, indent=2)
            
            # Create mock tokenizer files
            mock_tokenizer_config = {
                "do_lower_case": True,
                "do_basic_tokenize": True,
                "never_split": None,
                "unk_token": "[UNK]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "mask_token": "[MASK]",
                "tokenize_chinese_chars": True,
                "strip_accents": None,
                "model_max_length": 512,
                "special_tokens_map_file": None,
                "name_or_path": "HooshvareLab/bert-base-parsbert-uncased",
                "tokenizer_class": "BertTokenizer"
            }
            
            with open(bert_path / "tokenizer_config.json", "w") as f:
                json.dump(mock_tokenizer_config, f, indent=2)
            
            # Create mock vocab file (simplified)
            mock_vocab = {
                "[PAD]": 0,
                "[UNK]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "[MASK]": 4,
                "و": 5,
                "در": 6,
                "به": 7,
                "از": 8,
                "که": 9,
                "این": 10,
                "برای": 11,
                "با": 12,
                "است": 13,
                "قرارداد": 14,
                "دادگاه": 15,
                "شکایت": 16,
                "وکالت": 17,
                "قانون": 18,
                "حقوق": 19,
                "مدنی": 20,
                "کیفری": 21
            }
            
            with open(bert_path / "vocab.txt", "w", encoding="utf-8") as f:
                for token, idx in sorted(mock_vocab.items(), key=lambda x: x[1]):
                    f.write(f"{token}\n")
            
            # Create mock model info
            model_info = {
                "model_name": "HooshvareLab/bert-base-parsbert-uncased",
                "local_path": str(bert_path),
                "vocab_size": len(mock_vocab),
                "verified_at": time.time(),
                "test_passed": True,
                "is_mock": True,
                "mock_created_at": time.time()
            }
            
            with open(bert_path.parent / "model_info.json", "w") as f:
                json.dump(model_info, f, indent=2)
            
            print("✅ Mock Persian BERT structure created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Mock creation failed: {e}")
            return False
    
    def create_enhanced_sample_data(self) -> bool:
        """Create enhanced sample data without overwriting existing"""
        enhanced_datasets_dir = self.enhanced_dir / "datasets"
        enhanced_datasets_dir.mkdir(exist_ok=True)
        
        # Check if enhanced data already exists
        enhanced_docs_file = enhanced_datasets_dir / "enhanced_legal_documents.json"
        if enhanced_docs_file.exists():
            print("✅ Enhanced sample data already exists")
            return True
        
        print("📄 Creating enhanced Persian legal sample data...")
        
        enhanced_documents = {
            "documents": [
                {
                    "id": "enhanced_contract_001",
                    "title": "قرارداد خرید و فروش ملک - نمونه پیشرفته",
                    "content": """قرارداد خرید و فروش ملک مسکونی
                    
این قرارداد در تاریخ ۱۴۰۲/۱۲/۲۰ بین:
- طرف اول: آقای محمد رضا احمدی فرزند علی، کد ملی: ۱۲۳۴۵۶۷۸۹۰ (فروشنده)
- طرف دوم: خانم فاطمه محمدی فرزند حسن، کد ملی: ۰۹۸۷۶۵۴۳۲۱ (خریدار)

منعقد گردید.

ماده ۱ - موضوع قرارداد:
فروشنده متعهد می‌شود ملک مسکونی واقع در تهران، منطقه ۳، خیابان ولیعصر، خیابان فلسطین، پلاک ۱۲۵، طبقه سوم، واحد شماره ۶ را با مساحت ۱۲۰ متر مربع به خریدار منتقل نماید.

ماده ۲ - مبلغ و نحوه پرداخت:
مبلغ کل معامله ۱۸،۰۰۰،۰۰۰،۰۰۰ (هجده میلیارد) ریال است که به شرح زیر پرداخت می‌گردد:
- پیش پرداخت: ۵،۰۰۰،۰۰۰،۰۰۰ ریال نقد
- باقیمانده: ۱۳،۰۰۰،۰۰۰،۰۰۰ ریال طی ۶ قسط ماهانه

ماده ۳ - تحویل ملک:
تحویل ملک حداکثر تا تاریخ ۱۴۰۳/۲/۱۵ انجام خواهد شد.""",
                    "category": "قرارداد",
                    "subcategory": "خرید و فروش ملک",
                    "legal_domain": "حقوق مدنی",
                    "complexity": "متوسط",
                    "entities": {
                        "parties": ["محمد رضا احمدی", "فاطمه محمدی"],
                        "amounts": ["۱۸،۰۰۰،۰۰۰،۰۰۰ ریال", "۵،۰۰۰،۰۰۰،۰۰۰ ریال"],
                        "dates": ["۱۴۰۲/۱۲/۲۰", "۱۴۰۳/۲/۱۵"],
                        "locations": ["تهران، منطقه ۳، خیابان ولیعصر"]
                    }
                },
                {
                    "id": "enhanced_judgment_001",
                    "title": "دادنامه دادگاه - نمونه پیشرفته",
                    "content": """دادنامه شماره ۱۴۰۲۰۹۸۷۶۵۴۳۲۱
                    
دادگاه عمومی حقوقی تهران - شعبه ۱۵
قاضی: جناب آقای دکتر حسن محمدی

در خصوص دعوای خانم مریم احمدی علیه آقای علی رضایی

با عنایت به اوراق پرونده و مستندات ارائه شده:

موضوع دعوا: مطالبه مبلغ ۵۰،۰۰۰،۰۰۰ ریال بابت قرارداد خرید و فروش

دلائل و مستندات:
۱. قرارداد خرید و فروش مورخ ۱۴۰۲/۵/۱۰
۲. فیش واریزی به مبلغ ۲۰،۰۰۰،۰۰۰ ریال
۳. شهادت شهود

رای دادگاه:
با توجه به مستندات ارائه شده، دادگاه دعوای خواهان را وارد تشخیص و حکم به محکومیت خوانده صادر می‌نماید.""",
                    "category": "دادنامه",
                    "subcategory": "حقوقی",
                    "legal_domain": "آیین دادرسی مدنی",
                    "complexity": "پیشرفته",
                    "entities": {
                        "parties": ["مریم احمدی", "علی رضایی"],
                        "judge": ["حسن محمدی"],
                        "amounts": ["۵۰،۰۰۰،۰۰۰ ریال", "۲۰،۰۰۰،۰۰۰ ریال"],
                        "case_number": ["۱۴۰۲۰۹۸۷۶۵۴۳۲۱"],
                        "court": ["دادگاه عمومی حقوقی تهران - شعبه ۱۵"]
                    }
                }
            ],
            "metadata": {
                "version": "enhanced_mock_1.0",
                "created_at": time.time(),
                "enhancement_level": "detailed_entities_with_mock",
                "language": "persian",
                "is_mock_data": True
            }
        }
        
        try:
            with open(enhanced_docs_file, "w", encoding="utf-8") as f:
                json.dump(enhanced_documents, f, ensure_ascii=False, indent=2)
            
            print("✅ Enhanced sample data created")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create enhanced data: {e}")
            return False
    
    def run_mock_enhancement(self) -> bool:
        """Run mock enhancement for testing purposes"""
        print("🚀 RUNNING MOCK SYSTEM ENHANCEMENT (Testing Mode)")
        print("=" * 55)
        
        # Check existing setup first
        existing_status = self.check_existing_setup()
        
        success_count = 0
        total_steps = 4
        
        # Step 1: Mock Dependencies
        print(f"\n📦 Step 1/{total_steps}: Mock dependencies check...")
        if self.mock_install_dependencies():
            success_count += 1
        
        # Step 2: Mock Persian BERT
        print(f"\n📥 Step 2/{total_steps}: Mock Persian BERT creation...")
        if self.create_mock_persian_bert():
            success_count += 1
        
        # Step 3: Enhanced sample data
        print(f"\n📄 Step 3/{total_steps}: Enhanced sample data...")
        if self.create_enhanced_sample_data():
            success_count += 1
        
        # Step 4: Integration check
        print(f"\n🔧 Step 4/{total_steps}: Mock integration verification...")
        integration_success = self.verify_mock_integration()
        if integration_success:
            success_count += 1
        
        success_rate = success_count / total_steps
        print(f"\n📊 MOCK ENHANCEMENT RESULTS: {success_count}/{total_steps} ({success_rate:.1%})")
        
        if success_rate >= 0.75:
            print("✅ Mock system enhancement successful!")
            print(f"📁 Mock enhanced models: {self.enhanced_dir}")
            print("🔄 Existing functionality preserved")
            print("🎭 Running in MOCK MODE for testing")
            return True
        else:
            print("⚠️ Mock enhancement partially successful")
            return False
    
    def verify_mock_integration(self) -> bool:
        """Verify mock enhancement integration"""
        try:
            # Check mock model
            bert_path = self.enhanced_dir / "persian_bert"
            if bert_path.exists():
                print("✅ Mock Persian BERT structure available")
                
                # Check model info
                info_file = self.enhanced_dir / "model_info.json"
                if info_file.exists():
                    with open(info_file) as f:
                        info = json.load(f)
                    print(f"✅ Mock model info verified: {info.get('vocab_size', 'unknown')} vocab size")
                    print(f"🎭 Mock mode: {info.get('is_mock', False)}")
                
                # Check config files
                config_file = bert_path / "config.json"
                if config_file.exists():
                    print("✅ Mock config.json available")
                
                vocab_file = bert_path / "vocab.txt"
                if vocab_file.exists():
                    print("✅ Mock vocab.txt available")
                
                return True
            else:
                print("❌ Mock model not found")
                return False
                
        except Exception as e:
            print(f"❌ Mock integration verification failed: {e}")
            return False

def main():
    """Main mock enhancement function"""
    try:
        enhancer = MockEnhancedModelSetup()
        success = enhancer.run_mock_enhancement()
        
        if success:
            print("\n🎉 MOCK ENHANCEMENT COMPLETED!")
            print("✅ Existing system preserved")
            print("✅ Mock functionality added for testing")
            print("📁 Check ai_models/enhanced/ for mock components")
            print("🎭 Ready for integration testing with mock data")
        else:
            print("\n⚠️ Mock enhancement needs attention")
            print("🔄 Existing system remains unchanged")
        
        return success
        
    except Exception as e:
        print(f"❌ Mock enhancement error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)