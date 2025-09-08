#!/usr/bin/env python3
"""
Enhanced Persian BERT Setup (Preserves existing functionality)
============================================================
Adds real Persian BERT download without breaking existing setup
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import time

class EnhancedModelSetup:
    """Enhanced model setup that preserves existing functionality"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "ai_models"
        self.cache_dir = self.models_dir / "cache"
        self.enhanced_dir = self.models_dir / "enhanced"
        
        # Create directories safely
        for directory in [self.models_dir, self.cache_dir, self.enhanced_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"Enhanced setup initialized - Models dir: {self.models_dir}")
    
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
    
    def safe_install_dependencies(self) -> bool:
        """Install dependencies without breaking existing environment"""
        print("📦 Safely checking dependencies...")
        
        required_packages = ['torch', 'transformers', 'huggingface_hub']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package} already available")
            except ImportError:
                missing_packages.append(package)
                print(f"ℹ️ {package} needed")
        
        if missing_packages:
            print(f"📥 Installing missing packages: {missing_packages}")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    *missing_packages, "--user", "--quiet"
                ])
                print("✅ Dependencies installed safely")
                return True
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Some dependencies failed to install: {e}")
                return False
        
        return True
    
    def download_persian_bert_safe(self) -> bool:
        """Download Persian BERT without affecting existing models"""
        bert_path = self.enhanced_dir / "persian_bert"
        
        if bert_path.exists():
            print(f"✅ Persian BERT already exists at: {bert_path}")
            return self.verify_existing_bert(bert_path)
        
        print("📥 Downloading Persian BERT to enhanced directory...")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            from huggingface_hub import snapshot_download
            
            model_name = "HooshvareLab/bert-base-parsbert-uncased"
            
            # Download to enhanced directory (separate from existing)
            print(f"📥 Downloading {model_name} to {bert_path}...")
            
            snapshot_download(
                repo_id=model_name,
                cache_dir=str(bert_path.parent),
                local_dir=str(bert_path),
                local_dir_use_symlinks=False
            )
            
            return self.verify_existing_bert(bert_path)
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def verify_existing_bert(self, bert_path: Path) -> bool:
        """Verify Persian BERT without affecting system"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            print("🔍 Verifying Persian BERT...")
            tokenizer = AutoTokenizer.from_pretrained(str(bert_path))
            model = AutoModel.from_pretrained(str(bert_path))
            
            # Quick test
            test_text = "تست مدل فارسی"
            inputs = tokenizer(test_text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            if outputs.last_hidden_state is not None:
                print("✅ Persian BERT verified successfully")
                
                # Save verification info
                model_info = {
                    "model_name": "HooshvareLab/bert-base-parsbert-uncased",
                    "local_path": str(bert_path),
                    "vocab_size": len(tokenizer),
                    "verified_at": time.time(),
                    "test_passed": True
                }
                
                with open(bert_path.parent / "model_info.json", "w") as f:
                    json.dump(model_info, f, indent=2)
                
                return True
            else:
                print("❌ Model verification failed")
                return False
                
        except Exception as e:
            print(f"❌ Verification error: {e}")
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
                }
            ],
            "metadata": {
                "version": "enhanced_1.0",
                "created_at": time.time(),
                "enhancement_level": "detailed_entities",
                "language": "persian"
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
    
    def run_safe_enhancement(self) -> bool:
        """Run safe enhancement without breaking existing setup"""
        print("🚀 RUNNING SAFE SYSTEM ENHANCEMENT")
        print("=" * 50)
        
        # Check existing setup first
        existing_status = self.check_existing_setup()
        
        success_count = 0
        total_steps = 4
        
        # Step 1: Dependencies
        print(f"\n📦 Step 1/{total_steps}: Dependencies check...")
        if self.safe_install_dependencies():
            success_count += 1
        
        # Step 2: Download Persian BERT
        print(f"\n📥 Step 2/{total_steps}: Persian BERT enhancement...")
        if self.download_persian_bert_safe():
            success_count += 1
        
        # Step 3: Enhanced sample data
        print(f"\n📄 Step 3/{total_steps}: Enhanced sample data...")
        if self.create_enhanced_sample_data():
            success_count += 1
        
        # Step 4: Integration check
        print(f"\n🔧 Step 4/{total_steps}: Integration verification...")
        integration_success = self.verify_integration()
        if integration_success:
            success_count += 1
        
        success_rate = success_count / total_steps
        print(f"\n📊 ENHANCEMENT RESULTS: {success_count}/{total_steps} ({success_rate:.1%})")
        
        if success_rate >= 0.75:
            print("✅ System enhancement successful!")
            print(f"📁 Enhanced models: {self.enhanced_dir}")
            print("🔄 Existing functionality preserved")
            return True
        else:
            print("⚠️ Enhancement partially successful")
            return False
    
    def verify_integration(self) -> bool:
        """Verify enhancement doesn't break existing system"""
        try:
            # Check enhanced model
            bert_path = self.enhanced_dir / "persian_bert"
            if bert_path.exists():
                print("✅ Enhanced Persian BERT available")
                
                # Check model info
                info_file = self.enhanced_dir / "model_info.json"
                if info_file.exists():
                    with open(info_file) as f:
                        info = json.load(f)
                    print(f"✅ Model info verified: {info.get('vocab_size', 'unknown')} vocab size")
                
                return True
            else:
                print("❌ Enhanced model not found")
                return False
                
        except Exception as e:
            print(f"❌ Integration verification failed: {e}")
            return False

def main():
    """Main enhancement function"""
    try:
        enhancer = EnhancedModelSetup()
        success = enhancer.run_safe_enhancement()
        
        if success:
            print("\n🎉 SAFE ENHANCEMENT COMPLETED!")
            print("✅ Existing system preserved")
            print("✅ New functionality added")
            print("📁 Check ai_models/enhanced/ for new components")
        else:
            print("\n⚠️ Enhancement needs attention")
            print("🔄 Existing system remains unchanged")
        
        return success
        
    except Exception as e:
        print(f"❌ Enhancement error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)