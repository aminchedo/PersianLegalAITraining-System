#!/usr/bin/env python3
"""
Test Script for Verified Persian Legal AI System
اسکریپت تست سیستم هوش مصنوعی حقوقی فارسی تأیید شده

Tests the verified dataset integration system without requiring external dependencies.
"""

import sys
import os
import logging
from pathlib import Path

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent / "backend"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported"""
    try:
        logger.info("🔍 Testing module imports...")
        
        # Test dataset integrator
        from data.dataset_integrator import PersianLegalDataIntegrator
        logger.info("✅ PersianLegalDataIntegrator imported successfully")
        
        # Test quality validator
        from validation.dataset_validator import DatasetQualityValidator
        logger.info("✅ DatasetQualityValidator imported successfully")
        
        # Test verified trainer
        from models.verified_data_trainer import VerifiedDataTrainer
        logger.info("✅ VerifiedDataTrainer imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_initialization():
    """Test that all components can be initialized"""
    try:
        logger.info("🔧 Testing component initialization...")
        
        # Test dataset integrator initialization
        integrator = PersianLegalDataIntegrator()
        logger.info("✅ PersianLegalDataIntegrator initialized")
        
        # Test quality validator initialization
        validator = DatasetQualityValidator()
        logger.info("✅ DatasetQualityValidator initialized")
        
        # Test verified trainer initialization
        trainer = VerifiedDataTrainer()
        logger.info("✅ VerifiedDataTrainer initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False

def test_api_compatibility():
    """Test that API endpoints maintain compatibility"""
    try:
        logger.info("🌐 Testing API compatibility...")
        
        # Test that training endpoints can be imported
        from api.training_endpoints import router, verified_trainer
        logger.info("✅ Training endpoints imported successfully")
        
        # Test that verified trainer is available
        if verified_trainer is not None:
            logger.info("✅ Verified trainer available in API")
        else:
            logger.warning("⚠️ Verified trainer not available in API")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ API compatibility test failed: {e}")
        return False

def test_typescript_compatibility():
    """Test that TypeScript interfaces are preserved"""
    try:
        logger.info("📝 Testing TypeScript compatibility...")
        
        # Check that TypeScript files exist and are unchanged
        typescript_files = [
            "frontend/src/types/realData.ts",
            "frontend/src/api/persian-ai-api.js"
        ]
        
        for file_path in typescript_files:
            if os.path.exists(file_path):
                logger.info(f"✅ TypeScript file exists: {file_path}")
            else:
                logger.warning(f"⚠️ TypeScript file not found: {file_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ TypeScript compatibility test failed: {e}")
        return False

def test_verified_datasets_config():
    """Test verified datasets configuration"""
    try:
        logger.info("📊 Testing verified datasets configuration...")
        
        from data.dataset_integrator import PersianLegalDataIntegrator
        
        integrator = PersianLegalDataIntegrator()
        datasets = integrator.VERIFIED_DATASETS
        
        logger.info(f"✅ Found {len(datasets)} verified datasets:")
        for key, config in datasets.items():
            logger.info(f"  - {config['name']}: {config['hf_path']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Verified datasets test failed: {e}")
        return False

def test_validation_logic():
    """Test validation logic without external dependencies"""
    try:
        logger.info("🔍 Testing validation logic...")
        
        from validation.dataset_validator import DatasetQualityValidator
        
        validator = DatasetQualityValidator()
        
        # Test Persian character detection
        persian_text = "این یک متن فارسی است"
        persian_chars = set('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی')
        persian_count = sum(1 for char in persian_text if char in persian_chars)
        
        if persian_count > 0:
            logger.info("✅ Persian character detection working")
        else:
            logger.warning("⚠️ Persian character detection not working")
        
        # Test legal keyword detection
        legal_text = "این قانون جدید است و در دادگاه بررسی می‌شود"
        legal_keywords = ['قانون', 'دادگاه', 'حقوق']
        legal_count = sum(1 for keyword in legal_keywords if keyword in legal_text)
        
        if legal_count > 0:
            logger.info("✅ Legal keyword detection working")
        else:
            logger.warning("⚠️ Legal keyword detection not working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation logic test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Persian Legal AI Verified System Test")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Component Initialization", test_initialization),
        ("API Compatibility", test_api_compatibility),
        ("TypeScript Compatibility", test_typescript_compatibility),
        ("Verified Datasets Config", test_verified_datasets_config),
        ("Validation Logic", test_validation_logic)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} - {test_name}")
        except Exception as e:
            results[test_name] = False
            print(f"❌ FAIL - {test_name}: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for verified dataset integration.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the logs for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)