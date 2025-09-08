#!/usr/bin/env python3
"""
Simple Test for Rapid Training System Components
"""

import sys
import os

def test_imports():
    """Test if we can import the modules"""
    print("🧪 Testing Module Imports...")
    
    try:
        # Add backend to path
        backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
        sys.path.insert(0, backend_path)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        # Test dataset integration
        from backend.data.dataset_integration import PersianLegalDataIntegrator
        print("✅ Dataset integration module imported successfully")
        
        # Test quality validator
        from backend.validation.dataset_validator import DatasetQualityValidator
        print("✅ Quality validator module imported successfully")
        
        # Test enhanced trainer
        from backend.models.enhanced_dora_trainer import DataEnhancedDoraTrainer
        print("✅ Enhanced trainer module imported successfully")
        
        # Test rapid trainer
        from backend.services.rapid_trainer import RapidTrainingOrchestrator
        print("✅ Rapid trainer module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without heavy dependencies"""
    print("\n🧪 Testing Basic Functionality...")
    
    try:
        from backend.data.dataset_integration import PersianLegalDataIntegrator
        
        # Test integrator initialization
        integrator = PersianLegalDataIntegrator()
        print("✅ Dataset integrator initialized")
        
        # Test dataset registry
        registry = integrator.DATASET_REGISTRY
        print(f"✅ Found {len(registry)} registered datasets")
        
        # Test quality validator
        from backend.validation.dataset_validator import DatasetQualityValidator
        validator = DatasetQualityValidator()
        print("✅ Quality validator initialized")
        
        # Test legal keywords
        keywords = validator.legal_keywords
        print(f"✅ Found {len(keywords)} legal keywords")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Simple Rapid Training System Test")
    print("=" * 40)
    
    tests = [
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print("=" * 40)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Basic tests passed! System components are accessible.")
        return True
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)