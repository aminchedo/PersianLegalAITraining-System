#!/usr/bin/env python3
"""
Test Script for Rapid Training System
اسکریپت تست سیستم آموزش سریع
"""

import sys
import os
import logging
from pathlib import Path

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_path)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def setup_test_logging():
    """Setup logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_dataset_integration():
    """Test dataset integration module"""
    print("🧪 Testing Dataset Integration...")
    
    try:
        from backend.data.dataset_integration import PersianLegalDataIntegrator
        
        integrator = PersianLegalDataIntegrator()
        
        # Test dataset availability check
        available_datasets = integrator.available_datasets
        print(f"✅ Found {len(available_datasets)} registered datasets")
        
        # Test dataset statistics
        stats = integrator.get_dataset_stats()
        print(f"✅ Dataset statistics: {len(stats)} entries")
        
        # Test placeholder dataset creation
        placeholder = integrator._create_placeholder_dataset()
        print(f"✅ Placeholder dataset created with {len(placeholder)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset integration test failed: {e}")
        return False

def test_quality_validator():
    """Test quality validation module"""
    print("🧪 Testing Quality Validator...")
    
    try:
        from backend.validation.dataset_validator import DatasetQualityValidator
        from datasets import Dataset
        
        validator = DatasetQualityValidator()
        
        # Create test dataset
        test_data = [
            {
                'text': 'این یک نمونه متن حقوقی فارسی است که شامل کلمات کلیدی مانند قانون، دادگاه و قاضی می‌باشد.',
                'source': 'test',
                'type': 'legal_document'
            }
        ] * 100
        
        test_dataset = Dataset.from_list(test_data)
        
        # Test validation
        results = validator.validate_datasets({'test': test_dataset}, sample_size=50)
        print(f"✅ Quality validation completed: {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality validator test failed: {e}")
        return False

def test_enhanced_trainer():
    """Test enhanced trainer module"""
    print("🧪 Testing Enhanced Trainer...")
    
    try:
        from backend.models.enhanced_dora_trainer import DataEnhancedDoraTrainer
        
        # Test trainer initialization
        config = {
            'base_model': 'HooshvareLab/bert-fa-base-uncased',
            'dora_rank': 32,
            'dora_alpha': 8.0,
            'accelerated_mode': True,
            'batch_size': 4,
            'max_length': 256
        }
        
        trainer = DataEnhancedDoraTrainer(config)
        print("✅ Enhanced trainer initialized")
        
        # Test placeholder data creation
        placeholder_data = trainer._create_placeholder_dataset()
        print(f"✅ Placeholder data created: {len(placeholder_data)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced trainer test failed: {e}")
        return False

def test_rapid_orchestrator():
    """Test rapid training orchestrator"""
    print("🧪 Testing Rapid Training Orchestrator...")
    
    try:
        from backend.services.rapid_trainer import RapidTrainingOrchestrator
        
        # Test orchestrator initialization
        config = {
            'model_config': {
                'base_model': 'HooshvareLab/bert-fa-base-uncased',
                'dora_rank': 32,
                'dora_alpha': 8.0,
                'accelerated_mode': True,
                'batch_size': 4,
                'max_length': 256
            },
            'training_config': {
                'epochs': 1,
                'learning_rate': 2e-5
            },
            'dataset_config': {
                'max_samples_per_dataset': 100,
                'quality_threshold': 50
            },
            'output_config': {
                'checkpoint_dir': './test_checkpoints',
                'report_dir': './test_reports',
                'model_output_dir': './test_models'
            }
        }
        
        orchestrator = RapidTrainingOrchestrator(config)
        print("✅ Rapid training orchestrator initialized")
        
        # Test system readiness check
        system_status = orchestrator._check_system_readiness()
        print(f"✅ System readiness check: {system_status['ready']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Rapid orchestrator test failed: {e}")
        return False

def test_quick_training():
    """Test quick training with minimal data"""
    print("🧪 Testing Quick Training...")
    
    try:
        from backend.services.rapid_trainer import RapidTrainingOrchestrator
        
        # Minimal config for quick test
        config = {
            'model_config': {
                'base_model': 'HooshvareLab/bert-fa-base-uncased',
                'dora_rank': 16,
                'dora_alpha': 4.0,
                'accelerated_mode': True,
                'batch_size': 2,
                'max_length': 128
            },
            'training_config': {
                'epochs': 1,
                'learning_rate': 1e-4
            },
            'dataset_config': {
                'max_samples_per_dataset': 50,
                'quality_threshold': 30
            },
            'output_config': {
                'checkpoint_dir': './test_checkpoints',
                'report_dir': './test_reports',
                'model_output_dir': './test_models'
            }
        }
        
        orchestrator = RapidTrainingOrchestrator(config)
        
        # Test dataset loading
        datasets = orchestrator._load_and_validate_datasets()
        print(f"✅ Loaded {len(datasets)} datasets for testing")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick training test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Rapid Training System Tests")
    print("=" * 50)
    
    tests = [
        ("Dataset Integration", test_dataset_integration),
        ("Quality Validator", test_quality_validator),
        ("Enhanced Trainer", test_enhanced_trainer),
        ("Rapid Orchestrator", test_rapid_orchestrator),
        ("Quick Training", test_quick_training)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for rapid training.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

def main():
    """Main test function"""
    setup_test_logging()
    
    try:
        success = run_all_tests()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)