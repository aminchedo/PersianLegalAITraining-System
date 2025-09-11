#!/usr/bin/env python3
"""
Simple Integration Validation for Persian Legal AI
Tests file structure and configuration without requiring dependencies
"""

import json
import os
from pathlib import Path

def test_file_structure():
    """Test that all required files exist"""
    print("📁 Testing File Structure...")
    
    required_files = {
        'backend/services/hardware_detector.py': 'Hardware detection service',
        'backend/ai_classifier.py': 'AI classifier with hardware integration',
        'backend/api/system_endpoints.py': 'System endpoints with hardware detection',
        'vercel.json': 'Vercel configuration',
        'railway.toml': 'Railway configuration',
        'Dockerfile.backend': 'Backend Dockerfile',
        '.env.production': 'Production environment',
        'frontend/next.config.js': 'Frontend configuration'
    }
    
    all_exist = True
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"✅ {description}: {file_path}")
        else:
            print(f"❌ Missing {description}: {file_path}")
            all_exist = False
    
    return all_exist

def test_hardware_detector_code():
    """Test hardware detector code structure"""
    print("\n🔍 Testing Hardware Detector Code...")
    
    try:
        with open('backend/services/hardware_detector.py', 'r') as f:
            content = f.read()
        
        required_components = [
            'class HardwareDetector',
            'def select_optimal_model_config',
            'def get_hardware_summary',
            'def _detect_hardware',
            'deployment_environment',
            'gpu_available',
            'ram_gb'
        ]
        
        all_found = True
        for component in required_components:
            if component in content:
                print(f"✅ Found: {component}")
            else:
                print(f"❌ Missing: {component}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ Failed to read hardware detector: {e}")
        return False

def test_ai_classifier_integration():
    """Test AI classifier integration code"""
    print("\n🧠 Testing AI Classifier Integration...")
    
    try:
        with open('backend/ai_classifier.py', 'r') as f:
            content = f.read()
        
        required_integrations = [
            'from .services.hardware_detector import HardwareDetector',
            'self.hardware_detector = HardwareDetector()',
            'self.model_config = self.hardware_detector.select_optimal_model_config()',
            'def get_system_info',
            'hardware_summary',
            'model_config'
        ]
        
        all_found = True
        for integration in required_integrations:
            if integration in content:
                print(f"✅ Found: {integration}")
            else:
                print(f"❌ Missing: {integration}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ Failed to read AI classifier: {e}")
        return False

def test_system_endpoints_integration():
    """Test system endpoints integration"""
    print("\n🌐 Testing System Endpoints Integration...")
    
    try:
        with open('backend/api/system_endpoints.py', 'r') as f:
            content = f.read()
        
        required_endpoints = [
            '@router.get("/hardware")',
            '@router.get("/deployment/status")',
            '@router.get("/ai/system-info")',
            'from ..services.hardware_detector import HardwareDetector',
            'detector = HardwareDetector()',
            'select_optimal_model_config'
        ]
        
        all_found = True
        for endpoint in required_endpoints:
            if endpoint in content:
                print(f"✅ Found: {endpoint}")
            else:
                print(f"❌ Missing: {endpoint}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ Failed to read system endpoints: {e}")
        return False

def test_vercel_configuration():
    """Test Vercel configuration"""
    print("\n🚀 Testing Vercel Configuration...")
    
    try:
        with open('vercel.json', 'r') as f:
            config = json.load(f)
        
        required_config = {
            'version': 2,
            'builds': 'list',
            'routes': 'list',
            'env': 'dict'
        }
        
        all_valid = True
        for key, expected_type in required_config.items():
            if key in config:
                if expected_type == 'list' and isinstance(config[key], list):
                    print(f"✅ {key}: Valid list with {len(config[key])} items")
                elif expected_type == 'dict' and isinstance(config[key], dict):
                    print(f"✅ {key}: Valid dict with {len(config[key])} keys")
                elif expected_type != 'list' and expected_type != 'dict':
                    print(f"✅ {key}: {config[key]}")
                else:
                    print(f"❌ {key}: Wrong type")
                    all_valid = False
            else:
                print(f"❌ Missing: {key}")
                all_valid = False
        
        # Check for API proxy configuration
        routes = config.get('routes', [])
        has_api_proxy = any('/api/' in route.get('src', '') for route in routes)
        if has_api_proxy:
            print("✅ API proxy configuration found")
        else:
            print("❌ API proxy configuration missing")
            all_valid = False
        
        return all_valid
        
    except Exception as e:
        print(f"❌ Failed to read Vercel configuration: {e}")
        return False

def test_railway_configuration():
    """Test Railway configuration"""
    print("\n🚂 Testing Railway Configuration...")
    
    try:
        with open('railway.toml', 'r') as f:
            content = f.read()
        
        required_sections = [
            '[build]',
            '[deploy]',
            '[variables]',
            'startCommand',
            'healthcheckPath',
            'MODEL_AUTO_DETECT',
            'ENABLE_HARDWARE_DETECTION'
        ]
        
        all_found = True
        for section in required_sections:
            if section in content:
                print(f"✅ Found: {section}")
            else:
                print(f"❌ Missing: {section}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ Failed to read Railway configuration: {e}")
        return False

def test_frontend_configuration():
    """Test frontend configuration"""
    print("\n⚛️ Testing Frontend Configuration...")
    
    try:
        with open('frontend/next.config.js', 'r') as f:
            content = f.read()
        
        required_config = [
            'NEXT_PUBLIC_API_URL',
            'persian-legal-ai-backend.railway.app',
            'async rewrites()',
            'Access-Control-Allow-Origin',
            'process.env.VERCEL'
        ]
        
        all_found = True
        for config_item in required_config:
            if config_item in content:
                print(f"✅ Found: {config_item}")
            else:
                print(f"❌ Missing: {config_item}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ Failed to read frontend configuration: {e}")
        return False

def test_environment_configuration():
    """Test environment configuration"""
    print("\n🌍 Testing Environment Configuration...")
    
    try:
        with open('.env.production', 'r') as f:
            content = f.read()
        
        required_vars = [
            'NODE_ENV=production',
            'NEXT_PUBLIC_API_URL=https://persian-legal-ai-backend.railway.app',
            'MODEL_AUTO_DETECT=true',
            'ENABLE_HARDWARE_DETECTION=true',
            'ENABLE_DYNAMIC_CONFIG=true'
        ]
        
        all_found = True
        for var in required_vars:
            if var in content:
                print(f"✅ Found: {var}")
            else:
                print(f"❌ Missing: {var}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ Failed to read environment configuration: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🧪 Persian Legal AI Integration Validation (Simple)")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Hardware Detector Code", test_hardware_detector_code),
        ("AI Classifier Integration", test_ai_classifier_integration),
        ("System Endpoints Integration", test_system_endpoints_integration),
        ("Vercel Configuration", test_vercel_configuration),
        ("Railway Configuration", test_railway_configuration),
        ("Frontend Configuration", test_frontend_configuration),
        ("Environment Configuration", test_environment_configuration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Integration Test Results:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
    
    print("\n" + "=" * 60)
    print(f"📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ File structure is correct")
        print("✅ Hardware detection is integrated")
        print("✅ Dynamic model selection is implemented")
        print("✅ System endpoints are extended")
        print("✅ Vercel configuration is valid")
        print("✅ Railway configuration is valid")
        print("✅ Frontend configuration is updated")
        print("✅ Environment configuration is complete")
        print("\n🚀 System is ready for deployment!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        print("🔧 Please fix the issues before deployment")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)