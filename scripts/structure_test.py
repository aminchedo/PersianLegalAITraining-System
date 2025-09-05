#!/usr/bin/env python3
"""
Structure Test for Rapid Training System
Tests the file structure and basic Python syntax
"""

import os
import sys
import ast
from pathlib import Path

def test_file_structure():
    """Test if all required files exist"""
    print("🧪 Testing File Structure...")
    
    required_files = [
        "backend/data/dataset_integration.py",
        "backend/data/__init__.py",
        "backend/models/enhanced_dora_trainer.py",
        "backend/validation/dataset_validator.py",
        "backend/validation/__init__.py",
        "backend/services/rapid_trainer.py",
        "scripts/rapid_training_launcher.py",
        "scripts/test_rapid_training.py",
        "config/rapid_training_config.json",
        "RAPID_TRAINING_GUIDE.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print(f"✅ All {len(required_files)} required files exist")
    return True

def test_python_syntax():
    """Test Python syntax of key files"""
    print("\n🧪 Testing Python Syntax...")
    
    python_files = [
        "backend/data/dataset_integration.py",
        "backend/validation/dataset_validator.py",
        "backend/services/rapid_trainer.py",
        "scripts/rapid_training_launcher.py"
    ]
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST to check syntax
            ast.parse(content)
            print(f"✅ {file_path} - syntax OK")
            
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
            print(f"❌ {file_path} - syntax error: {e}")
        except Exception as e:
            syntax_errors.append(f"{file_path}: {e}")
            print(f"❌ {file_path} - error: {e}")
    
    if syntax_errors:
        print(f"❌ Syntax errors found: {syntax_errors}")
        return False
    
    print(f"✅ All {len(python_files)} Python files have valid syntax")
    return True

def test_json_config():
    """Test JSON configuration file"""
    print("\n🧪 Testing JSON Configuration...")
    
    config_file = "config/rapid_training_config.json"
    try:
        import json
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Check required sections
        required_sections = ['model_config', 'training_config', 'dataset_config', 'output_config']
        missing_sections = []
        
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
            else:
                print(f"✅ {section} section found")
        
        if missing_sections:
            print(f"❌ Missing config sections: {missing_sections}")
            return False
        
        print(f"✅ JSON configuration is valid with {len(config)} sections")
        return True
        
    except Exception as e:
        print(f"❌ JSON config error: {e}")
        return False

def test_import_structure():
    """Test import structure without actually importing"""
    print("\n🧪 Testing Import Structure...")
    
    # Check if files have proper import statements
    files_to_check = [
        "backend/data/dataset_integration.py",
        "backend/validation/dataset_validator.py",
        "backend/services/rapid_trainer.py"
    ]
    
    import_issues = []
    for file_path in files_to_check:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for basic import patterns
            if 'from typing import' in content:
                print(f"✅ {file_path} - has typing imports")
            else:
                import_issues.append(f"{file_path} - missing typing imports")
            
            if 'import logging' in content:
                print(f"✅ {file_path} - has logging")
            else:
                import_issues.append(f"{file_path} - missing logging")
                
        except Exception as e:
            import_issues.append(f"{file_path} - error: {e}")
    
    if import_issues:
        print(f"❌ Import issues: {import_issues}")
        return False
    
    print(f"✅ All {len(files_to_check)} files have proper import structure")
    return True

def test_documentation():
    """Test documentation completeness"""
    print("\n🧪 Testing Documentation...")
    
    # Check if README exists and has content
    readme_file = "RAPID_TRAINING_GUIDE.md"
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if len(content) > 1000:  # Should be substantial
            print(f"✅ {readme_file} - comprehensive documentation")
            return True
        else:
            print(f"❌ {readme_file} - too short")
            return False
    else:
        print(f"❌ {readme_file} - missing")
        return False

def main():
    """Main test function"""
    print("🚀 Rapid Training System Structure Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_python_syntax),
        ("JSON Configuration", test_json_config),
        ("Import Structure", test_import_structure),
        ("Documentation", test_documentation)
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
        print("🎉 All structure tests passed! System is properly organized.")
        print("\n📋 Next Steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run full tests: python3 scripts/test_rapid_training.py")
        print("  3. Start training: python3 scripts/rapid_training_launcher.py")
        return True
    else:
        print("⚠️  Some structure tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)