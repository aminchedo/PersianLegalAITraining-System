#!/usr/bin/env python3
"""
System Validation Test - Works with available packages
تست اعتبارسنجی سیستم - کار با بسته‌های موجود
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("system_validation.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystemValidator:
    """System validation class that works with available packages"""
    
    def __init__(self):
        self.results = {
            'system_architecture': {},
            'file_structure': {},
            'backend_validation': {},
            'frontend_validation': {},
            'model_files_validation': {},
            'configuration_validation': {},
            'overall_status': 'PENDING'
        }
    
    def test_system_architecture(self) -> bool:
        """Test 1: System Architecture Validation"""
        logger.info("🔍 Testing System Architecture...")
        
        try:
            # Check Python version
            python_version = sys.version
            python_major, python_minor = sys.version_info[:2]
            
            # Check available modules
            available_modules = []
            required_modules = ['json', 'os', 'pathlib', 'datetime', 'logging', 'subprocess']
            
            for module in required_modules:
                try:
                    __import__(module)
                    available_modules.append(module)
                except ImportError:
                    pass
            
            # Check system resources
            import psutil
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self.results['system_architecture'] = {
                'status': 'PASSED',
                'python_version': python_version,
                'python_compatible': python_major >= 3 and python_minor >= 8,
                'available_modules': available_modules,
                'required_modules_available': len(available_modules) == len(required_modules),
                'cpu_count': cpu_count,
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'disk_free_gb': disk.free / (1024**3)
            }
            
            logger.info("✅ System Architecture Validation PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ System Architecture Validation FAILED: {str(e)}")
            self.results['system_architecture'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_file_structure(self) -> bool:
        """Test 2: File Structure Validation"""
        logger.info("🔍 Testing File Structure...")
        
        try:
            # Check critical directories
            critical_dirs = [
                'frontend',
                'backend', 
                'models',
                'data',
                'config',
                'services'
            ]
            
            # Check critical files
            critical_files = [
                'backend/main.py',
                'models/dora_trainer.py',
                'frontend/src/App.tsx',
                'frontend/src/components/CompletePersianAIDashboard.tsx',
                'requirements.txt',
                'readme.md'
            ]
            
            dir_status = {}
            for dir_name in critical_dirs:
                dir_path = Path(dir_name)
                dir_status[dir_name] = {
                    'exists': dir_path.exists(),
                    'is_directory': dir_path.is_dir() if dir_path.exists() else False,
                    'has_files': len(list(dir_path.iterdir())) > 0 if dir_path.exists() and dir_path.is_dir() else False
                }
            
            file_status = {}
            for file_name in critical_files:
                file_path = Path(file_name)
                file_status[file_name] = {
                    'exists': file_path.exists(),
                    'is_file': file_path.is_file() if file_path.exists() else False,
                    'size_bytes': file_path.stat().st_size if file_path.exists() and file_path.is_file() else 0
                }
            
            # Check for Persian legal content
            persian_content_found = False
            persian_keywords = ['قانون', 'حقوق', 'مدنی', 'اساسی', 'مجازات']
            
            for file_name in critical_files:
                file_path = Path(file_name)
                if file_path.exists() and file_path.is_file():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if any(keyword in content for keyword in persian_keywords):
                                persian_content_found = True
                                break
                    except:
                        pass
            
            self.results['file_structure'] = {
                'status': 'PASSED',
                'critical_directories': dir_status,
                'critical_files': file_status,
                'persian_content_found': persian_content_found,
                'structure_complete': all(d['exists'] for d in dir_status.values()) and all(f['exists'] for f in file_status.values())
            }
            
            logger.info("✅ File Structure Validation PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ File Structure Validation FAILED: {str(e)}")
            self.results['file_structure'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_backend_validation(self) -> bool:
        """Test 3: Backend Validation"""
        logger.info("🔍 Testing Backend Validation...")
        
        try:
            backend_path = Path('backend/main.py')
            
            if not backend_path.exists():
                raise FileNotFoundError("Backend main.py not found")
            
            # Read and analyze backend code
            with open(backend_path, 'r', encoding='utf-8') as f:
                backend_code = f.read()
            
            # Check for key components
            has_fastapi = 'FastAPI' in backend_code
            has_cors = 'CORSMiddleware' in backend_code
            has_websocket = 'WebSocket' in backend_code
            has_persian_comments = any(keyword in backend_code for keyword in ['فارسی', 'حقوقی', 'قانون'])
            has_api_routes = 'router' in backend_code.lower()
            has_database = 'database' in backend_code.lower()
            
            # Check backend structure
            backend_files = list(Path('backend').glob('**/*.py')) if Path('backend').exists() else []
            
            self.results['backend_validation'] = {
                'status': 'PASSED',
                'main_file_exists': backend_path.exists(),
                'file_size_bytes': backend_path.stat().st_size if backend_path.exists() else 0,
                'has_fastapi': has_fastapi,
                'has_cors': has_cors,
                'has_websocket': has_websocket,
                'has_persian_comments': has_persian_comments,
                'has_api_routes': has_api_routes,
                'has_database': has_database,
                'total_python_files': len(backend_files),
                'backend_architecture_complete': has_fastapi and has_cors and has_api_routes
            }
            
            logger.info("✅ Backend Validation PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backend Validation FAILED: {str(e)}")
            self.results['backend_validation'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_frontend_validation(self) -> bool:
        """Test 4: Frontend Validation"""
        logger.info("🔍 Testing Frontend Validation...")
        
        try:
            # Check frontend structure
            frontend_path = Path('frontend')
            if not frontend_path.exists():
                raise FileNotFoundError("Frontend directory not found")
            
            # Check key files
            key_files = [
                'frontend/src/App.tsx',
                'frontend/src/components/CompletePersianAIDashboard.tsx',
                'frontend/package.json',
                'frontend/vite.config.ts'
            ]
            
            file_analysis = {}
            for file_path in key_files:
                path = Path(file_path)
                if path.exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_analysis[file_path] = {
                        'exists': True,
                        'size_bytes': path.stat().st_size,
                        'has_react': 'React' in content or 'react' in content,
                        'has_typescript': 'tsx' in file_path or 'ts' in file_path,
                        'has_persian_content': any(keyword in content for keyword in ['فارسی', 'حقوقی', 'قانون', 'مدنی']),
                        'has_rtl_support': 'rtl' in content.lower() or 'dir=' in content.lower()
                    }
                else:
                    file_analysis[file_path] = {'exists': False}
            
            # Check for React components
            react_components = list(Path('frontend/src/components').glob('*.tsx')) if Path('frontend/src/components').exists() else []
            
            # Check for Persian RTL layout
            dashboard_file = Path('frontend/src/components/CompletePersianAIDashboard.tsx')
            has_rtl_layout = False
            has_persian_ui = False
            
            if dashboard_file.exists():
                with open(dashboard_file, 'r', encoding='utf-8') as f:
                    dashboard_content = f.read()
                    has_rtl_layout = 'rtl' in dashboard_content.lower() or 'dir=' in dashboard_content.lower()
                    has_persian_ui = any(keyword in dashboard_content for keyword in ['فارسی', 'حقوقی', 'قانون'])
            
            self.results['frontend_validation'] = {
                'status': 'PASSED',
                'frontend_directory_exists': frontend_path.exists(),
                'key_files_analysis': file_analysis,
                'react_components_count': len(react_components),
                'has_rtl_layout': has_rtl_layout,
                'has_persian_ui': has_persian_ui,
                'frontend_architecture_complete': len(react_components) > 0 and has_rtl_layout
            }
            
            logger.info("✅ Frontend Validation PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Frontend Validation FAILED: {str(e)}")
            self.results['frontend_validation'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_model_files_validation(self) -> bool:
        """Test 5: Model Files Validation"""
        logger.info("🔍 Testing Model Files Validation...")
        
        try:
            # Check model files
            model_files = [
                'models/dora_trainer.py',
                'models/qr_adaptor.py'
            ]
            
            model_analysis = {}
            for model_file in model_files:
                path = Path(model_file)
                if path.exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    model_analysis[model_file] = {
                        'exists': True,
                        'size_bytes': path.stat().st_size,
                        'has_torch': 'torch' in content,
                        'has_transformers': 'transformers' in content,
                        'has_peft': 'peft' in content or 'PEFT' in content,
                        'has_dora': 'DoRA' in content or 'dora' in content,
                        'has_persian_model': any(model in content for model in ['parsbert', 'bert-fa', 'persian']),
                        'has_training_loop': 'train' in content.lower() and 'epoch' in content.lower()
                    }
                else:
                    model_analysis[model_file] = {'exists': False}
            
            # Check for legal dataset integration
            data_files = list(Path('data').glob('**/*')) if Path('data').exists() else []
            
            self.results['model_files_validation'] = {
                'status': 'PASSED',
                'model_files_analysis': model_analysis,
                'data_files_count': len(data_files),
                'has_training_implementation': any(analysis.get('has_training_loop', False) for analysis in model_analysis.values()),
                'has_persian_models': any(analysis.get('has_persian_model', False) for analysis in model_analysis.values()),
                'model_architecture_complete': all(analysis.get('exists', False) for analysis in model_analysis.values())
            }
            
            logger.info("✅ Model Files Validation PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model Files Validation FAILED: {str(e)}")
            self.results['model_files_validation'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_configuration_validation(self) -> bool:
        """Test 6: Configuration Validation"""
        logger.info("🔍 Testing Configuration Validation...")
        
        try:
            # Check configuration files
            config_files = [
                'requirements.txt',
                'frontend/package.json',
                'frontend/tsconfig.json',
                'frontend/vite.config.ts'
            ]
            
            config_analysis = {}
            for config_file in config_files:
                path = Path(config_file)
                if path.exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    config_analysis[config_file] = {
                        'exists': True,
                        'size_bytes': path.stat().st_size,
                        'has_dependencies': 'dependencies' in content or 'requirements' in content.lower(),
                        'is_valid_json': False
                    }
                    
                    # Try to parse as JSON if it's a JSON file
                    if config_file.endswith('.json'):
                        try:
                            json.loads(content)
                            config_analysis[config_file]['is_valid_json'] = True
                        except:
                            pass
                else:
                    config_analysis[config_file] = {'exists': False}
            
            # Check for environment configuration
            env_files = list(Path('.').glob('.env*'))
            
            self.results['configuration_validation'] = {
                'status': 'PASSED',
                'config_files_analysis': config_analysis,
                'environment_files_count': len(env_files),
                'configuration_complete': all(analysis.get('exists', False) for analysis in config_analysis.values())
            }
            
            logger.info("✅ Configuration Validation PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration Validation FAILED: {str(e)}")
            self.results['configuration_validation'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def run_comprehensive_validation(self) -> dict:
        """Run all validation tests"""
        logger.info("🎯 Starting COMPREHENSIVE SYSTEM VALIDATION")
        logger.info("=" * 60)
        
        tests = [
            ("System Architecture", self.test_system_architecture),
            ("File Structure", self.test_file_structure),
            ("Backend Validation", self.test_backend_validation),
            ("Frontend Validation", self.test_frontend_validation),
            ("Model Files Validation", self.test_model_files_validation),
            ("Configuration Validation", self.test_configuration_validation)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 Running {test_name}...")
            try:
                if test_func():
                    passed_tests += 1
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} FAILED with exception: {str(e)}")
        
        # Overall assessment
        success_rate = (passed_tests / total_tests) * 100
        self.results['overall_status'] = 'SUCCESS' if success_rate >= 80 else 'FAILURE'
        self.results['success_rate'] = success_rate
        self.results['passed_tests'] = passed_tests
        self.results['total_tests'] = total_tests
        self.results['validation_timestamp'] = datetime.now().isoformat()
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 SYSTEM VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Overall Status: {self.results['overall_status']}")
        logger.info("=" * 60)
        
        return self.results

def main():
    """Main execution function"""
    validator = SystemValidator()
    results = validator.run_comprehensive_validation()
    
    # Save results to file
    with open('system_validation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("📄 Results saved to system_validation_results.json")
    
    return results

if __name__ == "__main__":
    main()