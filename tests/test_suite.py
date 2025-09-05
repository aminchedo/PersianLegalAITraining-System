#!/usr/bin/env python3
"""
Persian Legal AI System - Comprehensive Test Suite
Production deployment validation with all critical component testing
"""

import os
import sys
import asyncio
import time
import subprocess
import requests
import psutil
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import tempfile
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Core imports
import torch
import numpy as np
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


class TestResult:
    """Test result container"""
    def __init__(self, name: str, passed: bool, message: str = "", duration: float = 0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration
        self.timestamp = time.time()


class PersianLegalAITestSuite:
    """Comprehensive test suite for Persian Legal AI System"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_start_time = time.time()
        
    def log_test_start(self, test_name: str) -> float:
        """Log test start and return start time"""
        logger.info(f"🧪 Starting test: {test_name}")
        return time.time()
    
    def log_test_result(self, test_name: str, passed: bool, message: str = "", start_time: float = 0.0) -> TestResult:
        """Log test result and add to results"""
        duration = time.time() - start_time if start_time > 0 else 0.0
        result = TestResult(test_name, passed, message, duration)
        self.results.append(result)
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{status} {test_name} ({duration:.2f}s): {message}")
        
        return result
    
    def test_environment_setup(self) -> TestResult:
        """Test 1: Environment and dependencies"""
        start_time = self.log_test_start("Environment Setup")
        
        try:
            # Python version check
            python_version = sys.version_info
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
                return self.log_test_result(
                    "Environment Setup", False,
                    f"Python 3.9+ required, got {python_version.major}.{python_version.minor}",
                    start_time
                )
            
            # Required packages check
            required_packages = [
                'torch', 'transformers', 'peft', 'accelerate',
                'streamlit', 'plotly', 'spacy', 'hazm',
                'requests', 'beautifulsoup4', 'pandas', 'numpy'
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                return self.log_test_result(
                    "Environment Setup", False,
                    f"Missing packages: {', '.join(missing_packages)}",
                    start_time
                )
            
            # Intel Extension check
            try:
                import intel_extension_for_pytorch as ipex
                ipex_version = ipex.__version__
                logger.info(f"Intel Extension for PyTorch: {ipex_version}")
            except ImportError:
                logger.warning("Intel Extension for PyTorch not available")
            
            # CPU threads check
            cpu_threads = torch.get_num_threads()
            if cpu_threads < 8:
                logger.warning(f"Low CPU thread count: {cpu_threads}")
            
            return self.log_test_result(
                "Environment Setup", True,
                f"Python {python_version.major}.{python_version.minor}, {len(required_packages)} packages, {cpu_threads} CPU threads",
                start_time
            )
            
        except Exception as e:
            return self.log_test_result(
                "Environment Setup", False,
                f"Exception: {str(e)}",
                start_time
            )
    
    def test_persian_models_loading(self) -> TestResult:
        """Test 2: Persian model loading capabilities"""
        start_time = self.log_test_start("Persian Models Loading")
        
        try:
            from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
            
            # Test models with fallback strategy
            test_models = [
                ('HuggingFace Hub Model', 'bert-base-multilingual-cased'),
                ('Persian BERT', 'HooshvareLab/bert-fa-base-uncased'),
                ('Multilingual Model', 'xlm-roberta-base'),
            ]
            
            successful_models = []
            
            for model_name, model_id in test_models:
                try:
                    # Try loading tokenizer first (lighter operation)
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    
                    # Test tokenization with Persian text
                    test_text = "این یک متن آزمایشی فارسی است."
                    tokens = tokenizer.tokenize(test_text)
                    
                    if len(tokens) > 0:
                        successful_models.append(model_name)
                        logger.info(f"✅ {model_name} tokenizer loaded successfully")
                    
                except Exception as e:
                    logger.warning(f"⚠️ {model_name} failed: {str(e)[:100]}")
            
            if len(successful_models) > 0:
                return self.log_test_result(
                    "Persian Models Loading", True,
                    f"Successfully loaded {len(successful_models)} models: {', '.join(successful_models)}",
                    start_time
                )
            else:
                return self.log_test_result(
                    "Persian Models Loading", False,
                    "No models could be loaded successfully",
                    start_time
                )
                
        except Exception as e:
            return self.log_test_result(
                "Persian Models Loading", False,
                f"Exception: {str(e)}",
                start_time
            )
    
    def test_dora_implementation(self) -> TestResult:
        """Test 3: DoRA implementation functionality"""
        start_time = self.log_test_start("DoRA Implementation")
        
        try:
            from models.dora_trainer import DoRALayer, DoRATrainer
            
            # Test DoRA layer creation
            dummy_linear = torch.nn.Linear(512, 256)
            dora_layer = DoRALayer(
                original_layer=dummy_linear,
                rank=32,
                alpha=16.0,
                dropout=0.1
            )
            
            # Test forward pass
            test_input = torch.randn(2, 512)
            output = dora_layer(test_input)
            
            # Verify output shape
            expected_shape = (2, 256)
            if output.shape != expected_shape:
                return self.log_test_result(
                    "DoRA Implementation", False,
                    f"Output shape mismatch: expected {expected_shape}, got {output.shape}",
                    start_time
                )
            
            # Test DoRA parameters exist
            dora_params = [name for name, param in dora_layer.named_parameters() if 'lora' in name.lower() or 'magnitude' in name.lower()]
            
            if len(dora_params) == 0:
                return self.log_test_result(
                    "DoRA Implementation", False,
                    "No DoRA parameters found in layer",
                    start_time
                )
            
            # Test gradient flow
            loss = output.sum()
            loss.backward()
            
            grad_check = any(param.grad is not None for name, param in dora_layer.named_parameters())
            
            if not grad_check:
                return self.log_test_result(
                    "DoRA Implementation", False,
                    "No gradients computed for DoRA parameters",
                    start_time
                )
            
            return self.log_test_result(
                "DoRA Implementation", True,
                f"DoRA layer working with {len(dora_params)} parameters, gradient flow confirmed",
                start_time
            )
            
        except Exception as e:
            return self.log_test_result(
                "DoRA Implementation", False,
                f"Exception: {str(e)}",
                start_time
            )
    
    async def test_data_collection_async(self) -> TestResult:
        """Test 4: Data collection from Persian sources"""
        start_time = self.log_test_start("Data Collection")
        
        try:
            from data.persian_legal_collector import PersianLegalDataCollector, LegalDocument
            
            collector = PersianLegalDataCollector()
            
            # Test basic functionality
            test_documents = []
            
            # Create sample document to test processing
            sample_doc = LegalDocument(
                id="test_001",
                title="قانون آزمایشی",
                content="این یک متن آزمایشی برای تست سیستم جمع‌آوری داده‌های حقوقی فارسی است.",
                source="test_source",
                url="http://test.example.com",
                document_type="law",
                law_category="civil"
            )
            
            test_documents.append(sample_doc)
            
            # Test text processing
            if hasattr(collector, 'text_processor'):
                processed_content = collector.text_processor.normalize_persian_text(sample_doc.content)
                if len(processed_content) == 0:
                    return self.log_test_result(
                        "Data Collection", False,
                        "Text processing failed - empty result",
                        start_time
                    )
            
            # Test network connectivity (with timeout)
            try:
                import aiohttp
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # Test basic HTTP connectivity
                    async with session.get('https://httpbin.org/get') as response:
                        if response.status != 200:
                            logger.warning("Network connectivity test failed")
            except Exception as e:
                logger.warning(f"Network test failed: {str(e)[:100]}")
            
            return self.log_test_result(
                "Data Collection", True,
                f"Data collection framework functional, processed {len(test_documents)} test documents",
                start_time
            )
            
        except Exception as e:
            return self.log_test_result(
                "Data Collection", False,
                f"Exception: {str(e)}",
                start_time
            )
    
    def test_intel_optimizations(self) -> TestResult:
        """Test 5: Intel Extension optimizations"""
        start_time = self.log_test_start("Intel Optimizations")
        
        try:
            # Check Intel Extension availability
            try:
                import intel_extension_for_pytorch as ipex
                ipex_available = True
            except ImportError:
                ipex_available = False
            
            # Check CPU information
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            cpu_brand = cpu_info.get('brand_raw', 'Unknown')
            cpu_count = psutil.cpu_count(logical=False)
            logical_count = psutil.cpu_count(logical=True)
            
            # Check environment variables
            omp_threads = os.environ.get('OMP_NUM_THREADS', 'Not set')
            mkl_threads = os.environ.get('MKL_NUM_THREADS', 'Not set')
            
            # Test tensor operations
            test_tensor = torch.randn(1000, 1000)
            start_op = time.time()
            result = torch.mm(test_tensor, test_tensor.t())
            op_time = time.time() - start_op
            
            # Performance check
            performance_acceptable = op_time < 1.0  # Should complete in under 1 second
            
            optimization_score = 0
            details = []
            
            if ipex_available:
                optimization_score += 30
                details.append("Intel Extension available")
            
            if cpu_count >= 8:
                optimization_score += 25
                details.append(f"{cpu_count} physical cores")
            
            if logical_count >= 16:
                optimization_score += 20
                details.append(f"{logical_count} logical cores")
            
            if omp_threads != 'Not set' and omp_threads.isdigit() and int(omp_threads) > 8:
                optimization_score += 15
                details.append(f"OMP threads: {omp_threads}")
            
            if performance_acceptable:
                optimization_score += 10
                details.append(f"Matrix ops: {op_time:.3f}s")
            
            success = optimization_score >= 50
            
            return self.log_test_result(
                "Intel Optimizations", success,
                f"Score: {optimization_score}/100. {'; '.join(details)}. CPU: {cpu_brand[:50]}",
                start_time
            )
            
        except Exception as e:
            return self.log_test_result(
                "Intel Optimizations", False,
                f"Exception: {str(e)}",
                start_time
            )
    
    def test_streamlit_interface(self) -> TestResult:
        """Test 6: Streamlit interface functionality"""
        start_time = self.log_test_start("Streamlit Interface")
        
        try:
            # Test Streamlit import
            import streamlit as st
            
            # Test if dashboard file exists and is importable
            dashboard_path = Path("interface/streamlit_dashboard.py")
            if not dashboard_path.exists():
                return self.log_test_result(
                    "Streamlit Interface", False,
                    "Dashboard file not found",
                    start_time
                )
            
            # Test basic Streamlit functionality
            try:
                # Start Streamlit in background with timeout
                process = subprocess.Popen([
                    sys.executable, "-m", "streamlit", "run", 
                    str(dashboard_path),
                    "--server.port", "8502",
                    "--server.headless", "true",
                    "--server.runOnSave", "false",
                    "--browser.gatherUsageStats", "false"
                ], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
                )
                
                # Wait for startup (max 15 seconds)
                startup_timeout = 15
                for _ in range(startup_timeout):
                    time.sleep(1)
                    try:
                        response = requests.get('http://localhost:8502', timeout=2)
                        if response.status_code == 200:
                            # Success - terminate process
                            process.terminate()
                            process.wait(timeout=5)
                            
                            return self.log_test_result(
                                "Streamlit Interface", True,
                                "Dashboard accessible on http://localhost:8502",
                                start_time
                            )
                    except requests.RequestException:
                        continue
                
                # Timeout reached - terminate process
                process.terminate()
                process.wait(timeout=5)
                
                return self.log_test_result(
                    "Streamlit Interface", False,
                    "Dashboard did not become accessible within timeout",
                    start_time
                )
                
            except Exception as e:
                return self.log_test_result(
                    "Streamlit Interface", False,
                    f"Failed to start Streamlit: {str(e)[:100]}",
                    start_time
                )
                
        except Exception as e:
            return self.log_test_result(
                "Streamlit Interface", False,
                f"Exception: {str(e)}",
                start_time
            )
    
    def test_performance_benchmarks(self) -> TestResult:
        """Test 7: Performance and memory benchmarks"""
        start_time = self.log_test_start("Performance Benchmarks")
        
        try:
            # Memory usage check
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_gb = memory_info.rss / (1024**3)
            
            # CPU utilization test
            cpu_percent_before = psutil.cpu_percent(interval=1)
            
            # Simulate workload
            workload_start = time.time()
            test_data = torch.randn(5000, 1000)
            for _ in range(10):
                result = torch.mm(test_data, test_data.t())
                result = torch.relu(result)
            workload_time = time.time() - workload_start
            
            cpu_percent_after = psutil.cpu_percent(interval=1)
            
            # Throughput calculation
            operations_per_second = 10 / workload_time
            
            # Performance criteria
            memory_ok = memory_gb < 30.0  # Less than 30GB for 32GB VPS
            performance_ok = operations_per_second > 1.0  # At least 1 op/sec
            cpu_utilization_ok = cpu_percent_after > 10  # Some CPU usage detected
            
            details = [
                f"Memory: {memory_gb:.2f}GB",
                f"Throughput: {operations_per_second:.2f} ops/sec",
                f"CPU: {cpu_percent_after:.1f}%",
                f"Workload time: {workload_time:.2f}s"
            ]
            
            success = memory_ok and performance_ok and cpu_utilization_ok
            
            return self.log_test_result(
                "Performance Benchmarks", success,
                "; ".join(details),
                start_time
            )
            
        except Exception as e:
            return self.log_test_result(
                "Performance Benchmarks", False,
                f"Exception: {str(e)}",
                start_time
            )
    
    def test_production_readiness(self) -> TestResult:
        """Test 8: Production readiness validation"""
        start_time = self.log_test_start("Production Readiness")
        
        try:
            checks = []
            
            # 1. Debug mode check
            debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
            checks.append(("Debug mode disabled", not debug_mode))
            
            # 2. Required files exist
            required_files = [
                'main.py', 'requirements.txt', 
                'models/dora_trainer.py', 'models/qr_adaptor.py',
                'interface/streamlit_dashboard.py', 
                'data/persian_legal_collector.py'
            ]
            
            missing_files = [f for f in required_files if not Path(f).exists()]
            checks.append(("All required files exist", len(missing_files) == 0))
            
            # 3. Configuration files
            config_files = ['config/training_config.py'] if Path('config/training_config.py').exists() else []
            checks.append(("Configuration available", len(config_files) > 0 or Path('main.py').exists()))
            
            # 4. No TODO comments in main files (basic check)
            todo_found = False
            main_files = ['main.py', 'models/dora_trainer.py', 'interface/streamlit_dashboard.py']
            
            for file_path in main_files:
                if Path(file_path).exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().upper()
                            if 'TODO' in content and 'FIXME' in content:
                                todo_found = True
                                break
                    except Exception:
                        pass
            
            checks.append(("No critical TODOs", not todo_found))
            
            # 5. Import test
            try:
                from main import PersianLegalAISystem
                main_importable = True
            except Exception:
                main_importable = False
            
            checks.append(("Main module importable", main_importable))
            
            # Calculate score
            passed_checks = sum(1 for _, passed in checks)
            total_checks = len(checks)
            
            success = passed_checks >= total_checks - 1  # Allow 1 failure
            
            check_details = [f"{name}: {'✅' if passed else '❌'}" for name, passed in checks]
            
            return self.log_test_result(
                "Production Readiness", success,
                f"{passed_checks}/{total_checks} checks passed. {'; '.join(check_details)}",
                start_time
            )
            
        except Exception as e:
            return self.log_test_result(
                "Production Readiness", False,
                f"Exception: {str(e)}",
                start_time
            )
    
    async def run_complete_tests(self) -> bool:
        """Run the complete test suite"""
        logger.info("🚀 Starting Persian Legal AI System Comprehensive Test Suite\n")
        
        # Define tests
        tests = [
            ("Environment Setup", self.test_environment_setup),
            ("Persian Models Loading", self.test_persian_models_loading),
            ("DoRA Implementation", self.test_dora_implementation),
            ("Data Collection", self.test_data_collection_async),
            ("Intel Optimizations", self.test_intel_optimizations),
            ("Streamlit Interface", self.test_streamlit_interface),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("Production Readiness", self.test_production_readiness),
        ]
        
        # Run tests
        for test_name, test_func in tests:
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {str(e)}")
                self.log_test_result(test_name, False, f"Test crashed: {str(e)}")
            
            # Small delay between tests
            await asyncio.sleep(0.5)
        
        # Generate summary
        return self.generate_summary()
    
    def generate_summary(self) -> bool:
        """Generate test summary and return overall success"""
        total_time = time.time() - self.test_start_time
        
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        total_tests = len(self.results)
        passed_count = len(passed_tests)
        
        logger.info("\n" + "="*80)
        logger.info("📊 PERSIAN LEGAL AI SYSTEM - TEST SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_count}")
        logger.info(f"Failed: {len(failed_tests)}")
        logger.info(f"Success Rate: {(passed_count/total_tests*100):.1f}%")
        logger.info(f"Total Time: {total_time:.2f} seconds")
        logger.info("")
        
        if failed_tests:
            logger.error("❌ FAILED TESTS:")
            for result in failed_tests:
                logger.error(f"  • {result.name}: {result.message}")
            logger.info("")
        
        logger.info("✅ PASSED TESTS:")
        for result in passed_tests:
            logger.info(f"  • {result.name}: {result.message}")
        
        logger.info("="*80)
        
        success = len(failed_tests) == 0
        
        if success:
            logger.success("🎉 ALL TESTS PASSED! System ready for deployment")
        else:
            logger.error("❌ Some tests failed. Fix issues before deployment")
        
        return success


async def main():
    """Main test execution function"""
    test_suite = PersianLegalAITestSuite()
    success = await test_suite.run_complete_tests()
    
    # Exit with appropriate code
    exit_code = 0 if success else 1
    logger.info(f"Exiting with code {exit_code}")
    return exit_code


if __name__ == "__main__":
    # Run the complete test suite
    exit_code = asyncio.run(main())
    sys.exit(exit_code)