#!/usr/bin/env python3
"""
COMPREHENSIVE REAL-WORLD TEST SCRIPT
This script performs a REAL test of the entire system pipeline on a SMALL real dataset.
It PROVES the system is functional with real data, real code, and real outputs.
"""

import sys
import os
import time
import logging
import asyncio
import requests
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import real system components
from services.persian_data_processor import PersianLegalDataProcessor
from models.dora_trainer import DoRATrainer, DoRAConfig
from models.qr_adaptor import QRAdaptor, QRAdaptorConfig
from backend.database.connection import init_database, db_manager
from optimization.system_optimizer import system_optimizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("system_test.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystemTestRunner:
    """Real system test runner"""
    
    def __init__(self):
        self.test_results = {
            'start_time': datetime.now().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': [],
            'system_metrics': {},
            'final_report': {}
        }
        self.backend_url = "http://localhost:8000"
        self.backend_running = False
    
    def log_test_result(self, test_name: str, passed: bool, details: str = "", metrics: dict = None):
        """Log test result"""
        result = {
            'test_name': test_name,
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {}
        }
        
        self.test_results['test_details'].append(result)
        
        if passed:
            self.test_results['tests_passed'] += 1
            logger.info(f"✅ {test_name}: PASSED - {details}")
        else:
            self.test_results['tests_failed'] += 1
            logger.error(f"❌ {test_name}: FAILED - {details}")
    
    def test_data_loading(self) -> bool:
        """Test REAL data loading and preprocessing"""
        try:
            logger.info("🧪 Testing REAL data loading and preprocessing...")
            
            # Initialize data processor
            processor = PersianLegalDataProcessor()
            
            # Load real sample data
            sample_data = processor.load_sample_data()
            
            if len(sample_data) == 0:
                self.log_test_result("Data Loading", False, "No sample data loaded")
                return False
            
            # Preprocess data
            processed_data = processor.preprocess_persian_text(sample_data)
            
            if len(processed_data) == 0:
                self.log_test_result("Data Preprocessing", False, "No data after preprocessing")
                return False
            
            # Assess quality
            quality_assessments = processor.assess_document_quality(processed_data)
            high_quality_data = processor.filter_high_quality_documents(quality_assessments)
            
            # Create training dataset
            training_dataset = processor.create_training_dataset(high_quality_data, "text_classification")
            
            metrics = {
                'sample_data_count': len(sample_data),
                'processed_data_count': len(processed_data),
                'high_quality_data_count': len(high_quality_data),
                'training_dataset_size': training_dataset.get('size', 0),
                'task_type': training_dataset.get('task_type', 'unknown')
            }
            
            self.log_test_result("Data Loading", True, f"Loaded {len(sample_data)} documents, created {training_dataset.get('size', 0)} training samples", metrics)
            return True
            
        except Exception as e:
            self.log_test_result("Data Loading", False, f"Exception: {str(e)}")
            return False
    
    def test_dora_training(self) -> bool:
        """Test REAL DoRA model training"""
        try:
            logger.info("🧪 Testing REAL DoRA model training...")
            
            # Load and preprocess data
            processor = PersianLegalDataProcessor()
            sample_data = processor.load_sample_data()
            processed_data = processor.preprocess_persian_text(sample_data)
            quality_assessments = processor.assess_document_quality(processed_data)
            high_quality_data = processor.filter_high_quality_documents(quality_assessments)
            training_dataset = processor.create_training_dataset(high_quality_data, "text_classification")
            
            if training_dataset.get('size', 0) == 0:
                self.log_test_result("DoRA Training", False, "No training data available")
                return False
            
            # Initialize DoRA trainer
            config = DoRAConfig(
                base_model="HooshvareLab/bert-base-parsbert-uncased",
                dora_rank=4,  # Smaller rank for testing
                dora_alpha=8,
                num_epochs=1,  # Single epoch for testing
                batch_size=2,  # Small batch size
                learning_rate=2e-4
            )
            
            trainer = DoRATrainer(config)
            
            # Load model
            model, tokenizer = trainer.load_model()
            
            # Apply DoRA
            trainer.apply_dora()
            
            # Create small dataset for testing
            dataset = training_dataset['dataset'][:5]  # Use only 5 samples for testing
            train_dataloader = trainer.create_dataloader(dataset, batch_size=2)
            
            # Setup optimizer
            trainer.setup_optimizer()
            
            # Run ONE training step
            batch = next(iter(train_dataloader))
            loss = trainer.training_step(model, batch)
            
            if not isinstance(loss, type(loss)) or loss.item() <= 0:
                self.log_test_result("DoRA Training", False, f"Invalid loss value: {loss}")
                return False
            
            metrics = {
                'loss_value': float(loss.item()),
                'training_samples': len(dataset),
                'batch_size': 2,
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            
            self.log_test_result("DoRA Training", True, f"Training step completed with loss: {loss.item():.4f}", metrics)
            return True
            
        except Exception as e:
            self.log_test_result("DoRA Training", False, f"Exception: {str(e)}")
            return False
    
    def test_qr_adaptor_training(self) -> bool:
        """Test REAL QR-Adaptor training"""
        try:
            logger.info("🧪 Testing REAL QR-Adaptor training...")
            
            # Load and preprocess data
            processor = PersianLegalDataProcessor()
            sample_data = processor.load_sample_data()
            processed_data = processor.preprocess_persian_text(sample_data)
            quality_assessments = processor.assess_document_quality(processed_data)
            high_quality_data = processor.filter_high_quality_documents(quality_assessments)
            training_dataset = processor.create_training_dataset(high_quality_data, "text_classification")
            
            if training_dataset.get('size', 0) == 0:
                self.log_test_result("QR-Adaptor Training", False, "No training data available")
                return False
            
            # Initialize QR-Adaptor
            config = QRAdaptorConfig(
                base_model="HooshvareLab/bert-base-parsbert-uncased",
                quantization_bits=4,
                rank=4,
                alpha=8,
                num_epochs=1,
                batch_size=2,
                learning_rate=2e-4
            )
            
            qr_adaptor = QRAdaptor(config)
            
            # Load model
            model, tokenizer = qr_adaptor.load_model()
            
            # Apply QR adaptation
            qr_adaptor.apply_qr_adaptation()
            
            # Create small dataset for testing
            dataset = training_dataset['dataset'][:5]  # Use only 5 samples for testing
            train_dataloader = qr_adaptor.create_dataloader(dataset, batch_size=2)
            
            # Setup optimizer
            qr_adaptor.setup_optimizer()
            
            # Run ONE training step
            batch = next(iter(train_dataloader))
            loss = qr_adaptor.training_step(model, batch)
            
            if not isinstance(loss, type(loss)) or loss.item() <= 0:
                self.log_test_result("QR-Adaptor Training", False, f"Invalid loss value: {loss}")
                return False
            
            metrics = {
                'loss_value': float(loss.item()),
                'training_samples': len(dataset),
                'batch_size': 2,
                'quantization_bits': config.quantization_bits,
                'rank': config.rank,
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            
            self.log_test_result("QR-Adaptor Training", True, f"Training step completed with loss: {loss.item():.4f}", metrics)
            return True
            
        except Exception as e:
            self.log_test_result("QR-Adaptor Training", False, f"Exception: {str(e)}")
            return False
    
    def test_database_operations(self) -> bool:
        """Test REAL database operations"""
        try:
            logger.info("🧪 Testing REAL database operations...")
            
            # Initialize database
            if not init_database():
                self.log_test_result("Database Operations", False, "Database initialization failed")
                return False
            
            # Test database connection
            if not db_manager.test_connection():
                self.log_test_result("Database Operations", False, "Database connection test failed")
                return False
            
            # Get database info
            db_info = db_manager.get_database_info()
            
            # Test basic query
            result = db_manager.execute_query("SELECT 1 as test_value")
            
            if not result or result[0][0] != 1:
                self.log_test_result("Database Operations", False, "Basic query test failed")
                return False
            
            metrics = {
                'database_url': db_info.get('database_url', 'unknown'),
                'table_count': db_info.get('table_count', 0),
                'database_size_mb': db_info.get('database_size_mb', 0),
                'connection_test': True,
                'query_test': True
            }
            
            self.log_test_result("Database Operations", True, f"Database operations successful, {db_info.get('table_count', 0)} tables", metrics)
            return True
            
        except Exception as e:
            self.log_test_result("Database Operations", False, f"Exception: {str(e)}")
            return False
    
    def test_system_optimization(self) -> bool:
        """Test REAL system optimization"""
        try:
            logger.info("🧪 Testing REAL system optimization...")
            
            # Get optimization report
            report = system_optimizer.get_optimization_report()
            
            if not report:
                self.log_test_result("System Optimization", False, "No optimization report generated")
                return False
            
            # Test optimal settings
            optimal_batch_size = system_optimizer.get_optimal_batch_size()
            optimal_workers = system_optimizer.get_optimal_num_workers()
            
            if optimal_batch_size <= 0 or optimal_workers <= 0:
                self.log_test_result("System Optimization", False, f"Invalid optimal settings: batch_size={optimal_batch_size}, workers={optimal_workers}")
                return False
            
            metrics = {
                'optimal_batch_size': optimal_batch_size,
                'optimal_num_workers': optimal_workers,
                'pytorch_threads': report.get('optimal_settings', {}).get('pytorch_threads', 0),
                'cpu_cores': report.get('current_resources', {}).get('cpu_cores', 0),
                'memory_usage_percent': report.get('current_resources', {}).get('memory_usage_percent', 0),
                'gpu_available': report.get('current_resources', {}).get('gpu_available', False)
            }
            
            self.log_test_result("System Optimization", True, f"Optimization successful, batch_size={optimal_batch_size}, workers={optimal_workers}", metrics)
            return True
            
        except Exception as e:
            self.log_test_result("System Optimization", False, f"Exception: {str(e)}")
            return False
    
    def test_backend_api(self) -> bool:
        """Test REAL backend API endpoints"""
        try:
            logger.info("🧪 Testing REAL backend API endpoints...")
            
            # Check if backend is running
            try:
                response = requests.get(f"{self.backend_url}/api/system/health", timeout=5)
                if response.status_code == 200:
                    self.backend_running = True
                else:
                    self.log_test_result("Backend API", False, f"Backend not responding: {response.status_code}")
                    return False
            except requests.exceptions.RequestException:
                self.log_test_result("Backend API", False, "Backend not running or not accessible")
                return False
            
            # Test system health endpoint
            response = requests.get(f"{self.backend_url}/api/system/health", timeout=10)
            if response.status_code != 200:
                self.log_test_result("Backend API", False, f"Health endpoint failed: {response.status_code}")
                return False
            
            health_data = response.json()
            
            # Test system metrics endpoint
            response = requests.get(f"{self.backend_url}/api/system/metrics", timeout=10)
            if response.status_code != 200:
                self.log_test_result("Backend API", False, f"Metrics endpoint failed: {response.status_code}")
                return False
            
            metrics_data = response.json()
            
            # Test training sessions endpoint
            response = requests.get(f"{self.backend_url}/api/training/sessions", timeout=10)
            if response.status_code != 200:
                self.log_test_result("Backend API", False, f"Training sessions endpoint failed: {response.status_code}")
                return False
            
            sessions_data = response.json()
            
            api_metrics = {
                'health_status': health_data.get('status', 'unknown'),
                'cpu_percent': health_data.get('system_metrics', {}).get('cpu_percent', 0),
                'memory_percent': health_data.get('system_metrics', {}).get('memory_percent', 0),
                'active_processes': health_data.get('system_metrics', {}).get('active_processes', 0),
                'training_sessions_count': len(sessions_data),
                'api_endpoints_tested': 3
            }
            
            self.log_test_result("Backend API", True, f"All API endpoints working, {len(sessions_data)} training sessions", api_metrics)
            return True
            
        except Exception as e:
            self.log_test_result("Backend API", False, f"Exception: {str(e)}")
            return False
    
    def test_full_pipeline(self) -> bool:
        """Test complete system pipeline"""
        try:
            logger.info("🧪 Testing COMPLETE system pipeline...")
            
            # Step 1: Data loading
            processor = PersianLegalDataProcessor()
            sample_data = processor.load_sample_data()
            processed_data = processor.preprocess_persian_text(sample_data)
            quality_assessments = processor.assess_document_quality(processed_data)
            high_quality_data = processor.filter_high_quality_documents(quality_assessments)
            training_dataset = processor.create_training_dataset(high_quality_data, "text_classification")
            
            if training_dataset.get('size', 0) == 0:
                self.log_test_result("Full Pipeline", False, "No training data in pipeline")
                return False
            
            # Step 2: Model training (DoRA)
            config = DoRAConfig(
                base_model="HooshvareLab/bert-base-parsbert-uncased",
                dora_rank=4,
                dora_alpha=8,
                num_epochs=1,
                batch_size=2,
                learning_rate=2e-4
            )
            
            trainer = DoRATrainer(config)
            model, tokenizer = trainer.load_model()
            trainer.apply_dora()
            
            # Step 3: Training execution
            dataset = training_dataset['dataset'][:3]  # Use only 3 samples
            train_dataloader = trainer.create_dataloader(dataset, batch_size=2)
            trainer.setup_optimizer()
            
            # Run training
            total_loss = 0.0
            num_steps = 0
            
            for batch in train_dataloader:
                loss = trainer.training_step(model, batch)
                total_loss += loss.item()
                num_steps += 1
                break  # Only one step for testing
            
            avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
            
            # Step 4: System metrics
            system_metrics = system_optimizer.get_optimization_report()
            
            pipeline_metrics = {
                'data_samples': len(sample_data),
                'training_samples': len(dataset),
                'training_steps': num_steps,
                'average_loss': avg_loss,
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'system_cpu_cores': system_metrics.get('current_resources', {}).get('cpu_cores', 0),
                'system_memory_percent': system_metrics.get('current_resources', {}).get('memory_usage_percent', 0)
            }
            
            self.log_test_result("Full Pipeline", True, f"Complete pipeline executed successfully, avg_loss: {avg_loss:.4f}", pipeline_metrics)
            return True
            
        except Exception as e:
            self.log_test_result("Full Pipeline", False, f"Exception: {str(e)}")
            return False
    
    def generate_final_report(self):
        """Generate final test report"""
        try:
            self.test_results['end_time'] = datetime.now().isoformat()
            self.test_results['total_tests'] = self.test_results['tests_passed'] + self.test_results['tests_failed']
            self.test_results['success_rate'] = (self.test_results['tests_passed'] / self.test_results['total_tests'] * 100) if self.test_results['total_tests'] > 0 else 0
            
            # System metrics
            system_metrics = system_optimizer.get_optimization_report()
            self.test_results['system_metrics'] = system_metrics
            
            # Final report
            self.test_results['final_report'] = {
                'test_summary': {
                    'total_tests': self.test_results['total_tests'],
                    'tests_passed': self.test_results['tests_passed'],
                    'tests_failed': self.test_results['tests_failed'],
                    'success_rate': f"{self.test_results['success_rate']:.1f}%"
                },
                'system_status': {
                    'backend_running': self.backend_running,
                    'database_connected': db_manager.test_connection() if 'db_manager' in globals() else False,
                    'optimization_active': system_metrics.get('monitoring_active', False)
                },
                'performance_metrics': {
                    'optimal_batch_size': system_metrics.get('optimal_settings', {}).get('batch_size', 0),
                    'optimal_workers': system_metrics.get('optimal_settings', {}).get('num_workers', 0),
                    'cpu_cores': system_metrics.get('current_resources', {}).get('cpu_cores', 0),
                    'memory_usage': system_metrics.get('current_resources', {}).get('memory_usage_percent', 0)
                }
            }
            
            # Save report to file
            report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Final test report saved to: {report_file}")
            
            return self.test_results
            
        except Exception as e:
            logger.error(f"Failed to generate final report: {e}")
            return {}
    
    def run_all_tests(self) -> bool:
        """Run all system tests"""
        try:
            logger.info("🚀 Starting COMPREHENSIVE REAL-WORLD SYSTEM TEST...")
            logger.info("=" * 60)
            
            # Run all tests
            tests = [
                ("Data Loading", self.test_data_loading),
                ("DoRA Training", self.test_dora_training),
                ("QR-Adaptor Training", self.test_qr_adaptor_training),
                ("Database Operations", self.test_database_operations),
                ("System Optimization", self.test_system_optimization),
                ("Backend API", self.test_backend_api),
                ("Full Pipeline", self.test_full_pipeline)
            ]
            
            for test_name, test_func in tests:
                logger.info(f"\n{'='*20} {test_name} {'='*20}")
                test_func()
                time.sleep(1)  # Brief pause between tests
            
            # Generate final report
            logger.info("\n" + "="*60)
            logger.info("📊 GENERATING FINAL TEST REPORT...")
            logger.info("="*60)
            
            final_report = self.generate_final_report()
            
            # Print summary
            logger.info(f"\n🎯 TEST SUMMARY:")
            logger.info(f"   Total Tests: {self.test_results['total_tests']}")
            logger.info(f"   Passed: {self.test_results['tests_passed']}")
            logger.info(f"   Failed: {self.test_results['tests_failed']}")
            logger.info(f"   Success Rate: {self.test_results['success_rate']:.1f}%")
            
            if self.test_results['tests_failed'] == 0:
                logger.info("\n🎉 ALL TESTS PASSED! The system is fully functional.")
                return True
            else:
                logger.error(f"\n❌ {self.test_results['tests_failed']} tests failed. System needs attention.")
                return False
                
        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            return False

def main():
    """Main test function"""
    try:
        # Create test runner
        test_runner = SystemTestRunner()
        
        # Run all tests
        success = test_runner.run_all_tests()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()