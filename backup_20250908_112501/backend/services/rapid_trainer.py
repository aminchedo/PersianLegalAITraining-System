"""
Rapid Training Orchestration Service
سرویس هماهنگی آموزش سریع
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
import psutil
import threading
from pathlib import Path

from ..data.dataset_integration import PersianLegalDataIntegrator
from ..validation.dataset_validator import DatasetQualityValidator
from ..models.enhanced_dora_trainer import DataEnhancedDoraTrainer

logger = logging.getLogger(__name__)

class RapidTrainingOrchestrator:
    """Orchestrates rapid training with premium datasets and quality assurance"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.data_integrator = PersianLegalDataIntegrator()
        self.quality_validator = DatasetQualityValidator()
        self.trainer = None
        
        # Training state
        self.training_active = False
        self.training_thread = None
        self.training_progress = {}
        self.training_results = {}
        
        # Performance monitoring
        self.system_monitor = SystemMonitor()
        
    def _get_default_config(self) -> Dict:
        """Get default training configuration"""
        return {
            'model_config': {
                'base_model': 'HooshvareLab/bert-fa-base-uncased',
                'dora_rank': 64,
                'dora_alpha': 16.0,
                'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj"],
                'accelerated_mode': True,
                'batch_size': 16,
                'gradient_accumulation_steps': 4,
                'max_length': 512
            },
            'training_config': {
                'epochs': 15,
                'learning_rate': 2e-5,
                'save_every_n_epochs': 2,
                'max_training_hours': 24,
                'early_stopping_patience': 3
            },
            'dataset_config': {
                'max_samples_per_dataset': 50000,
                'validation_split': 0.1,
                'quality_threshold': 70
            },
            'output_config': {
                'checkpoint_dir': './checkpoints',
                'report_dir': './reports',
                'model_output_dir': './models'
            }
        }
    
    def execute_rapid_training(self, custom_config: Optional[Dict] = None) -> Dict:
        """Execute fastest possible training with quality assurance"""
        logger.info("🎯 Starting rapid training protocol...")
        
        # Merge custom config with default
        if custom_config:
            self.config = self._merge_configs(self.config, custom_config)
        
        try:
            # Step 1: System readiness check
            logger.info("🔍 Checking system readiness...")
            system_status = self._check_system_readiness()
            if not system_status['ready']:
                return {
                    'success': False,
                    'error': 'System not ready for training',
                    'system_status': system_status
                }
            
            # Step 2: Load and validate premium datasets
            logger.info("📦 Loading premium datasets...")
            datasets = self._load_and_validate_datasets()
            
            if not datasets:
                return {
                    'success': False,
                    'error': 'No valid datasets found for training'
                }
            
            # Step 3: Initialize trainer with optimal configuration
            logger.info("⚙️ Configuring optimized trainer...")
            self.trainer = self._initialize_trainer()
            
            # Step 4: Execute accelerated training
            logger.info("🚀 Starting accelerated training...")
            training_results = self._execute_training(datasets)
            
            # Step 5: Post-training validation and reporting
            logger.info("🧪 Post-training validation...")
            validation_results = self._post_training_validation()
            
            # Step 6: Generate comprehensive report
            logger.info("📄 Generating training report...")
            report_path = self._generate_training_report(training_results, validation_results)
            
            return {
                'success': True,
                'training_results': training_results,
                'validation_results': validation_results,
                'report_path': report_path,
                'datasets_used': list(datasets.keys()),
                'system_info': system_status
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_progress': self.training_progress
            }
    
    def _check_system_readiness(self) -> Dict:
        """Check if system is ready for training"""
        status = {
            'ready': True,
            'issues': [],
            'system_info': {}
        }
        
        # Check memory
        memory = psutil.virtual_memory()
        if memory.available < 4 * 1024 * 1024 * 1024:  # 4GB
            status['issues'].append(f"Insufficient memory: {memory.available / (1024**3):.1f}GB available")
            status['ready'] = False
        
        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.free < 10 * 1024 * 1024 * 1024:  # 10GB
            status['issues'].append(f"Insufficient disk space: {disk.free / (1024**3):.1f}GB available")
            status['ready'] = False
        
        # Check CPU
        cpu_count = psutil.cpu_count()
        if cpu_count < 2:
            status['issues'].append(f"Insufficient CPU cores: {cpu_count}")
            status['ready'] = False
        
        status['system_info'] = {
            'memory_gb': memory.total / (1024**3),
            'available_memory_gb': memory.available / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'cpu_cores': cpu_count,
            'cpu_usage': psutil.cpu_percent()
        }
        
        return status
    
    def _load_and_validate_datasets(self) -> Dict:
        """Load and validate premium datasets"""
        # Load all available datasets
        all_datasets = self.data_integrator.load_all_available_datasets(
            max_samples_per_dataset=self.config['dataset_config']['max_samples_per_dataset']
        )
        
        if not all_datasets:
            logger.warning("⚠️ No datasets loaded, creating placeholder data")
            return self._create_placeholder_datasets()
        
        # Validate dataset quality
        logger.info("🔍 Validating dataset quality...")
        validation_results = self.quality_validator.validate_datasets(all_datasets)
        
        # Filter datasets based on quality
        quality_threshold = self.config['dataset_config']['quality_threshold']
        valid_datasets = {}
        
        for key, dataset in all_datasets.items():
            validation_result = validation_results.get(key, {})
            quality_score = validation_result.get('overall_quality_score', 0)
            
            if quality_score >= quality_threshold:
                valid_datasets[key] = dataset
                logger.info(f"✅ Dataset {key} passed quality check (Score: {quality_score}/100)")
            else:
                logger.warning(f"⚠️ Dataset {key} failed quality check (Score: {quality_score}/100)")
        
        # Export validation report
        validation_report_path = os.path.join(
            self.config['output_config']['report_dir'], 
            'dataset_validation_report.json'
        )
        self.quality_validator.export_validation_report(validation_results, validation_report_path)
        
        return valid_datasets
    
    def _create_placeholder_datasets(self) -> Dict:
        """Create placeholder datasets for testing"""
        from datasets import Dataset
        
        placeholder_data = [
            {
                'text': f'این یک نمونه متن حقوقی فارسی شماره {i} است که برای تست سیستم استفاده می‌شود. این متن شامل مفاهیم حقوقی مانند قانون، دادگاه، قاضی و احکام قضایی است. متن شامل مواد قانونی و مقررات حقوقی می‌باشد.',
                'source_dataset': 'placeholder',
                'type': 'legal_document',
                'length': 200
            }
            for i in range(5000)
        ]
        
        return {
            'placeholder': Dataset.from_list(placeholder_data)
        }
    
    def _initialize_trainer(self) -> DataEnhancedDoraTrainer:
        """Initialize the enhanced trainer"""
        trainer = DataEnhancedDoraTrainer(self.config['model_config'])
        
        # Load model
        trainer.load_model()
        
        # Setup optimizers
        trainer.setup_optimizers(
            learning_rate=self.config['training_config']['learning_rate']
        )
        
        return trainer
    
    def _execute_training(self, datasets: Dict) -> Dict:
        """Execute the actual training process"""
        start_time = time.time()
        
        # Load training data
        self.trainer.load_premium_training_data()
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        try:
            # Execute accelerated training
            training_results = self.trainer.accelerated_training(
                epochs=self.config['training_config']['epochs'],
                save_every_n_epochs=self.config['training_config']['save_every_n_epochs']
            )
            
            # Validate model performance
            validation_results = self.trainer.validate_model_performance()
            
            # Get training summary
            training_summary = self.trainer.get_training_summary()
            
            training_time = time.time() - start_time
            
            return {
                'training_results': training_results,
                'validation_results': validation_results,
                'training_summary': training_summary,
                'training_time_minutes': training_time / 60,
                'system_metrics': self.system_monitor.get_metrics()
            }
            
        finally:
            # Stop system monitoring
            self.system_monitor.stop_monitoring()
    
    def _post_training_validation(self) -> Dict:
        """Perform post-training validation"""
        if not self.trainer:
            return {'error': 'No trainer available for validation'}
        
        # Validate model performance
        validation_results = self.trainer.validate_model_performance()
        
        # Export training report
        report_path = self.trainer.export_training_report(
            os.path.join(self.config['output_config']['report_dir'], 'training_report.json')
        )
        
        return {
            'model_validation': validation_results,
            'training_report_path': report_path
        }
    
    def _generate_training_report(self, training_results: Dict, validation_results: Dict) -> str:
        """Generate comprehensive training report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_config': self.config,
            'training_results': training_results,
            'validation_results': validation_results,
            'system_info': self.system_monitor.get_metrics(),
            'dataset_info': self._get_dataset_info()
        }
        
        report_path = os.path.join(
            self.config['output_config']['report_dir'],
            f'rapid_training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Comprehensive training report saved to {report_path}")
        return report_path
    
    def _get_dataset_info(self) -> Dict:
        """Get information about datasets used"""
        if not self.trainer or not self.trainer.training_data:
            return {}
        
        return self.trainer._get_dataset_info()
    
    def _merge_configs(self, default_config: Dict, custom_config: Dict) -> Dict:
        """Merge custom configuration with default"""
        merged = default_config.copy()
        
        for key, value in custom_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def get_training_status(self) -> Dict:
        """Get current training status"""
        return {
            'training_active': self.training_active,
            'progress': self.training_progress,
            'results': self.training_results
        }
    
    def stop_training(self) -> bool:
        """Stop ongoing training"""
        if self.training_active and self.training_thread:
            self.training_active = False
            self.training_thread.join(timeout=30)
            return True
        return False

class SystemMonitor:
    """Monitor system resources during training"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = []
        self.start_time = None
    
    def start_monitoring(self):
        """Start system monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                    'disk_usage_percent': psutil.disk_usage('/').percent
                }
                
                self.metrics.append(metrics)
                
                # Keep only last 1000 measurements
                if len(self.metrics) > 1000:
                    self.metrics = self.metrics[-1000:]
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(10)
    
    def get_metrics(self) -> Dict:
        """Get monitoring metrics"""
        if not self.metrics:
            return {}
        
        # Calculate averages
        avg_cpu = sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics)
        avg_memory = sum(m['memory_percent'] for m in self.metrics) / len(self.metrics)
        min_memory_available = min(m['memory_available_gb'] for m in self.metrics)
        max_disk_usage = max(m['disk_usage_percent'] for m in self.metrics)
        
        return {
            'monitoring_duration_minutes': (time.time() - self.start_time) / 60 if self.start_time else 0,
            'measurements_count': len(self.metrics),
            'avg_cpu_percent': round(avg_cpu, 2),
            'avg_memory_percent': round(avg_memory, 2),
            'min_memory_available_gb': round(min_memory_available, 2),
            'max_disk_usage_percent': round(max_disk_usage, 2),
            'latest_metrics': self.metrics[-1] if self.metrics else {}
        }