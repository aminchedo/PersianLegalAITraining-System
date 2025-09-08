"""
Training Service for Persian Legal AI
سرویس آموزش برای هوش مصنوعی حقوقی فارسی
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass

from ..models.model_manager import ModelManager, ModelConfig, TrainingConfig, ModelType
from ..services.persian_data_processor import PersianLegalDataProcessor, DataSourceConfig
from ..optimization.intel_optimizer import IntelCPUOptimizer
from ..optimization.system_optimizer import SystemOptimizer
from ..optimization.memory_optimizer import MemoryOptimizer
from ..database.connection import db_manager

logger = logging.getLogger(__name__)

@dataclass
class TrainingRequest:
    """Training request configuration"""
    model_name: str
    model_type: str
    base_model: str
    dataset_sources: List[str]
    training_config: Dict[str, Any]
    model_config: Dict[str, Any]
    priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class TrainingResponse:
    """Training response"""
    session_id: str
    status: str
    message: str
    estimated_duration: Optional[str] = None

class TrainingService:
    """
    Core training orchestration service
    """
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.data_processor = PersianLegalDataProcessor(DataSourceConfig())
        self.intel_optimizer = IntelCPUOptimizer()
        self.system_optimizer = SystemOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        
        # Training queue
        self.training_queue = []
        self.max_concurrent_sessions = 2  # Limit concurrent training sessions
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Training Service initialized")
    
    def _start_background_tasks(self):
        """Start background monitoring and processing tasks"""
        try:
            # Start system monitoring
            self.system_optimizer.start_performance_monitoring()
            
            # Start memory monitoring
            self.memory_optimizer.monitor_memory_usage()
            
            # Start training queue processor
            asyncio.create_task(self._process_training_queue())
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
    
    async def submit_training_request(self, request: TrainingRequest) -> TrainingResponse:
        """Submit a training request"""
        try:
            # Validate request
            validation_result = await self._validate_training_request(request)
            if not validation_result['valid']:
                return TrainingResponse(
                    session_id="",
                    status="failed",
                    message=validation_result['error']
                )
            
            # Create training session
            session_id = await self._create_training_session(request)
            
            # Add to queue or start immediately
            if len(self.model_manager.active_trainers) < self.max_concurrent_sessions:
                # Start immediately
                success = await self.model_manager.start_training_session(session_id)
                status = "started" if success else "queued"
            else:
                # Add to queue
                self.training_queue.append(session_id)
                status = "queued"
            
            # Estimate duration
            estimated_duration = self._estimate_training_duration(request)
            
            return TrainingResponse(
                session_id=session_id,
                status=status,
                message=f"Training request submitted successfully",
                estimated_duration=estimated_duration
            )
            
        except Exception as e:
            logger.error(f"Failed to submit training request: {e}")
            return TrainingResponse(
                session_id="",
                status="failed",
                message=f"Failed to submit training request: {str(e)}"
            )
    
    async def _validate_training_request(self, request: TrainingRequest) -> Dict[str, Any]:
        """Validate training request"""
        try:
            # Check model type
            try:
                model_type = ModelType(request.model_type)
            except ValueError:
                return {
                    'valid': False,
                    'error': f"Invalid model type: {request.model_type}"
                }
            
            # Check base model availability
            if not request.base_model:
                return {
                    'valid': False,
                    'error': "Base model is required"
                }
            
            # Check dataset sources
            if not request.dataset_sources:
                return {
                    'valid': False,
                    'error': "At least one dataset source is required"
                }
            
            # Check system resources
            system_metrics = self.system_optimizer.collect_system_metrics()
            memory_usage = system_metrics.get('memory', {}).get('used_percent', 0)
            
            if memory_usage > 90:
                return {
                    'valid': False,
                    'error': "Insufficient system memory"
                }
            
            # Check training configuration
            if not request.training_config.get('epochs', 0) > 0:
                return {
                    'valid': False,
                    'error': "Invalid number of epochs"
                }
            
            return {'valid': True}
            
        except Exception as e:
            logger.error(f"Failed to validate training request: {e}")
            return {
                'valid': False,
                'error': f"Validation error: {str(e)}"
            }
    
    async def _create_training_session(self, request: TrainingRequest) -> str:
        """Create training session from request"""
        try:
            # Convert request to model and training configs
            model_config = ModelConfig(
                model_name=request.model_name,
                model_type=ModelType(request.model_type),
                base_model=request.base_model,
                **request.model_config
            )
            
            training_config = TrainingConfig(**request.training_config)
            
            # Prepare dataset configuration
            dataset_config = {
                'sources': request.dataset_sources,
                'task_type': request.training_config.get('task_type', 'text_generation'),
                'max_documents': request.training_config.get('max_documents', 1000)
            }
            
            # Create session
            session_id = await self.model_manager.create_training_session(
                model_config, training_config, dataset_config
            )
            
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create training session: {e}")
            raise
    
    def _estimate_training_duration(self, request: TrainingRequest) -> str:
        """Estimate training duration"""
        try:
            epochs = request.training_config.get('epochs', 10)
            batch_size = request.training_config.get('batch_size', 4)
            max_documents = request.training_config.get('max_documents', 1000)
            
            # Rough estimation based on system specs
            steps_per_epoch = max_documents // batch_size
            total_steps = steps_per_epoch * epochs
            
            # Assume 2-5 seconds per step on 24-core system
            estimated_seconds = total_steps * 3
            
            # Convert to human readable format
            if estimated_seconds < 3600:
                return f"{estimated_seconds // 60} minutes"
            elif estimated_seconds < 86400:
                return f"{estimated_seconds // 3600} hours"
            else:
                return f"{estimated_seconds // 86400} days"
                
        except Exception as e:
            logger.error(f"Failed to estimate training duration: {e}")
            return "Unknown"
    
    async def _process_training_queue(self):
        """Process training queue"""
        while True:
            try:
                # Check if we can start more training sessions
                if (self.training_queue and 
                    len(self.model_manager.active_trainers) < self.max_concurrent_sessions):
                    
                    # Get next session from queue
                    session_id = self.training_queue.pop(0)
                    
                    # Start training
                    success = await self.model_manager.start_training_session(session_id)
                    
                    if not success:
                        # Put back in queue if failed
                        self.training_queue.insert(0, session_id)
                
                # Wait before checking again
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in training queue processor: {e}")
                await asyncio.sleep(30)
    
    async def get_training_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get training session status"""
        try:
            return await self.model_manager.get_training_session_status(session_id)
            
        except Exception as e:
            logger.error(f"Failed to get training status: {e}")
            return None
    
    async def get_training_metrics(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get training metrics"""
        try:
            return await self.model_manager.get_training_metrics(session_id, limit)
            
        except Exception as e:
            logger.error(f"Failed to get training metrics: {e}")
            return []
    
    async def pause_training(self, session_id: str) -> bool:
        """Pause training session"""
        try:
            return await self.model_manager.pause_training_session(session_id)
            
        except Exception as e:
            logger.error(f"Failed to pause training: {e}")
            return False
    
    async def resume_training(self, session_id: str) -> bool:
        """Resume training session"""
        try:
            return await self.model_manager.resume_training_session(session_id)
            
        except Exception as e:
            logger.error(f"Failed to resume training: {e}")
            return False
    
    async def cancel_training(self, session_id: str) -> bool:
        """Cancel training session"""
        try:
            # Remove from queue if present
            if session_id in self.training_queue:
                self.training_queue.remove(session_id)
            
            return await self.model_manager.cancel_training_session(session_id)
            
        except Exception as e:
            logger.error(f"Failed to cancel training: {e}")
            return False
    
    async def list_training_sessions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List training sessions"""
        try:
            return await self.model_manager.list_training_sessions(status)
            
        except Exception as e:
            logger.error(f"Failed to list training sessions: {e}")
            return []
    
    async def get_system_performance(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            system_performance = await self.model_manager.get_system_performance()
            
            # Add training queue information
            system_performance['training_queue'] = {
                'queue_length': len(self.training_queue),
                'max_concurrent_sessions': self.max_concurrent_sessions,
                'active_sessions': len(self.model_manager.active_trainers)
            }
            
            return system_performance
            
        except Exception as e:
            logger.error(f"Failed to get system performance: {e}")
            return {}
    
    async def optimize_system_for_training(self) -> Dict[str, Any]:
        """Optimize system for training"""
        try:
            optimization_results = {}
            
            # Optimize CPU
            cpu_optimization = self.intel_optimizer.optimize_for_24_core_system()
            optimization_results['cpu_optimization'] = cpu_optimization
            
            # Optimize system
            system_optimization = self.system_optimizer.optimize_for_24_core_system()
            optimization_results['system_optimization'] = system_optimization
            
            # Optimize memory
            memory_usage = self.memory_optimizer.get_memory_usage()
            optimization_results['memory_optimization'] = memory_usage
            
            logger.info("System optimization completed")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Failed to optimize system: {e}")
            return {}
    
    async def prepare_training_data(self, sources: List[str], task_type: str, 
                                  max_documents: int = 1000) -> Dict[str, Any]:
        """Prepare training data from sources"""
        try:
            # Fetch documents from sources
            documents = await self.data_processor.fetch_legal_documents(
                sources=sources,
                date_range=(datetime.now() - timedelta(days=365), datetime.now()),
                categories=['laws', 'regulations', 'judgments']
            )
            
            # Limit documents
            documents = documents[:max_documents]
            
            # Preprocess Persian text
            processed_docs = self.data_processor.preprocess_persian_text(documents)
            
            # Assess quality
            quality_assessments = self.data_processor.assess_document_quality(processed_docs)
            
            # Filter high quality documents
            high_quality_docs = self.data_processor.filter_high_quality_documents(
                quality_assessments, min_quality_score=0.6
            )
            
            # Create training dataset
            training_dataset = self.data_processor.create_training_datasets(
                high_quality_docs, task_type
            )
            
            return {
                'total_documents': len(documents),
                'processed_documents': len(processed_docs),
                'high_quality_documents': len(high_quality_docs),
                'training_dataset': training_dataset,
                'quality_metrics': {
                    'avg_quality_score': sum(q.score for _, q in quality_assessments) / len(quality_assessments) if quality_assessments else 0,
                    'high_quality_ratio': len(high_quality_docs) / len(documents) if documents else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return {}
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        try:
            # This would typically come from a model registry
            # For now, return predefined models
            return [
                {
                    'name': 'PersianMind-v1.0',
                    'type': 'causal_lm',
                    'base_model': 'universitytehran/PersianMind-v1.0',
                    'description': 'Persian language model for legal text generation',
                    'supported_tasks': ['text_generation', 'question_answering']
                },
                {
                    'name': 'ParsBERT-Legal',
                    'type': 'bert',
                    'base_model': 'HooshvareLab/bert-base-parsbert-uncased',
                    'description': 'Persian BERT model for legal text understanding',
                    'supported_tasks': ['text_classification', 'named_entity_recognition']
                }
            ]
            
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    async def get_training_recommendations(self, model_name: str, task_type: str) -> Dict[str, Any]:
        """Get training recommendations for model and task"""
        try:
            recommendations = {
                'model_config': {},
                'training_config': {},
                'data_config': {}
            }
            
            # Model-specific recommendations
            if 'PersianMind' in model_name:
                recommendations['model_config'] = {
                    'dora_rank': 64,
                    'dora_alpha': 16.0,
                    'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                }
                recommendations['training_config'] = {
                    'batch_size': 4,
                    'learning_rate': 1e-4,
                    'epochs': 10,
                    'gradient_accumulation_steps': 2
                }
            elif 'ParsBERT' in model_name:
                recommendations['model_config'] = {
                    'dora_rank': 32,
                    'dora_alpha': 8.0,
                    'target_modules': ["query", "key", "value", "dense"]
                }
                recommendations['training_config'] = {
                    'batch_size': 8,
                    'learning_rate': 2e-5,
                    'epochs': 5,
                    'gradient_accumulation_steps': 1
                }
            
            # Task-specific recommendations
            if task_type == 'text_generation':
                recommendations['data_config'] = {
                    'max_length': 2048,
                    'min_length': 100,
                    'sources': ['naab', 'qavanin']
                }
            elif task_type == 'question_answering':
                recommendations['data_config'] = {
                    'max_length': 512,
                    'min_length': 50,
                    'sources': ['naab', 'majles']
                }
            elif task_type == 'text_classification':
                recommendations['data_config'] = {
                    'max_length': 256,
                    'min_length': 20,
                    'sources': ['qavanin', 'majles']
                }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get training recommendations: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup training service"""
        try:
            # Cancel all active training sessions
            for session_id in list(self.model_manager.active_trainers.keys()):
                await self.cancel_training(session_id)
            
            # Clear training queue
            self.training_queue.clear()
            
            # Cleanup model manager
            await self.model_manager.cleanup()
            
            # Cleanup optimizers
            self.system_optimizer.cleanup()
            self.memory_optimizer.cleanup()
            self.intel_optimizer.cleanup()
            
            logger.info("Training Service cleanup complete")
            
        except Exception as e:
            logger.error(f"Failed to cleanup training service: {e}")

# Global training service instance
training_service = TrainingService()