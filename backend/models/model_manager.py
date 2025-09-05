"""
Model Manager for Persian Legal AI Training System
مدیر مدل برای سیستم آموزش هوش مصنوعی حقوقی فارسی
"""

import os
import json
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid
import torch
from dataclasses import dataclass
from enum import Enum

from ..database.models import TrainingSession, ModelCheckpoint, TrainingMetrics, ModelRegistry
from ..database.connection import db_manager
from .dora_trainer import DoRATrainer
from .qr_adaptor import QRAdaptor, QuantizationConfig, RankConfig
from ..optimization.intel_optimizer import IntelCPUOptimizer, CPUConfig
from ..optimization.system_optimizer import SystemOptimizer, SystemConfig
from ..optimization.memory_optimizer import MemoryOptimizer, MemoryConfig

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types"""
    DORA = "dora"
    QR_ADAPTOR = "qr_adaptor"
    HYBRID = "hybrid"

class TrainingStatus(Enum):
    """Training status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str
    model_type: ModelType
    base_model: str
    dora_rank: int = 64
    dora_alpha: float = 16.0
    target_modules: List[str] = None
    quantization_config: Optional[QuantizationConfig] = None
    rank_config: Optional[RankConfig] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 4
    mixed_precision: bool = True
    gradient_checkpointing: bool = True

class ModelManager:
    """
    Central model management system for training coordination
    """
    
    def __init__(self):
        self.active_trainers = {}
        self.training_sessions = {}
        self.model_registry = {}
        self.checkpoint_manager = CheckpointManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize optimizers
        self.intel_optimizer = IntelCPUOptimizer(CPUConfig())
        self.system_optimizer = SystemOptimizer(SystemConfig())
        self.memory_optimizer = MemoryOptimizer(MemoryConfig())
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Model Manager initialized")
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        try:
            # Start performance monitoring
            self.system_optimizer.start_performance_monitoring()
            
            # Start memory monitoring
            self.memory_optimizer.monitor_memory_usage()
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
    
    async def create_training_session(self, model_config: ModelConfig, 
                                    training_config: TrainingConfig,
                                    dataset_config: Dict[str, Any]) -> str:
        """Create a new training session"""
        try:
            session_id = str(uuid.uuid4())
            
            # Create database record
            with db_manager.get_session_context() as session:
                db_session = TrainingSession(
                    id=session_id,
                    model_name=model_config.model_name,
                    model_type=model_config.model_type.value,
                    status=TrainingStatus.PENDING.value,
                    config={
                        'model_config': model_config.__dict__,
                        'training_config': training_config.__dict__,
                        'dataset_config': dataset_config
                    }
                )
                session.add(db_session)
                session.commit()
            
            # Store in memory
            self.training_sessions[session_id] = {
                'model_config': model_config,
                'training_config': training_config,
                'dataset_config': dataset_config,
                'status': TrainingStatus.PENDING,
                'created_at': datetime.utcnow()
            }
            
            logger.info(f"Created training session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create training session: {e}")
            raise
    
    async def start_training_session(self, session_id: str) -> bool:
        """Start training session"""
        try:
            if session_id not in self.training_sessions:
                raise ValueError(f"Training session {session_id} not found")
            
            session_data = self.training_sessions[session_id]
            model_config = session_data['model_config']
            training_config = session_data['training_config']
            
            # Update status to running
            await self._update_session_status(session_id, TrainingStatus.RUNNING)
            
            # Create trainer based on model type
            if model_config.model_type == ModelType.DORA:
                trainer = DoRATrainer(model_config.__dict__)
            elif model_config.model_type == ModelType.QR_ADAPTOR:
                trainer = QRAdaptor(
                    model_config.quantization_config or QuantizationConfig(),
                    model_config.rank_config or RankConfig()
                )
            else:
                raise ValueError(f"Unsupported model type: {model_config.model_type}")
            
            # Load and optimize model
            model, tokenizer = trainer.load_model()
            model = self.intel_optimizer.optimize_model_for_cpu(model)
            
            # Setup optimizers
            trainer.setup_optimizers(
                learning_rate=training_config.learning_rate,
                weight_decay=training_config.weight_decay
            )
            
            # Store active trainer
            self.active_trainers[session_id] = {
                'trainer': trainer,
                'model': model,
                'tokenizer': tokenizer,
                'start_time': datetime.utcnow(),
                'status': TrainingStatus.RUNNING
            }
            
            # Start training in background
            training_task = asyncio.create_task(
                self._run_training_loop(session_id, trainer, model, training_config)
            )
            
            logger.info(f"Started training session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training session: {e}")
            await self._update_session_status(session_id, TrainingStatus.FAILED, str(e))
            return False
    
    async def _run_training_loop(self, session_id: str, trainer: Any, model: Any, 
                               training_config: TrainingConfig):
        """Run training loop"""
        try:
            # Get dataset
            dataset = await self._load_training_dataset(session_id)
            
            # Training loop
            for epoch in range(training_config.epochs):
                if session_id not in self.active_trainers:
                    break  # Training cancelled
                
                # Update epoch
                await self._update_session_epoch(session_id, epoch)
                
                # Train epoch
                epoch_metrics = await self._train_epoch(
                    session_id, trainer, model, dataset, training_config, epoch
                )
                
                # Save checkpoint
                if epoch % training_config.save_steps == 0:
                    await self._save_checkpoint(session_id, trainer, epoch, epoch_metrics)
                
                # Log metrics
                await self._log_training_metrics(session_id, epoch, epoch_metrics)
            
            # Mark as completed
            await self._update_session_status(session_id, TrainingStatus.COMPLETED)
            
            # Register model
            await self._register_trained_model(session_id, trainer)
            
        except Exception as e:
            logger.error(f"Training loop failed for session {session_id}: {e}")
            await self._update_session_status(session_id, TrainingStatus.FAILED, str(e))
        
        finally:
            # Cleanup
            if session_id in self.active_trainers:
                del self.active_trainers[session_id]
    
    async def _train_epoch(self, session_id: str, trainer: Any, model: Any, 
                          dataset: Any, training_config: TrainingConfig, epoch: int) -> Dict[str, float]:
        """Train single epoch"""
        try:
            epoch_metrics = {
                'epoch': epoch,
                'total_loss': 0.0,
                'total_steps': 0,
                'learning_rate': 0.0
            }
            
            # Create data loader
            dataloader = self._create_dataloader(dataset, training_config)
            
            for step, batch in enumerate(dataloader):
                if session_id not in self.active_trainers:
                    break  # Training cancelled
                
                # Training step
                step_metrics = trainer.train_step(batch)
                
                # Accumulate metrics
                epoch_metrics['total_loss'] += step_metrics['loss']
                epoch_metrics['total_steps'] += 1
                epoch_metrics['learning_rate'] = step_metrics.get('learning_rate_direction', 0.0)
                
                # Update session progress
                await self._update_session_progress(session_id, epoch, step, step_metrics)
                
                # Log step metrics
                if step % training_config.logging_steps == 0:
                    logger.info(f"Session {session_id} - Epoch {epoch}, Step {step}, Loss: {step_metrics['loss']:.4f}")
            
            # Calculate average metrics
            if epoch_metrics['total_steps'] > 0:
                epoch_metrics['avg_loss'] = epoch_metrics['total_loss'] / epoch_metrics['total_steps']
            
            return epoch_metrics
            
        except Exception as e:
            logger.error(f"Failed to train epoch {epoch} for session {session_id}: {e}")
            raise
    
    async def _load_training_dataset(self, session_id: str) -> Any:
        """Load training dataset"""
        try:
            session_data = self.training_sessions[session_id]
            dataset_config = session_data['dataset_config']
            
            # Load dataset based on configuration
            # This would integrate with the Persian data processor
            # For now, return a placeholder
            return {"placeholder": "dataset"}
            
        except Exception as e:
            logger.error(f"Failed to load training dataset: {e}")
            raise
    
    def _create_dataloader(self, dataset: Any, training_config: TrainingConfig) -> Any:
        """Create data loader"""
        try:
            # Create data loader with optimized settings
            dataloader_config = self.intel_optimizer.optimize_data_loading(
                num_workers=training_config.dataloader_num_workers
            )
            
            # This would create actual PyTorch DataLoader
            # For now, return placeholder
            return [{"placeholder": "batch"} for _ in range(10)]
            
        except Exception as e:
            logger.error(f"Failed to create data loader: {e}")
            raise
    
    async def _update_session_status(self, session_id: str, status: TrainingStatus, 
                                   error_message: Optional[str] = None):
        """Update training session status"""
        try:
            # Update in memory
            if session_id in self.training_sessions:
                self.training_sessions[session_id]['status'] = status
            
            # Update in database
            with db_manager.get_session_context() as session:
                db_session = session.query(TrainingSession).filter_by(id=session_id).first()
                if db_session:
                    db_session.status = status.value
                    if error_message:
                        db_session.error_message = error_message
                    if status == TrainingStatus.RUNNING and not db_session.started_at:
                        db_session.started_at = datetime.utcnow()
                    elif status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]:
                        db_session.completed_at = datetime.utcnow()
                    session.commit()
            
        except Exception as e:
            logger.error(f"Failed to update session status: {e}")
    
    async def _update_session_epoch(self, session_id: str, epoch: int):
        """Update session epoch"""
        try:
            with db_manager.get_session_context() as session:
                db_session = session.query(TrainingSession).filter_by(id=session_id).first()
                if db_session:
                    db_session.current_epoch = epoch
                    session.commit()
            
        except Exception as e:
            logger.error(f"Failed to update session epoch: {e}")
    
    async def _update_session_progress(self, session_id: str, epoch: int, step: int, 
                                     step_metrics: Dict[str, float]):
        """Update session progress"""
        try:
            with db_manager.get_session_context() as session:
                db_session = session.query(TrainingSession).filter_by(id=session_id).first()
                if db_session:
                    db_session.current_step = step
                    db_session.current_loss = step_metrics.get('loss')
                    db_session.learning_rate = step_metrics.get('learning_rate_direction')
                    db_session.cpu_usage = step_metrics.get('cpu_usage')
                    db_session.memory_usage = step_metrics.get('memory_usage_mb')
                    session.commit()
            
        except Exception as e:
            logger.error(f"Failed to update session progress: {e}")
    
    async def _save_checkpoint(self, session_id: str, trainer: Any, epoch: int, 
                             epoch_metrics: Dict[str, float]):
        """Save training checkpoint"""
        try:
            checkpoint_dir = f"checkpoints/{session_id}"
            checkpoint_path = trainer.save_checkpoint(checkpoint_dir, epoch, epoch_metrics)
            
            # Save to database
            with db_manager.get_session_context() as session:
                checkpoint = ModelCheckpoint(
                    session_id=session_id,
                    epoch=epoch,
                    step=epoch_metrics.get('total_steps', 0),
                    checkpoint_type='epoch_end',
                    loss=epoch_metrics.get('avg_loss', 0.0),
                    learning_rate=epoch_metrics.get('learning_rate', 0.0),
                    file_path=checkpoint_path,
                    file_size_mb=os.path.getsize(checkpoint_path) / (1024 * 1024) if os.path.exists(checkpoint_path) else 0.0
                )
                session.add(checkpoint)
                session.commit()
            
            logger.info(f"Saved checkpoint for session {session_id}, epoch {epoch}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    async def _log_training_metrics(self, session_id: str, epoch: int, 
                                  epoch_metrics: Dict[str, float]):
        """Log training metrics to database"""
        try:
            with db_manager.get_session_context() as session:
                metrics = TrainingMetrics(
                    session_id=session_id,
                    epoch=epoch,
                    step=epoch_metrics.get('total_steps', 0),
                    loss=epoch_metrics.get('avg_loss', 0.0),
                    learning_rate=epoch_metrics.get('learning_rate', 0.0),
                    cpu_usage=self.system_optimizer.collect_system_metrics().get('cpu', {}).get('average_percent', 0.0),
                    memory_usage_mb=self.system_optimizer.collect_system_metrics().get('memory', {}).get('used_gb', 0.0) * 1024
                )
                session.add(metrics)
                session.commit()
            
        except Exception as e:
            logger.error(f"Failed to log training metrics: {e}")
    
    async def _register_trained_model(self, session_id: str, trainer: Any):
        """Register trained model in registry"""
        try:
            session_data = self.training_sessions[session_id]
            model_config = session_data['model_config']
            
            # Get best checkpoint
            with db_manager.get_session_context() as session:
                best_checkpoint = session.query(ModelCheckpoint).filter_by(
                    session_id=session_id,
                    checkpoint_type='best'
                ).first()
                
                if best_checkpoint:
                    # Register model
                    model_registry = ModelRegistry(
                        name=f"{model_config.model_name}_{session_id[:8]}",
                        model_type=model_config.model_type.value,
                        base_model=model_config.base_model,
                        training_session_id=session_id,
                        best_accuracy=best_checkpoint.accuracy,
                        best_loss=best_checkpoint.loss,
                        model_path=best_checkpoint.file_path,
                        is_active=True
                    )
                    session.add(model_registry)
                    session.commit()
                    
                    logger.info(f"Registered trained model: {model_registry.name}")
            
        except Exception as e:
            logger.error(f"Failed to register trained model: {e}")
    
    async def pause_training_session(self, session_id: str) -> bool:
        """Pause training session"""
        try:
            if session_id in self.active_trainers:
                await self._update_session_status(session_id, TrainingStatus.PAUSED)
                logger.info(f"Paused training session: {session_id}")
                return True
            else:
                logger.warning(f"Training session {session_id} not found or not active")
                return False
                
        except Exception as e:
            logger.error(f"Failed to pause training session: {e}")
            return False
    
    async def resume_training_session(self, session_id: str) -> bool:
        """Resume training session"""
        try:
            if session_id in self.training_sessions:
                session_data = self.training_sessions[session_id]
                if session_data['status'] == TrainingStatus.PAUSED:
                    return await self.start_training_session(session_id)
                else:
                    logger.warning(f"Training session {session_id} is not paused")
                    return False
            else:
                logger.warning(f"Training session {session_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to resume training session: {e}")
            return False
    
    async def cancel_training_session(self, session_id: str) -> bool:
        """Cancel training session"""
        try:
            if session_id in self.active_trainers:
                # Remove from active trainers
                del self.active_trainers[session_id]
                
                # Update status
                await self._update_session_status(session_id, TrainingStatus.CANCELLED)
                
                logger.info(f"Cancelled training session: {session_id}")
                return True
            else:
                logger.warning(f"Training session {session_id} not found or not active")
                return False
                
        except Exception as e:
            logger.error(f"Failed to cancel training session: {e}")
            return False
    
    async def get_training_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get training session status"""
        try:
            # Get from database
            with db_manager.get_session_context() as session:
                db_session = session.query(TrainingSession).filter_by(id=session_id).first()
                if db_session:
                    return db_session.to_dict()
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get training session status: {e}")
            return None
    
    async def get_training_metrics(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get training metrics for session"""
        try:
            with db_manager.get_session_context() as session:
                metrics = session.query(TrainingMetrics).filter_by(
                    session_id=session_id
                ).order_by(TrainingMetrics.timestamp.desc()).limit(limit).all()
                
                return [metric.to_dict() for metric in metrics]
            
        except Exception as e:
            logger.error(f"Failed to get training metrics: {e}")
            return []
    
    async def list_training_sessions(self, status: Optional[TrainingStatus] = None) -> List[Dict[str, Any]]:
        """List training sessions"""
        try:
            with db_manager.get_session_context() as session:
                query = session.query(TrainingSession)
                if status:
                    query = query.filter_by(status=status.value)
                
                sessions = query.order_by(TrainingSession.created_at.desc()).all()
                return [s.to_dict() for s in sessions]
            
        except Exception as e:
            logger.error(f"Failed to list training sessions: {e}")
            return []
    
    async def get_system_performance(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            return {
                'system_metrics': self.system_optimizer.get_performance_summary(),
                'memory_metrics': self.memory_optimizer.get_memory_usage(),
                'intel_optimization': self.intel_optimizer.get_optimization_summary(),
                'active_sessions': len(self.active_trainers),
                'total_sessions': len(self.training_sessions)
            }
            
        except Exception as e:
            logger.error(f"Failed to get system performance: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup model manager"""
        try:
            # Cancel all active training sessions
            for session_id in list(self.active_trainers.keys()):
                await self.cancel_training_session(session_id)
            
            # Cleanup optimizers
            self.system_optimizer.cleanup()
            self.memory_optimizer.cleanup()
            self.intel_optimizer.cleanup()
            
            logger.info("Model Manager cleanup complete")
            
        except Exception as e:
            logger.error(f"Failed to cleanup model manager: {e}")

class CheckpointManager:
    """Manages model checkpoints"""
    
    def __init__(self):
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, session_id: str, epoch: int, metrics: Dict[str, float]) -> str:
        """Save checkpoint"""
        try:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{session_id}_epoch_{epoch}.pt")
            # Implementation would save actual model checkpoint
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint"""
        try:
            # Implementation would load actual model checkpoint
            return {"checkpoint": "loaded"}
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

class PerformanceMonitor:
    """Monitors training performance"""
    
    def __init__(self):
        self.metrics_history = []
    
    def record_metrics(self, session_id: str, metrics: Dict[str, float]):
        """Record performance metrics"""
        try:
            self.metrics_history.append({
                'session_id': session_id,
                'timestamp': datetime.utcnow(),
                'metrics': metrics
            })
            
            # Keep only last 1000 entries
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
                
        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            if not self.metrics_history:
                return {}
            
            # Calculate averages
            recent_metrics = self.metrics_history[-100:] if len(self.metrics_history) >= 100 else self.metrics_history
            
            avg_loss = sum(m['metrics'].get('loss', 0) for m in recent_metrics) / len(recent_metrics)
            avg_cpu = sum(m['metrics'].get('cpu_usage', 0) for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m['metrics'].get('memory_usage_mb', 0) for m in recent_metrics) / len(recent_metrics)
            
            return {
                'average_loss': avg_loss,
                'average_cpu_usage': avg_cpu,
                'average_memory_usage_mb': avg_memory,
                'total_metrics_recorded': len(self.metrics_history)
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}