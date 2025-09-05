"""
Multi-GPU Training Support for Persian Legal AI
پشتیبانی آموزش چند GPU برای هوش مصنوعی حقوقی فارسی
"""

import logging
import os
import time
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import psutil

logger = logging.getLogger(__name__)

class MultiGPUTrainer:
    """Multi-GPU trainer with checkpoint resume and 24/7 support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        self.training_start_time = None
        self.is_training = False
        self.should_stop = False
        
        # Multi-GPU setup
        self.world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.rank = 0
        self.local_rank = 0
        
        # Performance monitoring
        self.gpu_utilization = []
        self.memory_usage = []
        self.training_metrics = []
        
        logger.info(f"Multi-GPU Trainer initialized with {self.world_size} GPUs")
    
    def setup_distributed_training(self, rank: int, world_size: int):
        """Setup distributed training environment"""
        try:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            
            # Initialize the process group
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
            
            self.rank = rank
            self.local_rank = rank
            self.world_size = world_size
            
            logger.info(f"Distributed training setup complete for rank {rank}/{world_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup distributed training: {e}")
            return False
    
    def cleanup_distributed_training(self):
        """Cleanup distributed training environment"""
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
            logger.info("Distributed training cleanup complete")
        except Exception as e:
            logger.error(f"Failed to cleanup distributed training: {e}")
    
    def save_checkpoint(self, model, optimizer, scheduler=None, epoch: int = None, step: int = None):
        """Save training checkpoint"""
        try:
            if epoch is None:
                epoch = self.current_epoch
            if step is None:
                step = self.current_step
            
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': self.best_loss,
                'training_start_time': self.training_start_time.isoformat() if self.training_start_time else None,
                'config': self.config,
                'gpu_utilization': self.gpu_utilization,
                'memory_usage': self.memory_usage,
                'training_metrics': self.training_metrics
            }
            
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            # Save latest checkpoint
            latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
            torch.save(checkpoint, latest_path)
            
            # Save best checkpoint if current loss is better
            if hasattr(self, 'current_loss') and self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
                best_path = self.checkpoint_dir / "best_checkpoint.pt"
                torch.save(checkpoint, best_path)
                logger.info(f"New best checkpoint saved with loss: {self.current_loss}")
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_path: str = None, model=None, optimizer=None, scheduler=None):
        """Load training checkpoint"""
        try:
            if checkpoint_path is None:
                checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pt"
            
            if not os.path.exists(checkpoint_path):
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return False
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Restore training state
            self.current_epoch = checkpoint.get('epoch', 0)
            self.current_step = checkpoint.get('step', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            
            training_start_time_str = checkpoint.get('training_start_time')
            if training_start_time_str:
                self.training_start_time = datetime.fromisoformat(training_start_time_str)
            
            self.gpu_utilization = checkpoint.get('gpu_utilization', [])
            self.memory_usage = checkpoint.get('memory_usage', [])
            self.training_metrics = checkpoint.get('training_metrics', [])
            
            # Restore model and optimizer states
            if model is not None:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            logger.info(f"Resuming from epoch {self.current_epoch}, step {self.current_step}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def get_available_checkpoints(self) -> List[Dict[str, Any]]:
        """Get list of available checkpoints"""
        checkpoints = []
        
        try:
            for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pt"):
                try:
                    checkpoint = torch.load(checkpoint_file, map_location='cpu')
                    checkpoints.append({
                        'file': str(checkpoint_file),
                        'epoch': checkpoint.get('epoch', 0),
                        'step': checkpoint.get('step', 0),
                        'best_loss': checkpoint.get('best_loss', float('inf')),
                        'created_at': datetime.fromtimestamp(checkpoint_file.stat().st_mtime).isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")
            
            # Sort by epoch and step
            checkpoints.sort(key=lambda x: (x['epoch'], x['step']))
            return checkpoints
            
        except Exception as e:
            logger.error(f"Failed to get available checkpoints: {e}")
            return []
    
    def monitor_gpu_usage(self):
        """Monitor GPU usage and memory"""
        try:
            if torch.cuda.is_available():
                gpu_util = []
                gpu_memory = []
                
                for i in range(torch.cuda.device_count()):
                    # Get GPU utilization (simplified - in real implementation use nvidia-ml-py)
                    gpu_util.append(0.0)  # Placeholder
                    
                    # Get GPU memory usage
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                    gpu_memory.append({
                        'allocated': memory_allocated,
                        'reserved': memory_reserved,
                        'total': torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    })
                
                self.gpu_utilization.append({
                    'timestamp': datetime.now().isoformat(),
                    'utilization': gpu_util,
                    'memory': gpu_memory
                })
                
                # Keep only last 1000 entries
                if len(self.gpu_utilization) > 1000:
                    self.gpu_utilization = self.gpu_utilization[-1000:]
                    
        except Exception as e:
            logger.error(f"Failed to monitor GPU usage: {e}")
    
    def monitor_system_resources(self):
        """Monitor system resources"""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            self.memory_usage.append({
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3)
            })
            
            # Keep only last 1000 entries
            if len(self.memory_usage) > 1000:
                self.memory_usage = self.memory_usage[-1000:]
                
        except Exception as e:
            logger.error(f"Failed to monitor system resources: {e}")
    
    def log_training_metrics(self, metrics: Dict[str, Any]):
        """Log training metrics"""
        try:
            self.training_metrics.append({
                'timestamp': datetime.now().isoformat(),
                'epoch': self.current_epoch,
                'step': self.current_step,
                'metrics': metrics
            })
            
            # Keep only last 10000 entries
            if len(self.training_metrics) > 10000:
                self.training_metrics = self.training_metrics[-10000:]
                
        except Exception as e:
            logger.error(f"Failed to log training metrics: {e}")
    
    def create_distributed_dataloader(self, dataset, batch_size: int, num_workers: int = 4):
        """Create distributed data loader"""
        try:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            )
            
            return dataloader, sampler
            
        except Exception as e:
            logger.error(f"Failed to create distributed dataloader: {e}")
            return None, None
    
    def wrap_model_with_ddp(self, model):
        """Wrap model with DistributedDataParallel"""
        try:
            if self.world_size > 1:
                model = DDP(model, device_ids=[self.local_rank])
            return model
        except Exception as e:
            logger.error(f"Failed to wrap model with DDP: {e}")
            return model
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        try:
            status = {
                'is_training': self.is_training,
                'current_epoch': self.current_epoch,
                'current_step': self.current_step,
                'best_loss': self.best_loss,
                'world_size': self.world_size,
                'rank': self.rank,
                'training_start_time': self.training_start_time.isoformat() if self.training_start_time else None,
                'uptime_seconds': (datetime.now() - self.training_start_time).total_seconds() if self.training_start_time else 0,
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'checkpoint_count': len(self.get_available_checkpoints())
            }
            
            # Add recent metrics
            if self.training_metrics:
                recent_metrics = self.training_metrics[-10:]  # Last 10 entries
                status['recent_metrics'] = recent_metrics
            
            # Add GPU utilization
            if self.gpu_utilization:
                recent_gpu = self.gpu_utilization[-1]  # Latest entry
                status['gpu_utilization'] = recent_gpu
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get training status: {e}")
            return {'error': str(e)}
    
    def start_24_7_training(self, model, dataset, optimizer, scheduler=None):
        """Start 24/7 training with automatic checkpointing and resume"""
        try:
            self.is_training = True
            self.should_stop = False
            self.training_start_time = datetime.now()
            
            logger.info("Starting 24/7 training mode")
            
            # Try to resume from checkpoint
            if self.load_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler):
                logger.info("Resumed training from checkpoint")
            else:
                logger.info("Starting fresh training")
            
            # Create distributed dataloader
            dataloader, sampler = self.create_distributed_dataloader(
                dataset, 
                self.config.get('batch_size', 8),
                self.config.get('num_workers', 4)
            )
            
            if dataloader is None:
                raise ValueError("Failed to create distributed dataloader")
            
            # Wrap model with DDP
            model = self.wrap_model_with_ddp(model)
            
            # Training loop
            while not self.should_stop:
                try:
                    # Set epoch for distributed sampler
                    if sampler is not None:
                        sampler.set_epoch(self.current_epoch)
                    
                    # Training epoch
                    epoch_loss = self.train_epoch(model, dataloader, optimizer, scheduler)
                    
                    # Log metrics
                    self.log_training_metrics({
                        'epoch_loss': epoch_loss,
                        'learning_rate': optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0
                    })
                    
                    # Monitor resources
                    self.monitor_gpu_usage()
                    self.monitor_system_resources()
                    
                    # Save checkpoint
                    if self.current_epoch % self.config.get('checkpoint_interval', 1) == 0:
                        self.save_checkpoint(model, optimizer, scheduler)
                    
                    self.current_epoch += 1
                    
                    # Check if we should continue training
                    if self.current_epoch >= self.config.get('max_epochs', 100):
                        logger.info("Reached maximum epochs, stopping training")
                        break
                        
                except Exception as e:
                    logger.error(f"Error in training epoch {self.current_epoch}: {e}")
                    # Save emergency checkpoint
                    self.save_checkpoint(model, optimizer, scheduler)
                    break
            
            self.is_training = False
            logger.info("24/7 training completed")
            
        except Exception as e:
            logger.error(f"Failed to start 24/7 training: {e}")
            self.is_training = False
            raise
    
    def train_epoch(self, model, dataloader, optimizer, scheduler=None):
        """Train one epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if self.should_stop:
                break
                
            try:
                optimizer.zero_grad()
                
                # Forward pass (this would be model-specific)
                # loss = model(batch)
                # For now, use a placeholder
                loss = torch.tensor(0.0, requires_grad=True)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                self.current_step += 1
                
                # Log progress
                if batch_idx % self.config.get('log_interval', 100) == 0:
                    logger.info(f"Epoch {self.current_epoch}, Step {self.current_step}, Loss: {loss.item():.4f}")
                
            except Exception as e:
                logger.error(f"Error in training step {self.current_step}: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def stop_training(self):
        """Stop training gracefully"""
        logger.info("Stopping training...")
        self.should_stop = True
        self.is_training = False
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.cleanup_distributed_training()
            logger.info("Multi-GPU trainer cleanup complete")
        except Exception as e:
            logger.error(f"Failed to cleanup multi-GPU trainer: {e}")

# Global trainer instance
multi_gpu_trainer = None

def get_multi_gpu_trainer(config: Dict[str, Any] = None) -> MultiGPUTrainer:
    """Get or create multi-GPU trainer instance"""
    global multi_gpu_trainer
    
    if multi_gpu_trainer is None:
        if config is None:
            config = {
                'checkpoint_dir': './checkpoints',
                'batch_size': 8,
                'num_workers': 4,
                'checkpoint_interval': 1,
                'max_epochs': 100,
                'log_interval': 100
            }
        multi_gpu_trainer = MultiGPUTrainer(config)
    
    return multi_gpu_trainer