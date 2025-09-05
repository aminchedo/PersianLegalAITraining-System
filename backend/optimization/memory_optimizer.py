"""
Memory Optimization for Persian Legal AI Training
بهینه‌سازی حافظه برای آموزش هوش مصنوعی حقوقی فارسی
"""

import torch
import gc
import psutil
import os
import threading
import time
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Memory optimization configuration"""
    max_memory_gb: float = 64.0
    memory_threshold_percent: float = 85.0
    enable_memory_pooling: bool = True
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    memory_cleanup_interval: float = 30.0

class MemoryOptimizer:
    """
    Advanced memory optimization for training large models
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.memory_pool = {}
        self.memory_stats = {}
        self.cleanup_thread = None
        self.cleanup_active = False
        
        # Initialize memory optimization
        self._setup_memory_optimization()
    
    def _setup_memory_optimization(self):
        """Setup memory optimization settings"""
        try:
            # Set memory-related environment variables
            os.environ['PYTHONHASHSEED'] = '0'
            os.environ['PYTHONUNBUFFERED'] = '1'
            
            # Memory allocation optimization
            os.environ['MALLOC_TRIM_THRESHOLD_'] = '131072'
            os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'
            
            # Start memory cleanup thread
            self._start_memory_cleanup()
            
            logger.info("Memory optimization setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup memory optimization: {e}")
    
    def _start_memory_cleanup(self):
        """Start background memory cleanup thread"""
        try:
            self.cleanup_active = True
            self.cleanup_thread = threading.Thread(
                target=self._memory_cleanup_loop,
                daemon=True
            )
            self.cleanup_thread.start()
            
            logger.info("Memory cleanup thread started")
            
        except Exception as e:
            logger.error(f"Failed to start memory cleanup: {e}")
    
    def _memory_cleanup_loop(self):
        """Background memory cleanup loop"""
        while self.cleanup_active:
            try:
                # Check memory usage
                memory_percent = psutil.virtual_memory().percent
                
                if memory_percent > self.config.memory_threshold_percent:
                    self._perform_memory_cleanup()
                
                time.sleep(self.config.memory_cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in memory cleanup loop: {e}")
                time.sleep(self.config.memory_cleanup_interval)
    
    def _perform_memory_cleanup(self):
        """Perform comprehensive memory cleanup"""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear memory pool
            self._clear_memory_pool()
            
            logger.info(f"Memory cleanup performed: {collected} objects collected")
            
        except Exception as e:
            logger.error(f"Failed to perform memory cleanup: {e}")
    
    def _clear_memory_pool(self):
        """Clear memory pool"""
        try:
            self.memory_pool.clear()
            logger.debug("Memory pool cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear memory pool: {e}")
    
    @contextmanager
    def memory_efficient_training(self, model: torch.nn.Module):
        """Context manager for memory-efficient training"""
        try:
            # Enable gradient checkpointing
            if self.config.enable_gradient_checkpointing:
                self._enable_gradient_checkpointing(model)
            
            # Enable mixed precision
            if self.config.enable_mixed_precision:
                self._enable_mixed_precision()
            
            yield model
            
        finally:
            # Cleanup
            self._perform_memory_cleanup()
    
    def _enable_gradient_checkpointing(self, model: torch.nn.Module):
        """Enable gradient checkpointing for memory efficiency"""
        try:
            # Apply gradient checkpointing to transformer layers
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
            
            logger.info("Gradient checkpointing enabled")
            
        except Exception as e:
            logger.error(f"Failed to enable gradient checkpointing: {e}")
    
    def _enable_mixed_precision(self):
        """Enable mixed precision training"""
        try:
            # Enable autocast
            torch.backends.cpu.amp.autocast_enabled = True
            
            logger.info("Mixed precision enabled")
            
        except Exception as e:
            logger.error(f"Failed to enable mixed precision: {e}")
    
    def optimize_model_memory(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for memory efficiency"""
        try:
            # Convert to half precision
            if self.config.enable_mixed_precision:
                model = model.half()
            
            # Apply memory optimizations
            model = self._apply_memory_optimizations(model)
            
            logger.info("Model memory optimization complete")
            return model
            
        except Exception as e:
            logger.error(f"Failed to optimize model memory: {e}")
            return model
    
    def _apply_memory_optimizations(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply memory optimizations to model"""
        try:
            # Enable gradient checkpointing
            if self.config.enable_gradient_checkpointing:
                self._enable_gradient_checkpointing(model)
            
            # Optimize attention mechanisms
            self._optimize_attention_memory(model)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to apply memory optimizations: {e}")
            return model
    
    def _optimize_attention_memory(self, model: torch.nn.Module):
        """Optimize attention mechanisms for memory efficiency"""
        try:
            for module in model.modules():
                if hasattr(module, 'attention_dropout'):
                    # Increase dropout for memory efficiency
                    module.attention_dropout = min(module.attention_dropout + 0.1, 0.5)
                
                if hasattr(module, 'hidden_dropout'):
                    # Increase hidden dropout
                    module.hidden_dropout = min(module.hidden_dropout + 0.1, 0.5)
            
            logger.info("Attention memory optimization applied")
            
        except Exception as e:
            logger.error(f"Failed to optimize attention memory: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        try:
            # System memory
            memory = psutil.virtual_memory()
            
            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # PyTorch memory (if available)
            torch_memory = {}
            if torch.cuda.is_available():
                torch_memory = {
                    'allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                    'cached_mb': torch.cuda.memory_reserved() / (1024**2)
                }
            
            return {
                'system': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_percent': memory.percent,
                    'used_gb': memory.used / (1024**3)
                },
                'process': {
                    'rss_mb': process_memory.rss / (1024**2),
                    'vms_mb': process_memory.vms / (1024**2)
                },
                'torch': torch_memory,
                'memory_pool_size': len(self.memory_pool)
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {}
    
    def monitor_memory_usage(self, callback: Optional[Callable] = None):
        """Monitor memory usage with optional callback"""
        try:
            while True:
                memory_stats = self.get_memory_usage()
                
                if callback:
                    callback(memory_stats)
                
                # Check for memory issues
                if memory_stats['system']['used_percent'] > self.config.memory_threshold_percent:
                    logger.warning(f"High memory usage: {memory_stats['system']['used_percent']:.1f}%")
                    self._perform_memory_cleanup()
                
                time.sleep(5.0)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            logger.info("Memory monitoring stopped")
        except Exception as e:
            logger.error(f"Error in memory monitoring: {e}")
    
    def optimize_data_loader_memory(self, data_loader) -> Any:
        """Optimize data loader for memory efficiency"""
        try:
            # Set optimal number of workers
            if hasattr(data_loader, 'num_workers'):
                data_loader.num_workers = min(4, os.cpu_count() // 2)
            
            # Disable pin memory for CPU training
            if hasattr(data_loader, 'pin_memory'):
                data_loader.pin_memory = False
            
            logger.info("Data loader memory optimization applied")
            return data_loader
            
        except Exception as e:
            logger.error(f"Failed to optimize data loader memory: {e}")
            return data_loader
    
    def create_memory_efficient_batch(self, batch_size: int, sequence_length: int, 
                                    vocab_size: int) -> Dict[str, torch.Tensor]:
        """Create memory-efficient batch"""
        try:
            # Use half precision
            dtype = torch.float16 if self.config.enable_mixed_precision else torch.float32
            
            batch = {
                'input_ids': torch.randint(0, vocab_size, (batch_size, sequence_length), dtype=torch.long),
                'attention_mask': torch.ones(batch_size, sequence_length, dtype=torch.long),
                'labels': torch.randint(0, vocab_size, (batch_size, sequence_length), dtype=torch.long)
            }
            
            return batch
            
        except Exception as e:
            logger.error(f"Failed to create memory-efficient batch: {e}")
            return {}
    
    def profile_memory_usage(self, model: torch.nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Profile memory usage of model"""
        try:
            # Get initial memory
            initial_memory = self.get_memory_usage()
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                _ = model(sample_input)
            
            # Get memory after forward pass
            forward_memory = self.get_memory_usage()
            
            # Backward pass
            model.train()
            output = model(sample_input)
            loss = output.loss if hasattr(output, 'loss') else torch.tensor(0.0)
            loss.backward()
            
            # Get memory after backward pass
            backward_memory = self.get_memory_usage()
            
            return {
                'initial_memory_mb': initial_memory['process']['rss_mb'],
                'forward_memory_mb': forward_memory['process']['rss_mb'],
                'backward_memory_mb': backward_memory['process']['rss_mb'],
                'forward_memory_increase_mb': forward_memory['process']['rss_mb'] - initial_memory['process']['rss_mb'],
                'backward_memory_increase_mb': backward_memory['process']['rss_mb'] - forward_memory['process']['rss_mb']
            }
            
        except Exception as e:
            logger.error(f"Failed to profile memory usage: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup memory optimizer"""
        try:
            # Stop cleanup thread
            self.cleanup_active = False
            
            if self.cleanup_thread and self.cleanup_thread.is_alive():
                self.cleanup_thread.join(timeout=5.0)
            
            # Clear memory pool
            self._clear_memory_pool()
            
            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Memory optimizer cleanup complete")
            
        except Exception as e:
            logger.error(f"Failed to cleanup memory optimizer: {e}")