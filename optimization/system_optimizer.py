"""
Real Platform-Agnostic System Optimizer
بهینه‌ساز واقعی سیستم مستقل از سکو
"""

import os
import logging
import multiprocessing
import psutil
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import torch
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SystemResources:
    """System resource information"""
    cpu_cores: int
    cpu_usage: float
    memory_total_gb: float
    memory_available_gb: float
    memory_usage_percent: float
    disk_total_gb: float
    disk_free_gb: float
    disk_usage_percent: float
    gpu_available: bool
    gpu_count: int = 0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0

@dataclass
class OptimizationConfig:
    """Optimization configuration"""
    max_cpu_usage: float = 80.0
    max_memory_usage: float = 80.0
    max_disk_usage: float = 90.0
    enable_gpu_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    num_workers: Optional[int] = None
    batch_size_multiplier: float = 1.0

class SystemOptimizer:
    """
    Real platform-agnostic system optimizer
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.resources = None
        self.optimization_history = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Initialize optimization
        self._initialize_optimization()
    
    def _initialize_optimization(self):
        """Initialize system optimization"""
        try:
            # Get system resources
            self.resources = self._get_system_resources()
            
            # Apply initial optimizations
            self._apply_initial_optimizations()
            
            logger.info("System optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system optimizer: {e}")
            raise
    
    def _get_system_resources(self) -> SystemResources:
        """Get current system resource information"""
        try:
            # CPU information
            cpu_cores = multiprocessing.cpu_count()
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            memory_usage_percent = memory.percent
            
            # Disk information
            disk = psutil.disk_usage('/')
            disk_total_gb = disk.total / (1024**3)
            disk_free_gb = disk.free / (1024**3)
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # GPU information
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if gpu_available else 0
            gpu_memory_total_gb = 0.0
            gpu_memory_used_gb = 0.0
            
            if gpu_available and gpu_count > 0:
                gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_memory_used_gb = torch.cuda.memory_allocated(0) / (1024**3)
            
            return SystemResources(
                cpu_cores=cpu_cores,
                cpu_usage=cpu_usage,
                memory_total_gb=memory_total_gb,
                memory_available_gb=memory_available_gb,
                memory_usage_percent=memory_usage_percent,
                disk_total_gb=disk_total_gb,
                disk_free_gb=disk_free_gb,
                disk_usage_percent=disk_usage_percent,
                gpu_available=gpu_available,
                gpu_count=gpu_count,
                gpu_memory_total_gb=gpu_memory_total_gb,
                gpu_memory_used_gb=gpu_memory_used_gb
            )
            
        except Exception as e:
            logger.error(f"Failed to get system resources: {e}")
            raise
    
    def _apply_initial_optimizations(self):
        """Apply initial system optimizations"""
        try:
            # PyTorch optimizations
            self._optimize_pytorch()
            
            # CPU optimizations
            if self.config.enable_cpu_optimization:
                self._optimize_cpu()
            
            # Memory optimizations
            if self.config.enable_memory_optimization:
                self._optimize_memory()
            
            # GPU optimizations
            if self.config.enable_gpu_optimization and self.resources.gpu_available:
                self._optimize_gpu()
            
            logger.info("Initial optimizations applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply initial optimizations: {e}")
            raise
    
    def _optimize_pytorch(self):
        """Optimize PyTorch settings"""
        try:
            # Set number of threads based on CPU cores
            optimal_threads = min(self.resources.cpu_cores, 8)
            torch.set_num_threads(optimal_threads)
            
            # Enable optimizations
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Set memory management
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"PyTorch optimized with {optimal_threads} threads")
            
        except Exception as e:
            logger.error(f"Failed to optimize PyTorch: {e}")
    
    def _optimize_cpu(self):
        """Optimize CPU usage"""
        try:
            # Set process priority (platform specific)
            if hasattr(os, 'nice'):
                try:
                    os.nice(-5)  # Higher priority
                except PermissionError:
                    logger.warning("Cannot set process priority (requires root)")
            
            # Set CPU affinity if available
            try:
                process = psutil.Process()
                available_cpus = list(range(self.resources.cpu_cores))
                process.cpu_affinity(available_cpus)
                logger.info(f"CPU affinity set to {available_cpus}")
            except (AttributeError, psutil.AccessDenied):
                logger.warning("Cannot set CPU affinity")
            
            logger.info("CPU optimization applied")
            
        except Exception as e:
            logger.error(f"Failed to optimize CPU: {e}")
    
    def _optimize_memory(self):
        """Optimize memory usage"""
        try:
            # Set memory limits based on available memory
            available_memory_gb = self.resources.memory_available_gb
            
            # Calculate optimal batch size based on memory
            if available_memory_gb < 4:
                optimal_batch_size = 4
            elif available_memory_gb < 8:
                optimal_batch_size = 8
            elif available_memory_gb < 16:
                optimal_batch_size = 16
            else:
                optimal_batch_size = 32
            
            # Store optimal batch size
            self.optimal_batch_size = optimal_batch_size
            
            # Clear system caches if memory usage is high
            if self.resources.memory_usage_percent > 80:
                self._clear_system_caches()
            
            logger.info(f"Memory optimization applied, optimal batch size: {optimal_batch_size}")
            
        except Exception as e:
            logger.error(f"Failed to optimize memory: {e}")
    
    def _optimize_gpu(self):
        """Optimize GPU usage"""
        try:
            if not torch.cuda.is_available():
                return
            
            # Set GPU memory fraction
            gpu_memory_fraction = 0.8  # Use 80% of GPU memory
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
            
            # Enable mixed precision
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            logger.info(f"GPU optimization applied, memory fraction: {gpu_memory_fraction}")
            
        except Exception as e:
            logger.error(f"Failed to optimize GPU: {e}")
    
    def _clear_system_caches(self):
        """Clear system caches to free memory"""
        try:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("System caches cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear system caches: {e}")
    
    def get_optimal_batch_size(self, base_batch_size: int = 8) -> int:
        """Get optimal batch size based on system resources"""
        try:
            if hasattr(self, 'optimal_batch_size'):
                return int(self.optimal_batch_size * self.config.batch_size_multiplier)
            
            # Calculate based on available memory
            available_memory_gb = self.resources.memory_available_gb
            
            if available_memory_gb < 4:
                multiplier = 0.5
            elif available_memory_gb < 8:
                multiplier = 1.0
            elif available_memory_gb < 16:
                multiplier = 2.0
            else:
                multiplier = 4.0
            
            return int(base_batch_size * multiplier * self.config.batch_size_multiplier)
            
        except Exception as e:
            logger.error(f"Failed to calculate optimal batch size: {e}")
            return base_batch_size
    
    def get_optimal_num_workers(self) -> int:
        """Get optimal number of workers for data loading"""
        try:
            if self.config.num_workers is not None:
                return self.config.num_workers
            
            # Calculate based on CPU cores
            cpu_cores = self.resources.cpu_cores
            
            if cpu_cores <= 2:
                return 1
            elif cpu_cores <= 4:
                return 2
            elif cpu_cores <= 8:
                return 4
            else:
                return min(8, cpu_cores // 2)
            
        except Exception as e:
            logger.error(f"Failed to calculate optimal num workers: {e}")
            return 2
    
    def monitor_system(self, interval: float = 5.0):
        """Start system monitoring"""
        try:
            if self.monitoring_active:
                logger.warning("System monitoring already active")
                return
            
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval,),
                daemon=True
            )
            self.monitor_thread.start()
            
            logger.info(f"System monitoring started with {interval}s interval")
            
        except Exception as e:
            logger.error(f"Failed to start system monitoring: {e}")
    
    def _monitor_loop(self, interval: float):
        """System monitoring loop"""
        try:
            while self.monitoring_active:
                # Get current resources
                current_resources = self._get_system_resources()
                
                # Check for optimization opportunities
                self._check_optimization_opportunities(current_resources)
                
                # Store monitoring data
                self.optimization_history.append({
                    'timestamp': time.time(),
                    'resources': current_resources,
                    'optimizations_applied': []
                })
                
                # Keep only last 100 entries
                if len(self.optimization_history) > 100:
                    self.optimization_history = self.optimization_history[-100:]
                
                time.sleep(interval)
                
        except Exception as e:
            logger.error(f"System monitoring loop failed: {e}")
        finally:
            self.monitoring_active = False
    
    def _check_optimization_opportunities(self, resources: SystemResources):
        """Check for optimization opportunities"""
        try:
            optimizations = []
            
            # Check CPU usage
            if resources.cpu_usage > self.config.max_cpu_usage:
                optimizations.append("high_cpu_usage")
                self._reduce_cpu_load()
            
            # Check memory usage
            if resources.memory_usage_percent > self.config.max_memory_usage:
                optimizations.append("high_memory_usage")
                self._reduce_memory_usage()
            
            # Check disk usage
            if resources.disk_usage_percent > self.config.max_disk_usage:
                optimizations.append("high_disk_usage")
                self._cleanup_disk_space()
            
            # Check GPU memory
            if resources.gpu_available and resources.gpu_memory_used_gb > 0:
                gpu_memory_percent = (resources.gpu_memory_used_gb / resources.gpu_memory_total_gb) * 100
                if gpu_memory_percent > 90:
                    optimizations.append("high_gpu_memory_usage")
                    self._clear_gpu_cache()
            
            # Update optimization history
            if optimizations and self.optimization_history:
                self.optimization_history[-1]['optimizations_applied'] = optimizations
            
        except Exception as e:
            logger.error(f"Failed to check optimization opportunities: {e}")
    
    def _reduce_cpu_load(self):
        """Reduce CPU load"""
        try:
            # Reduce PyTorch threads
            current_threads = torch.get_num_threads()
            if current_threads > 2:
                torch.set_num_threads(max(2, current_threads - 1))
                logger.info(f"Reduced PyTorch threads to {torch.get_num_threads()}")
            
        except Exception as e:
            logger.error(f"Failed to reduce CPU load: {e}")
    
    def _reduce_memory_usage(self):
        """Reduce memory usage"""
        try:
            # Clear caches
            self._clear_system_caches()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Memory usage reduced")
            
        except Exception as e:
            logger.error(f"Failed to reduce memory usage: {e}")
    
    def _cleanup_disk_space(self):
        """Cleanup disk space"""
        try:
            # Clear temporary files
            import tempfile
            temp_dir = tempfile.gettempdir()
            
            # This is a simplified cleanup - in production, implement proper cleanup
            logger.info("Disk cleanup initiated")
            
        except Exception as e:
            logger.error(f"Failed to cleanup disk space: {e}")
    
    def _clear_gpu_cache(self):
        """Clear GPU cache"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear GPU cache: {e}")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        try:
            self.monitoring_active = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)
            
            logger.info("System monitoring stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop system monitoring: {e}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization report"""
        try:
            current_resources = self._get_system_resources()
            
            return {
                'timestamp': time.time(),
                'current_resources': {
                    'cpu_cores': current_resources.cpu_cores,
                    'cpu_usage': current_resources.cpu_usage,
                    'memory_usage_percent': current_resources.memory_usage_percent,
                    'disk_usage_percent': current_resources.disk_usage_percent,
                    'gpu_available': current_resources.gpu_available,
                    'gpu_memory_used_gb': current_resources.gpu_memory_used_gb
                },
                'optimization_config': {
                    'max_cpu_usage': self.config.max_cpu_usage,
                    'max_memory_usage': self.config.max_memory_usage,
                    'max_disk_usage': self.config.max_disk_usage,
                    'enable_gpu_optimization': self.config.enable_gpu_optimization
                },
                'optimal_settings': {
                    'batch_size': self.get_optimal_batch_size(),
                    'num_workers': self.get_optimal_num_workers(),
                    'pytorch_threads': torch.get_num_threads()
                },
                'monitoring_active': self.monitoring_active,
                'optimization_history_count': len(self.optimization_history)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate optimization report: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup optimizer resources"""
        try:
            self.stop_monitoring()
            self._clear_system_caches()
            logger.info("System optimizer cleanup completed")
            
        except Exception as e:
            logger.error(f"Failed to cleanup system optimizer: {e}")

# Global optimizer instance
system_optimizer = SystemOptimizer()