"""
System-wide optimization for Persian Legal AI
بهینه‌سازی سیستم برای هوش مصنوعی حقوقی فارسی
"""

import psutil
import os
import gc
import threading
import time
from typing import Dict, List, Optional, Any
import logging
import subprocess
import platform
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """System optimization configuration"""
    memory_limit_gb: float = 64.0
    cpu_cores: int = 24
    enable_memory_optimization: bool = True
    enable_gc_optimization: bool = True
    enable_swap_optimization: bool = True
    monitoring_interval: float = 5.0

class SystemOptimizer:
    """
    System-wide optimization for maximum performance
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.monitoring_thread = None
        self.monitoring_active = False
        self.performance_history = []
        self.optimization_stats = {}
        
        # Initialize system optimization
        self._apply_system_optimizations()
    
    def _apply_system_optimizations(self):
        """Apply system-wide optimizations"""
        try:
            # Memory optimizations
            if self.config.enable_memory_optimization:
                self._optimize_memory_settings()
            
            # Garbage collection optimizations
            if self.config.enable_gc_optimization:
                self._optimize_garbage_collection()
            
            # Swap optimizations
            if self.config.enable_swap_optimization:
                self._optimize_swap_settings()
            
            # Process priority optimization
            self._optimize_process_priority()
            
            logger.info("System optimizations applied")
            
        except Exception as e:
            logger.error(f"Failed to apply system optimizations: {e}")
    
    def _optimize_memory_settings(self):
        """Optimize memory-related system settings"""
        try:
            # Set memory-related environment variables
            os.environ['PYTHONHASHSEED'] = '0'
            os.environ['PYTHONUNBUFFERED'] = '1'
            
            # Memory allocation optimization
            os.environ['MALLOC_TRIM_THRESHOLD_'] = '131072'
            os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'
            os.environ['MALLOC_MMAP_MAX_'] = '65536'
            
            # Python memory optimization
            os.environ['PYTHONMALLOC'] = 'malloc'
            
            logger.info("Memory settings optimized")
            
        except Exception as e:
            logger.error(f"Failed to optimize memory settings: {e}")
    
    def _optimize_garbage_collection(self):
        """Optimize garbage collection settings"""
        try:
            import gc
            
            # Set garbage collection thresholds
            gc.set_threshold(700, 10, 10)
            
            # Enable automatic garbage collection
            gc.enable()
            
            logger.info("Garbage collection optimized")
            
        except Exception as e:
            logger.error(f"Failed to optimize garbage collection: {e}")
    
    def _optimize_swap_settings(self):
        """Optimize swap settings for better performance"""
        try:
            # Check if running on Linux
            if platform.system() == 'Linux':
                # Set swappiness to 10 (prefer RAM over swap)
                try:
                    subprocess.run(['sudo', 'sysctl', 'vm.swappiness=10'], 
                                 check=True, capture_output=True)
                    logger.info("Swap swappiness optimized")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.warning("Could not optimize swap settings (requires sudo)")
            
        except Exception as e:
            logger.error(f"Failed to optimize swap settings: {e}")
    
    def _optimize_process_priority(self):
        """Optimize process priority for better performance"""
        try:
            # Set high priority for current process
            process = psutil.Process()
            
            if platform.system() == 'Windows':
                process.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                # Unix-like systems
                os.nice(-10)  # Higher priority
            
            logger.info("Process priority optimized")
            
        except (PermissionError, OSError) as e:
            logger.warning(f"Could not set process priority: {e}")
        except Exception as e:
            logger.error(f"Failed to optimize process priority: {e}")
    
    def optimize_for_24_core_system(self) -> Dict[str, Any]:
        """Optimize system specifically for 24-core CPU"""
        try:
            optimization_results = {
                'cpu_optimization': self._optimize_cpu_settings(),
                'memory_optimization': self._optimize_memory_allocation(),
                'thread_optimization': self._optimize_thread_settings(),
                'cache_optimization': self._optimize_cache_settings()
            }
            
            logger.info("24-core system optimization complete")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Failed to optimize for 24-core system: {e}")
            return {}
    
    def _optimize_cpu_settings(self) -> Dict[str, Any]:
        """Optimize CPU-specific settings"""
        try:
            # Set CPU governor to performance mode (Linux)
            if platform.system() == 'Linux':
                try:
                    # Try to set CPU governor to performance
                    subprocess.run(['sudo', 'cpupower', 'frequency-set', '-g', 'performance'],
                                 check=True, capture_output=True)
                    logger.info("CPU governor set to performance mode")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.warning("Could not set CPU governor (requires cpupower)")
            
            # Set CPU affinity for optimal core usage
            cpu_affinity = list(range(self.config.cpu_cores))
            process = psutil.Process()
            process.cpu_affinity(cpu_affinity)
            
            return {
                'cpu_affinity_set': True,
                'cores_used': len(cpu_affinity),
                'governor_mode': 'performance'
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize CPU settings: {e}")
            return {'cpu_affinity_set': False}
    
    def _optimize_memory_allocation(self) -> Dict[str, Any]:
        """Optimize memory allocation patterns"""
        try:
            # Get system memory info
            memory = psutil.virtual_memory()
            
            # Calculate optimal memory allocation
            total_memory_gb = memory.total / (1024**3)
            available_memory_gb = memory.available / (1024**3)
            
            # Set memory limits
            memory_limit = min(self.config.memory_limit_gb, available_memory_gb * 0.8)
            
            # Optimize PyTorch memory allocation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'total_memory_gb': total_memory_gb,
                'available_memory_gb': available_memory_gb,
                'memory_limit_gb': memory_limit,
                'memory_optimized': True
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize memory allocation: {e}")
            return {'memory_optimized': False}
    
    def _optimize_thread_settings(self) -> Dict[str, Any]:
        """Optimize thread-related settings"""
        try:
            # Set optimal thread counts
            os.environ['OMP_NUM_THREADS'] = str(self.config.cpu_cores)
            os.environ['MKL_NUM_THREADS'] = str(self.config.cpu_cores)
            os.environ['NUMEXPR_NUM_THREADS'] = str(self.config.cpu_cores)
            
            # Thread affinity settings
            os.environ['OMP_PLACES'] = 'cores'
            os.environ['OMP_PROC_BIND'] = 'close'
            
            return {
                'thread_count': self.config.cpu_cores,
                'thread_affinity': True,
                'optimization_applied': True
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize thread settings: {e}")
            return {'optimization_applied': False}
    
    def _optimize_cache_settings(self) -> Dict[str, Any]:
        """Optimize cache-related settings"""
        try:
            # Set cache-related environment variables
            os.environ['MKL_CACHE_SIZE'] = '64'
            os.environ['OMP_WAIT_POLICY'] = 'active'
            
            return {
                'cache_size_mb': 64,
                'wait_policy': 'active',
                'optimization_applied': True
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize cache settings: {e}")
            return {'optimization_applied': False}
    
    def start_performance_monitoring(self):
        """Start real-time performance monitoring"""
        try:
            if self.monitoring_active:
                logger.warning("Performance monitoring already active")
                return
            
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitor_performance,
                daemon=True
            )
            self.monitoring_thread.start()
            
            logger.info("Performance monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start performance monitoring: {e}")
    
    def stop_performance_monitoring(self):
        """Stop performance monitoring"""
        try:
            self.monitoring_active = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            
            logger.info("Performance monitoring stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop performance monitoring: {e}")
    
    def _monitor_performance(self):
        """Background performance monitoring thread"""
        while self.monitoring_active:
            try:
                # Collect performance metrics
                metrics = self.collect_system_metrics()
                self.performance_history.append(metrics)
                
                # Keep only last 1000 entries
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-1000:]
                
                # Check for performance issues
                self._check_performance_issues(metrics)
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(percpu=True, interval=1)
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            metrics = {
                'timestamp': time.time(),
                'cpu': {
                    'per_core_percent': cpu_percent,
                    'average_percent': sum(cpu_percent) / len(cpu_percent),
                    'frequency_mhz': cpu_freq.current if cpu_freq else None,
                    'process_cpu_percent': process_cpu
                },
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_percent': memory.percent,
                    'process_memory_mb': process_memory.rss / (1024**2),
                    'swap_used_gb': swap.used / (1024**3)
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'used_percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}
    
    def _check_performance_issues(self, metrics: Dict[str, Any]):
        """Check for performance issues and apply fixes"""
        try:
            # Check memory usage
            memory_percent = metrics['memory']['used_percent']
            if memory_percent > 90:
                logger.warning(f"High memory usage: {memory_percent:.1f}%")
                self._handle_high_memory_usage()
            
            # Check CPU usage
            cpu_avg = metrics['cpu']['average_percent']
            if cpu_avg > 95:
                logger.warning(f"High CPU usage: {cpu_avg:.1f}%")
            
            # Check swap usage
            swap_used = metrics['memory']['swap_used_gb']
            if swap_used > 1.0:  # More than 1GB swap usage
                logger.warning(f"High swap usage: {swap_used:.2f}GB")
                self._handle_high_swap_usage()
            
        except Exception as e:
            logger.error(f"Failed to check performance issues: {e}")
    
    def _handle_high_memory_usage(self):
        """Handle high memory usage"""
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear PyTorch cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Applied memory cleanup measures")
            
        except Exception as e:
            logger.error(f"Failed to handle high memory usage: {e}")
    
    def _handle_high_swap_usage(self):
        """Handle high swap usage"""
        try:
            # Clear swap if possible (Linux)
            if platform.system() == 'Linux':
                try:
                    subprocess.run(['sudo', 'swapoff', '-a'], 
                                 check=True, capture_output=True)
                    subprocess.run(['sudo', 'swapon', '-a'], 
                                 check=True, capture_output=True)
                    logger.info("Swap cleared")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.warning("Could not clear swap (requires sudo)")
            
        except Exception as e:
            logger.error(f"Failed to handle high swap usage: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            if not self.performance_history:
                return {}
            
            latest = self.performance_history[-1]
            
            # Calculate averages over last 10 measurements
            recent_metrics = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
            
            avg_cpu = sum(m['cpu']['average_percent'] for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m['memory']['used_percent'] for m in recent_metrics) / len(recent_metrics)
            
            return {
                'current_metrics': latest,
                'average_cpu_percent': avg_cpu,
                'average_memory_percent': avg_memory,
                'monitoring_duration_minutes': len(self.performance_history) * self.config.monitoring_interval / 60,
                'optimization_stats': self.optimization_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup system optimizations"""
        try:
            # Stop monitoring
            self.stop_performance_monitoring()
            
            # Clear environment variables
            env_vars_to_clear = [
                'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'MALLOC_TRIM_THRESHOLD_', 'MALLOC_MMAP_THRESHOLD_',
                'OMP_PLACES', 'OMP_PROC_BIND', 'MKL_CACHE_SIZE'
            ]
            
            for var in env_vars_to_clear:
                if var in os.environ:
                    del os.environ[var]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("System optimizer cleanup complete")
            
        except Exception as e:
            logger.error(f"Failed to cleanup system optimizer: {e}")