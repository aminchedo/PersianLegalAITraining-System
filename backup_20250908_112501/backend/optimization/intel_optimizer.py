"""
Intel CPU Optimization for 24-Core Systems
بهینه‌سازی پردازنده Intel برای سیستم‌های 24 هسته‌ای
"""

import torch
import psutil
import os
import threading
import multiprocessing as mp
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np
from dataclasses import dataclass
import subprocess
import platform

logger = logging.getLogger(__name__)

@dataclass
class CPUConfig:
    """CPU optimization configuration"""
    physical_cores: int = 24
    logical_cores: int = 48
    enable_intel_extension: bool = True
    use_mixed_precision: bool = True
    numa_aware: bool = True
    thread_affinity: bool = True
    amx_enabled: bool = True
    avx512_enabled: bool = True

class IntelCPUOptimizer:
    """
    Maximum CPU utilization for model training on Intel systems
    """
    
    def __init__(self, config: Optional[CPUConfig] = None):
        self.config = config or CPUConfig()
        self.original_thread_count = None
        self.numa_topology = None
        self.cpu_info = {}
        self.performance_counters = {}
        
        # Initialize system detection
        self._detect_system_capabilities()
        self._setup_intel_optimizations()
    
    def _detect_system_capabilities(self):
        """Detect system capabilities and CPU features"""
        try:
            # Get CPU information
            self.cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'cpu_freq': psutil.cpu_freq(),
                'cpu_percent': psutil.cpu_percent(percpu=True),
                'architecture': platform.machine(),
                'processor': platform.processor()
            }
            
            # Detect Intel features
            self._detect_intel_features()
            
            # Detect NUMA topology
            self._detect_numa_topology()
            
            logger.info(f"System detected: {self.cpu_info['physical_cores']} physical cores, "
                       f"{self.cpu_info['logical_cores']} logical cores")
            
        except Exception as e:
            logger.error(f"Failed to detect system capabilities: {e}")
    
    def _detect_intel_features(self):
        """Detect Intel-specific CPU features"""
        try:
            # Check for Intel Extension for PyTorch
            try:
                import intel_extension_for_pytorch as ipex
                self.cpu_info['intel_extension_available'] = True
                logger.info("Intel Extension for PyTorch detected")
            except ImportError:
                self.cpu_info['intel_extension_available'] = False
                logger.warning("Intel Extension for PyTorch not available")
            
            # Check CPU flags for AMX, AVX-512
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    
                self.cpu_info['amx_available'] = 'amx' in cpuinfo.lower()
                self.cpu_info['avx512_available'] = 'avx512' in cpuinfo.lower()
                self.cpu_info['avx2_available'] = 'avx2' in cpuinfo.lower()
                
                logger.info(f"CPU features - AMX: {self.cpu_info['amx_available']}, "
                           f"AVX-512: {self.cpu_info['avx512_available']}, "
                           f"AVX2: {self.cpu_info['avx2_available']}")
                           
            except Exception as e:
                logger.warning(f"Could not detect CPU features: {e}")
                self.cpu_info.update({
                    'amx_available': False,
                    'avx512_available': False,
                    'avx2_available': True
                })
                
        except Exception as e:
            logger.error(f"Failed to detect Intel features: {e}")
    
    def _detect_numa_topology(self):
        """Detect NUMA topology for memory optimization"""
        try:
            # Check if NUMA is available
            numa_available = os.path.exists('/sys/devices/system/node')
            
            if numa_available:
                # Get NUMA node information
                nodes = []
                for node_dir in os.listdir('/sys/devices/system/node'):
                    if node_dir.startswith('node'):
                        node_id = int(node_dir[4:])
                        cpu_list_file = f'/sys/devices/system/node/{node_dir}/cpulist'
                        
                        if os.path.exists(cpu_list_file):
                            with open(cpu_list_file, 'r') as f:
                                cpu_list = f.read().strip()
                            nodes.append({'id': node_id, 'cpus': cpu_list})
                
                self.numa_topology = {
                    'available': True,
                    'nodes': nodes,
                    'num_nodes': len(nodes)
                }
                
                logger.info(f"NUMA topology detected: {len(nodes)} nodes")
            else:
                self.numa_topology = {'available': False, 'nodes': [], 'num_nodes': 1}
                logger.info("NUMA not available, using single node")
                
        except Exception as e:
            logger.error(f"Failed to detect NUMA topology: {e}")
            self.numa_topology = {'available': False, 'nodes': [], 'num_nodes': 1}
    
    def _setup_intel_optimizations(self):
        """Setup Intel-specific optimizations"""
        try:
            # Set environment variables for Intel optimizations
            os.environ['OMP_NUM_THREADS'] = str(self.cpu_info['physical_cores'])
            os.environ['MKL_NUM_THREADS'] = str(self.cpu_info['physical_cores'])
            os.environ['NUMEXPR_NUM_THREADS'] = str(self.cpu_info['physical_cores'])
            
            # Intel MKL optimizations
            os.environ['MKL_DYNAMIC'] = 'FALSE'
            os.environ['MKL_INTERFACE_LAYER'] = 'LP64,GNU'
            
            # Memory optimization
            os.environ['MALLOC_TRIM_THRESHOLD_'] = '131072'
            os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'
            
            # Thread affinity
            if self.config.thread_affinity:
                self._setup_thread_affinity()
            
            logger.info("Intel optimizations configured")
            
        except Exception as e:
            logger.error(f"Failed to setup Intel optimizations: {e}")
    
    def _setup_thread_affinity(self):
        """Setup thread affinity for optimal CPU utilization"""
        try:
            if self.numa_topology['available'] and len(self.numa_topology['nodes']) > 1:
                # NUMA-aware thread affinity
                for i, node in enumerate(self.numa_topology['nodes']):
                    cpu_list = node['cpus']
                    # Set thread affinity for each NUMA node
                    os.environ[f'OMP_PLACES'] = f'cores'
                    os.environ[f'OMP_PROC_BIND'] = f'close'
            else:
                # Single node thread affinity
                os.environ['OMP_PLACES'] = 'cores'
                os.environ['OMP_PROC_BIND'] = 'close'
            
            logger.info("Thread affinity configured")
            
        except Exception as e:
            logger.error(f"Failed to setup thread affinity: {e}")
    
    def optimize_model_for_cpu(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply Intel optimizations to model"""
        try:
            # Enable Intel optimizations
            if self.cpu_info.get('intel_extension_available', False):
                import intel_extension_for_pytorch as ipex
                
                # Apply Intel optimizations
                model = ipex.optimize(
                    model,
                    level="O1",  # Aggressive optimization
                    auto_kernel_selection=True,
                    sample_input=None
                )
                
                logger.info("Applied Intel Extension optimizations to model")
            
            # Set optimal thread count
            self._set_optimal_thread_count()
            
            # Enable CPU-specific optimizations
            torch.set_num_threads(self.cpu_info['physical_cores'])
            torch.set_num_interop_threads(min(4, self.cpu_info['physical_cores'] // 6))
            
            # Enable MKL-DNN optimizations
            if torch.backends.mkldnn.is_available():
                torch.backends.mkldnn.enabled = True
                logger.info("Enabled MKL-DNN optimizations")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to optimize model for CPU: {e}")
            return model
    
    def _set_optimal_thread_count(self):
        """Set optimal thread count for different operations"""
        try:
            # PyTorch thread settings
            torch.set_num_threads(self.cpu_info['physical_cores'])
            torch.set_num_interop_threads(min(4, self.cpu_info['physical_cores'] // 6))
            
            # NumPy thread settings
            import numpy as np
            np.seterr(all='ignore')  # Suppress numerical warnings
            
            # Set BLAS thread count
            os.environ['OPENBLAS_NUM_THREADS'] = str(self.cpu_info['physical_cores'])
            os.environ['MKL_NUM_THREADS'] = str(self.cpu_info['physical_cores'])
            
            logger.info(f"Set thread count: {self.cpu_info['physical_cores']} threads")
            
        except Exception as e:
            logger.error(f"Failed to set optimal thread count: {e}")
    
    def enable_mixed_precision_cpu(self) -> Dict[str, Any]:
        """Enable mixed precision training optimized for CPU"""
        try:
            # CPU-optimized mixed precision settings
            mixed_precision_config = {
                'enabled': True,
                'dtype': torch.float16,  # Use FP16 for CPU
                'autocast_enabled': True,
                'grad_scaler': None  # No gradient scaling for CPU
            }
            
            # Enable autocast for CPU
            torch.backends.cpu.amp.autocast_enabled = True
            
            logger.info("Enabled CPU-optimized mixed precision")
            return mixed_precision_config
            
        except Exception as e:
            logger.error(f"Failed to enable mixed precision: {e}")
            return {'enabled': False}
    
    def setup_numa_awareness(self) -> Dict[str, Any]:
        """Setup NUMA-aware memory allocation"""
        try:
            if not self.numa_topology['available']:
                return {'enabled': False, 'reason': 'NUMA not available'}
            
            numa_config = {
                'enabled': True,
                'num_nodes': self.numa_topology['num_nodes'],
                'node_configs': []
            }
            
            # Configure each NUMA node
            for node in self.numa_topology['nodes']:
                node_config = {
                    'node_id': node['id'],
                    'cpu_list': node['cpus'],
                    'memory_policy': 'local'  # Prefer local memory
                }
                numa_config['node_configs'].append(node_config)
            
            # Set NUMA memory policy
            os.environ['OMP_PLACES'] = 'cores'
            os.environ['OMP_PROC_BIND'] = 'close'
            
            logger.info(f"NUMA awareness configured for {self.numa_topology['num_nodes']} nodes")
            return numa_config
            
        except Exception as e:
            logger.error(f"Failed to setup NUMA awareness: {e}")
            return {'enabled': False, 'error': str(e)}
    
    def monitor_resource_utilization(self) -> Dict[str, Any]:
        """Monitor real-time resource utilization"""
        try:
            # CPU utilization per core
            cpu_percent = psutil.cpu_percent(percpu=True, interval=1)
            
            # Memory information
            memory = psutil.virtual_memory()
            
            # CPU frequency
            cpu_freq = psutil.cpu_freq()
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            utilization = {
                'timestamp': psutil.time.time(),
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
                    'process_memory_mb': process_memory.rss / (1024**2)
                },
                'system': {
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None,
                    'num_processes': len(psutil.pids())
                }
            }
            
            return utilization
            
        except Exception as e:
            logger.error(f"Failed to monitor resource utilization: {e}")
            return {}
    
    def optimize_data_loading(self, num_workers: Optional[int] = None) -> Dict[str, Any]:
        """Optimize data loading for CPU training"""
        try:
            if num_workers is None:
                # Optimal number of workers for CPU
                num_workers = min(self.cpu_info['physical_cores'], 8)
            
            data_loading_config = {
                'num_workers': num_workers,
                'pin_memory': False,  # Not needed for CPU
                'persistent_workers': True,
                'prefetch_factor': 2,
                'multiprocessing_context': 'spawn'
            }
            
            logger.info(f"Data loading optimized: {num_workers} workers")
            return data_loading_config
            
        except Exception as e:
            logger.error(f"Failed to optimize data loading: {e}")
            return {'num_workers': 4}
    
    def setup_distributed_training(self, num_processes: Optional[int] = None) -> Dict[str, Any]:
        """Setup distributed training for CPU"""
        try:
            if num_processes is None:
                num_processes = min(self.cpu_info['physical_cores'] // 2, 8)
            
            distributed_config = {
                'backend': 'gloo',  # CPU backend
                'num_processes': num_processes,
                'rank': 0,
                'world_size': num_processes,
                'init_method': 'tcp://localhost:12355'
            }
            
            # Set distributed environment variables
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['WORLD_SIZE'] = str(num_processes)
            os.environ['RANK'] = '0'
            
            logger.info(f"Distributed training configured: {num_processes} processes")
            return distributed_config
            
        except Exception as e:
            logger.error(f"Failed to setup distributed training: {e}")
            return {}
    
    def benchmark_performance(self, model: torch.nn.Module, sample_input: torch.Tensor, 
                            num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance on CPU"""
        try:
            model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(sample_input)
            
            # Benchmark
            import time
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(sample_input)
            
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_iteration = total_time / num_iterations
            throughput = num_iterations / total_time
            
            benchmark_results = {
                'total_time_seconds': total_time,
                'avg_time_per_iteration_ms': avg_time_per_iteration * 1000,
                'throughput_iterations_per_second': throughput,
                'num_iterations': num_iterations
            }
            
            logger.info(f"Benchmark completed: {throughput:.2f} iterations/sec")
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Failed to benchmark performance: {e}")
            return {}
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of applied optimizations"""
        return {
            'cpu_info': self.cpu_info,
            'numa_topology': self.numa_topology,
            'intel_features': {
                'extension_available': self.cpu_info.get('intel_extension_available', False),
                'amx_available': self.cpu_info.get('amx_available', False),
                'avx512_available': self.cpu_info.get('avx512_available', False)
            },
            'optimizations_applied': {
                'thread_count': self.cpu_info['physical_cores'],
                'numa_aware': self.numa_topology['available'],
                'mixed_precision': self.config.use_mixed_precision,
                'intel_extension': self.cpu_info.get('intel_extension_available', False)
            }
        }
    
    def cleanup(self):
        """Cleanup optimizations and restore original settings"""
        try:
            # Restore original thread count if saved
            if self.original_thread_count:
                torch.set_num_threads(self.original_thread_count)
            
            # Clear environment variables
            env_vars_to_clear = [
                'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'MKL_DYNAMIC', 'MKL_INTERFACE_LAYER', 'OPENBLAS_NUM_THREADS'
            ]
            
            for var in env_vars_to_clear:
                if var in os.environ:
                    del os.environ[var]
            
            logger.info("Intel optimizer cleanup complete")
            
        except Exception as e:
            logger.error(f"Failed to cleanup optimizations: {e}")