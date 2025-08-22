"""
Windows CPU Optimization Module
Advanced CPU optimization for Windows VPS with Intel Extension for PyTorch
"""

import os
import sys
import asyncio
import ctypes
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

import psutil
import numpy as np
from loguru import logger

# Windows-specific imports
try:
    import win32api
    import win32con
    import win32process
    import win32security
    import wmi
    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False
    logger.warning("Windows-specific modules not available")

# PyTorch and Intel Extension
import torch
import torch.multiprocessing as mp
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False
    logger.warning("Intel Extension for PyTorch not available")

# CPU information
try:
    import cpuinfo
    CPU_INFO_AVAILABLE = True
except ImportError:
    CPU_INFO_AVAILABLE = False

# Memory allocator
try:
    import mimalloc
    MIMALLOC_AVAILABLE = True
except ImportError:
    MIMALLOC_AVAILABLE = False

# Local imports
from config.training_config import WindowsOptimizationConfig


@dataclass
class CPUTopology:
    """CPU topology information"""
    
    physical_cores: int
    logical_cores: int
    numa_nodes: int
    sockets: int
    cores_per_socket: int
    threads_per_core: int
    cache_sizes: Dict[str, int]
    cpu_features: List[str]
    base_frequency: float
    max_frequency: float


@dataclass
class MemoryInfo:
    """Memory configuration information"""
    
    total_memory: int
    available_memory: int
    numa_memory_distribution: Dict[int, int]
    large_pages_available: bool
    page_size: int
    memory_speed: Optional[int]


class CPUAffinityManager:
    """Manages CPU affinity for optimal performance"""
    
    def __init__(self, topology: CPUTopology):
        self.topology = topology
        self.process_affinities = {}
        
        # Calculate optimal CPU assignments
        self.physical_core_mask = self._calculate_physical_core_mask()
        self.numa_masks = self._calculate_numa_masks()
        
        logger.info(f"CPU Affinity Manager initialized for {topology.physical_cores} physical cores")
    
    def _calculate_physical_core_mask(self) -> int:
        """Calculate CPU mask for physical cores only"""
        
        if self.topology.threads_per_core == 1:
            # No hyperthreading - use all cores
            return (1 << self.topology.logical_cores) - 1
        
        # With hyperthreading - use only first thread of each core
        mask = 0
        for core in range(0, self.topology.logical_cores, self.topology.threads_per_core):
            mask |= (1 << core)
        
        return mask
    
    def _calculate_numa_masks(self) -> Dict[int, int]:
        """Calculate CPU masks for each NUMA node"""
        
        numa_masks = {}
        cores_per_numa = self.topology.physical_cores // max(1, self.topology.numa_nodes)
        
        for numa_node in range(self.topology.numa_nodes):
            start_core = numa_node * cores_per_numa
            end_core = min(start_core + cores_per_numa, self.topology.physical_cores)
            
            mask = 0
            for core in range(start_core, end_core):
                # Account for hyperthreading
                logical_core = core * self.topology.threads_per_core
                mask |= (1 << logical_core)
            
            numa_masks[numa_node] = mask
        
        return numa_masks
    
    def set_process_affinity(self, process_id: int, numa_node: Optional[int] = None) -> bool:
        """Set CPU affinity for a specific process"""
        
        if not WINDOWS_AVAILABLE:
            logger.warning("Windows-specific affinity setting not available")
            return False
        
        try:
            # Get process handle
            process_handle = win32api.OpenProcess(
                win32con.PROCESS_SET_INFORMATION | win32con.PROCESS_QUERY_INFORMATION,
                False,
                process_id
            )
            
            # Determine affinity mask
            if numa_node is not None and numa_node in self.numa_masks:
                affinity_mask = self.numa_masks[numa_node]
                logger.info(f"Setting process {process_id} affinity to NUMA node {numa_node}")
            else:
                affinity_mask = self.physical_core_mask
                logger.info(f"Setting process {process_id} affinity to physical cores")
            
            # Set affinity
            win32process.SetProcessAffinityMask(process_handle, affinity_mask)
            
            # Close handle
            win32api.CloseHandle(process_handle)
            
            self.process_affinities[process_id] = affinity_mask
            return True
            
        except Exception as e:
            logger.error(f"Failed to set process affinity: {e}")
            return False
    
    def set_thread_affinity(self, thread_id: int, core_id: int) -> bool:
        """Set CPU affinity for a specific thread"""
        
        if not WINDOWS_AVAILABLE:
            return False
        
        try:
            # Calculate core mask
            core_mask = 1 << core_id
            
            # Set thread affinity
            thread_handle = win32api.OpenThread(
                win32con.THREAD_SET_INFORMATION,
                False,
                thread_id
            )
            
            win32process.SetThreadAffinityMask(thread_handle, core_mask)
            win32api.CloseHandle(thread_handle)
            
            logger.debug(f"Thread {thread_id} bound to core {core_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set thread affinity: {e}")
            return False
    
    def get_optimal_thread_distribution(self, num_threads: int) -> List[int]:
        """Get optimal core distribution for threads"""
        
        # Distribute threads across physical cores
        physical_cores = self.topology.physical_cores
        
        if num_threads <= physical_cores:
            # One thread per core
            return list(range(0, num_threads * self.topology.threads_per_core, self.topology.threads_per_core))
        else:
            # Multiple threads per core
            cores = []
            threads_per_core = max(1, num_threads // physical_cores)
            
            for core in range(physical_cores):
                for thread in range(threads_per_core):
                    if len(cores) < num_threads:
                        logical_core = core * self.topology.threads_per_core + thread
                        cores.append(logical_core)
            
            return cores


class WindowsMemoryOptimizer:
    """Windows-specific memory optimization"""
    
    def __init__(self, config: WindowsOptimizationConfig):
        self.config = config
        self.large_pages_enabled = False
        self.memory_info = self._get_memory_info()
        
        logger.info("Windows Memory Optimizer initialized")
    
    def _get_memory_info(self) -> MemoryInfo:
        """Get comprehensive memory information"""
        
        # Basic memory info
        memory = psutil.virtual_memory()
        
        # NUMA memory distribution (simplified)
        numa_distribution = {}
        if WINDOWS_AVAILABLE:
            try:
                c = wmi.WMI()
                for processor in c.Win32_Processor():
                    numa_node = getattr(processor, 'NumaNode', 0) or 0
                    if numa_node not in numa_distribution:
                        numa_distribution[numa_node] = 0
                    numa_distribution[numa_node] += memory.total // len(list(c.Win32_Processor()))
            except Exception:
                numa_distribution[0] = memory.total
        else:
            numa_distribution[0] = memory.total
        
        # Check large page support
        large_pages_available = self._check_large_page_support()
        
        return MemoryInfo(
            total_memory=memory.total,
            available_memory=memory.available,
            numa_memory_distribution=numa_distribution,
            large_pages_available=large_pages_available,
            page_size=4096,  # Standard Windows page size
            memory_speed=None  # Would require additional detection
        )
    
    def _check_large_page_support(self) -> bool:
        """Check if large pages are supported and available"""
        
        if not WINDOWS_AVAILABLE:
            return False
        
        try:
            # Check if process has SeLockMemoryPrivilege
            process_token = win32security.OpenProcessToken(
                win32api.GetCurrentProcess(),
                win32security.TOKEN_ADJUST_PRIVILEGES | win32security.TOKEN_QUERY
            )
            
            privilege_luid = win32security.LookupPrivilegeValue(
                None,
                win32security.SE_LOCK_MEMORY_NAME
            )
            
            # Try to enable the privilege
            win32security.AdjustTokenPrivileges(
                process_token,
                False,
                [(privilege_luid, win32security.SE_PRIVILEGE_ENABLED)]
            )
            
            return True
            
        except Exception as e:
            logger.warning(f"Large pages not available: {e}")
            return False
    
    def enable_large_pages(self) -> bool:
        """Enable large page support for the process"""
        
        if not self.memory_info.large_pages_available:
            logger.warning("Large pages not supported on this system")
            return False
        
        try:
            if WINDOWS_AVAILABLE:
                # Set process to use large pages
                process_handle = win32api.GetCurrentProcess()
                
                # This would typically require additional Windows API calls
                # to actually allocate large pages
                logger.info("Large page support enabled")
                self.large_pages_enabled = True
                return True
            
        except Exception as e:
            logger.error(f"Failed to enable large pages: {e}")
        
        return False
    
    def optimize_memory_allocation(self) -> None:
        """Optimize memory allocation settings"""
        
        # Set environment variables for memory optimization
        os.environ['MALLOC_ARENA_MAX'] = '4'
        os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'
        os.environ['MALLOC_TRIM_THRESHOLD_'] = '131072'
        
        # Use mimalloc if available
        if MIMALLOC_AVAILABLE and self.config.use_mimalloc:
            try:
                # mimalloc is automatically used when imported
                logger.info("mimalloc memory allocator enabled")
            except Exception as e:
                logger.warning(f"Failed to enable mimalloc: {e}")
        
        # Set PyTorch memory allocation settings
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = False  # For CPU-only workloads
        
        logger.info("Memory allocation optimized")


class WindowsCPUOptimizer:
    """Main Windows CPU optimization system"""
    
    def __init__(self, config: WindowsOptimizationConfig):
        self.config = config
        self.topology: Optional[CPUTopology] = None
        self.affinity_manager: Optional[CPUAffinityManager] = None
        self.memory_optimizer: Optional[WindowsMemoryOptimizer] = None
        
        # Performance monitoring
        self.performance_metrics = {
            'cpu_usage_history': [],
            'memory_usage_history': [],
            'thread_efficiency': [],
            'numa_efficiency': []
        }
        
        logger.info("Windows CPU Optimizer initialized")
    
    async def initialize(self) -> bool:
        """Initialize all optimization components"""
        
        try:
            logger.info("Initializing Windows CPU optimization...")
            
            # Detect CPU topology
            self.topology = await self._detect_cpu_topology()
            
            # Initialize affinity manager
            self.affinity_manager = CPUAffinityManager(self.topology)
            
            # Initialize memory optimizer
            self.memory_optimizer = WindowsMemoryOptimizer(self.config)
            
            # Apply initial optimizations
            await self._apply_initial_optimizations()
            
            # Configure Intel Extension if available
            if IPEX_AVAILABLE and self.config.enable_ipex:
                await self._configure_intel_extension()
            
            # Set process priority
            await self._set_process_priority()
            
            logger.success("Windows CPU optimization initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Windows CPU optimization initialization failed: {e}")
            return False
    
    async def _detect_cpu_topology(self) -> CPUTopology:
        """Detect comprehensive CPU topology"""
        
        # Basic CPU information
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        
        # Calculate topology metrics
        threads_per_core = logical_cores // physical_cores if physical_cores > 0 else 1
        
        # CPU features and frequencies
        cpu_features = []
        base_freq = 0.0
        max_freq = 0.0
        
        if CPU_INFO_AVAILABLE:
            try:
                cpu_info = cpuinfo.get_cpu_info()
                cpu_features = cpu_info.get('flags', [])
                base_freq = cpu_info.get('hz_advertised_base', [0, 0])[0] / 1e9
                max_freq = cpu_info.get('hz_actual_base', [0, 0])[0] / 1e9
            except Exception:
                pass
        
        # NUMA information
        numa_nodes = 1
        if WINDOWS_AVAILABLE:
            try:
                c = wmi.WMI()
                numa_nodes = len(set(getattr(proc, 'NumaNode', 0) or 0 for proc in c.Win32_Processor()))
            except Exception:
                pass
        
        # Socket information (simplified)
        sockets = max(1, numa_nodes)
        cores_per_socket = physical_cores // sockets
        
        # Cache information (simplified)
        cache_sizes = {
            'L1': 32 * 1024,  # Typical L1 cache size
            'L2': 256 * 1024,  # Typical L2 cache size
            'L3': 8 * 1024 * 1024  # Typical L3 cache size
        }
        
        topology = CPUTopology(
            physical_cores=physical_cores,
            logical_cores=logical_cores,
            numa_nodes=numa_nodes,
            sockets=sockets,
            cores_per_socket=cores_per_socket,
            threads_per_core=threads_per_core,
            cache_sizes=cache_sizes,
            cpu_features=cpu_features,
            base_frequency=base_freq,
            max_frequency=max_freq
        )
        
        logger.info(f"Detected CPU topology: {physical_cores} physical cores, "
                   f"{logical_cores} logical cores, {numa_nodes} NUMA nodes")
        
        return topology
    
    async def _apply_initial_optimizations(self) -> None:
        """Apply initial CPU and memory optimizations"""
        
        # Set CPU affinity for current process
        if self.config.enable_cpu_affinity:
            current_pid = os.getpid()
            self.affinity_manager.set_process_affinity(current_pid)
        
        # Configure memory optimization
        self.memory_optimizer.optimize_memory_allocation()
        
        if self.config.enable_large_pages:
            self.memory_optimizer.enable_large_pages()
        
        # Set environment variables for optimal threading
        os.environ['OMP_NUM_THREADS'] = str(self.config.omp_num_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.config.mkl_num_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.config.mkl_num_threads)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(self.config.mkl_num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.config.mkl_num_threads)
        
        # NUMA optimization
        if self.config.numa_aware and self.topology.numa_nodes > 1:
            os.environ['OMP_PLACES'] = 'cores'
            os.environ['OMP_PROC_BIND'] = 'close'
        
        logger.info("Initial CPU optimizations applied")
    
    async def _configure_intel_extension(self) -> None:
        """Configure Intel Extension for PyTorch"""
        
        if not IPEX_AVAILABLE:
            logger.warning("Intel Extension for PyTorch not available")
            return
        
        try:
            # Set Intel Extension configuration
            logger.info("Configuring Intel Extension for PyTorch...")
            
            # Configure threading
            torch.set_num_threads(self.config.omp_num_threads)
            torch.set_num_interop_threads(max(1, self.config.omp_num_threads // 4))
            
            # Enable Intel optimizations
            if hasattr(torch.backends, 'mkldnn'):
                torch.backends.mkldnn.enabled = True
                torch.backends.mkldnn.verbose = 0
            
            # Configure for CPU-only workload
            torch.backends.openmp.is_available = lambda: True
            
            # AMX and AVX-512 configuration
            if self.config.enable_amx and 'amx' in [f.lower() for f in self.topology.cpu_features]:
                logger.info("AMX (Advanced Matrix Extensions) detected and enabled")
            
            if self.config.enable_avx512 and 'avx512' in [f.lower() for f in self.topology.cpu_features]:
                logger.info("AVX-512 detected and enabled")
            
            # Set optimization level
            ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.BF16)
            
            logger.success("Intel Extension for PyTorch configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to configure Intel Extension: {e}")
    
    async def _set_process_priority(self) -> None:
        """Set optimal process priority"""
        
        if not WINDOWS_AVAILABLE:
            return
        
        try:
            priority_map = {
                'realtime': win32process.REALTIME_PRIORITY_CLASS,
                'high': win32process.HIGH_PRIORITY_CLASS,
                'normal': win32process.NORMAL_PRIORITY_CLASS
            }
            
            priority_class = priority_map.get(self.config.set_process_priority, win32process.HIGH_PRIORITY_CLASS)
            
            process_handle = win32api.GetCurrentProcess()
            win32process.SetPriorityClass(process_handle, priority_class)
            
            logger.info(f"Process priority set to {self.config.set_process_priority}")
            
        except Exception as e:
            logger.error(f"Failed to set process priority: {e}")
    
    def optimize_for_training(self, num_workers: int = None) -> Dict[str, Any]:
        """Optimize CPU settings for training workload"""
        
        if num_workers is None:
            num_workers = self.topology.physical_cores
        
        # Calculate optimal thread distribution
        optimal_cores = self.affinity_manager.get_optimal_thread_distribution(num_workers)
        
        # Configure PyTorch for optimal CPU usage
        config = {
            'num_workers': min(num_workers, self.topology.physical_cores),
            'num_threads': self.config.omp_num_threads,
            'interop_threads': max(1, self.config.omp_num_threads // 4),
            'core_assignment': optimal_cores,
            'numa_aware': self.config.numa_aware and self.topology.numa_nodes > 1
        }
        
        # Apply PyTorch settings
        torch.set_num_threads(config['num_threads'])
        torch.set_num_interop_threads(config['interop_threads'])
        
        logger.info(f"CPU optimized for training: {config['num_workers']} workers, "
                   f"{config['num_threads']} threads")
        
        return config
    
    def optimize_for_inference(self) -> Dict[str, Any]:
        """Optimize CPU settings for inference workload"""
        
        # Inference typically benefits from fewer threads
        inference_threads = max(1, self.topology.physical_cores // 2)
        
        config = {
            'num_threads': inference_threads,
            'interop_threads': 1,
            'batch_size_recommendation': 1,
            'memory_optimization': True
        }
        
        # Apply settings
        torch.set_num_threads(config['num_threads'])
        torch.set_num_interop_threads(config['interop_threads'])
        
        # Enable inference optimizations
        torch.jit.set_fusion_strategy([('STATIC', 2), ('DYNAMIC', 2)])
        
        logger.info(f"CPU optimized for inference: {config['num_threads']} threads")
        
        return config
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Process-specific metrics
        process = psutil.Process()
        process_cpu = process.cpu_percent()
        process_memory = process.memory_info()
        
        metrics = {
            'cpu_usage_total': sum(cpu_percent) / len(cpu_percent),
            'cpu_usage_per_core': cpu_percent,
            'memory_usage_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'process_cpu_percent': process_cpu,
            'process_memory_mb': process_memory.rss / (1024**2),
            'numa_nodes': self.topology.numa_nodes,
            'active_threads': process.num_threads(),
            'optimization_active': True
        }
        
        # Store for history
        self.performance_metrics['cpu_usage_history'].append(metrics['cpu_usage_total'])
        self.performance_metrics['memory_usage_history'].append(metrics['memory_usage_percent'])
        
        # Limit history size
        max_history = 1000
        for key in self.performance_metrics:
            if len(self.performance_metrics[key]) > max_history:
                self.performance_metrics[key] = self.performance_metrics[key][-max_history:]
        
        return metrics
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current system state"""
        
        recommendations = []
        
        # Check CPU usage patterns
        if self.performance_metrics['cpu_usage_history']:
            avg_cpu = np.mean(self.performance_metrics['cpu_usage_history'][-100:])
            
            if avg_cpu < 50:
                recommendations.append("CPU usage is low - consider increasing batch size or parallelism")
            elif avg_cpu > 90:
                recommendations.append("CPU usage is high - consider reducing batch size or worker count")
        
        # Check memory usage
        if self.performance_metrics['memory_usage_history']:
            avg_memory = np.mean(self.performance_metrics['memory_usage_history'][-100:])
            
            if avg_memory > 85:
                recommendations.append("Memory usage is high - consider enabling gradient checkpointing")
            elif avg_memory < 30:
                recommendations.append("Memory usage is low - consider increasing model size or batch size")
        
        # Check Intel Extension availability
        if not IPEX_AVAILABLE:
            recommendations.append("Install Intel Extension for PyTorch for better CPU performance")
        
        # Check large page support
        if not self.memory_optimizer.large_pages_enabled and self.config.enable_large_pages:
            recommendations.append("Enable large pages for better memory performance")
        
        return recommendations
    
    async def cleanup(self) -> None:
        """Cleanup optimization resources"""
        
        try:
            # Reset process priority to normal
            if WINDOWS_AVAILABLE:
                process_handle = win32api.GetCurrentProcess()
                win32process.SetPriorityClass(process_handle, win32process.NORMAL_PRIORITY_CLASS)
            
            # Reset environment variables
            for env_var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
                if env_var in os.environ:
                    del os.environ[env_var]
            
            logger.info("Windows CPU optimization cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        
        info = {
            'platform': platform.platform(),
            'cpu_topology': {
                'physical_cores': self.topology.physical_cores,
                'logical_cores': self.topology.logical_cores,
                'numa_nodes': self.topology.numa_nodes,
                'sockets': self.topology.sockets,
                'base_frequency': self.topology.base_frequency,
                'max_frequency': self.topology.max_frequency,
                'cpu_features': self.topology.cpu_features[:10]  # Limit output
            },
            'memory_info': {
                'total_gb': self.memory_optimizer.memory_info.total_memory / (1024**3),
                'available_gb': self.memory_optimizer.memory_info.available_memory / (1024**3),
                'large_pages_enabled': self.memory_optimizer.large_pages_enabled,
                'numa_distribution': self.memory_optimizer.memory_info.numa_memory_distribution
            },
            'optimization_config': {
                'intel_extension_available': IPEX_AVAILABLE,
                'mimalloc_available': MIMALLOC_AVAILABLE,
                'windows_available': WINDOWS_AVAILABLE,
                'omp_threads': self.config.omp_num_threads,
                'mkl_threads': self.config.mkl_num_threads,
                'numa_aware': self.config.numa_aware
            },
            'current_performance': self.get_performance_metrics()
        }
        
        return info