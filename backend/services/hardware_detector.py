"""
Dynamic hardware detection and model selection for Persian Legal AI
هشدار سخت‌افزار پویا و انتخاب مدل برای هوش مصنوعی حقوقی فارسی

Integrates with existing backend/services/ structure
"""
import torch
import psutil
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import os
import platform

# Try to import GPU utilities
try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class HardwareDetector:
    """
    Hardware detection and optimization service for Persian Legal AI
    Integrates with existing system architecture
    """
    
    def __init__(self):
        """Initialize hardware detector with comprehensive system analysis"""
        self.hardware_info = self._detect_hardware()
        self.optimization_config = self._generate_optimization_config()
        logger.info(f"Hardware detector initialized: {self.get_hardware_summary()}")
        
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect available hardware resources with detailed analysis"""
        try:
            # Basic system information
            info = {
                'cpu_cores': psutil.cpu_count(logical=False),
                'cpu_logical_cores': psutil.cpu_count(logical=True),
                'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'available_ram_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': 0,
                'gpu_memory_gb': 0,
                'gpu_devices': [],
                'platform': 'cpu',
                'os_type': platform.system(),
                'architecture': platform.machine(),
                'python_version': platform.python_version()
            }
            
            # GPU Detection with PyTorch
            if torch.cuda.is_available():
                try:
                    info['gpu_count'] = torch.cuda.device_count()
                    info['platform'] = 'cuda'
                    
                    # Get GPU device information
                    gpu_devices = []
                    total_gpu_memory = 0
                    
                    for i in range(torch.cuda.device_count()):
                        device_props = torch.cuda.get_device_properties(i)
                        gpu_memory_gb = device_props.total_memory / (1024**3)
                        total_gpu_memory += gpu_memory_gb
                        
                        gpu_devices.append({
                            'id': i,
                            'name': device_props.name,
                            'memory_gb': round(gpu_memory_gb, 2),
                            'compute_capability': f"{device_props.major}.{device_props.minor}",
                            'multiprocessor_count': device_props.multiprocessor_count
                        })
                    
                    info['gpu_memory_gb'] = round(total_gpu_memory, 2)
                    info['gpu_devices'] = gpu_devices
                    
                except Exception as e:
                    logger.warning(f"Failed to get detailed GPU info: {e}")
                    info['gpu_available'] = False
                    info['platform'] = 'cpu'
            
            # Additional GPU information with GPUtil if available
            if GPU_UTIL_AVAILABLE and info['gpu_available']:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        for i, gpu in enumerate(gpus):
                            if i < len(info['gpu_devices']):
                                info['gpu_devices'][i].update({
                                    'utilization': gpu.load * 100,
                                    'memory_used_gb': round(gpu.memoryUsed / 1024, 2),
                                    'memory_free_gb': round(gpu.memoryFree / 1024, 2),
                                    'temperature': gpu.temperature
                                })
                except Exception as e:
                    logger.warning(f"Failed to get GPU utilization info: {e}")
            
            # Detect deployment environment
            info['deployment_environment'] = self._detect_deployment_environment(info)
            
            return info
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            # Return minimal fallback configuration
            return {
                'cpu_cores': 1,
                'cpu_logical_cores': 1,
                'ram_gb': 1.0,
                'available_ram_gb': 0.5,
                'gpu_available': False,
                'gpu_count': 0,
                'gpu_memory_gb': 0,
                'gpu_devices': [],
                'platform': 'cpu',
                'os_type': 'unknown',
                'architecture': 'unknown',
                'python_version': '3.9',
                'deployment_environment': 'serverless'
            }
    
    def _detect_deployment_environment(self, hardware_info: Dict[str, Any]) -> str:
        """Detect deployment environment (local, railway, vercel, etc.)"""
        try:
            # Check environment variables for deployment platforms
            if os.getenv('VERCEL'):
                return 'vercel'
            elif os.getenv('RAILWAY_ENVIRONMENT'):
                return 'railway'
            elif os.getenv('RENDER'):
                return 'render'
            elif os.getenv('HEROKU_APP_NAME'):
                return 'heroku'
            elif os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
                return 'aws_lambda'
            elif os.getenv('GOOGLE_CLOUD_PROJECT'):
                return 'gcp'
            elif os.getenv('AZURE_FUNCTIONS_ENVIRONMENT'):
                return 'azure'
            elif hardware_info['ram_gb'] < 2:
                return 'serverless'
            elif hardware_info['ram_gb'] < 4:
                return 'container'
            else:
                return 'dedicated'
                
        except Exception as e:
            logger.warning(f"Failed to detect deployment environment: {e}")
            return 'unknown'
    
    def _generate_optimization_config(self) -> Dict[str, Any]:
        """Generate optimization configuration based on hardware"""
        hw = self.hardware_info
        config = {
            'memory_optimization': True,
            'cpu_optimization': True,
            'gpu_optimization': hw['gpu_available'],
            'batch_processing': True,
            'caching_enabled': True,
            'async_processing': True
        }
        
        # Environment-specific optimizations
        env = hw['deployment_environment']
        if env in ['vercel', 'serverless', 'aws_lambda']:
            config.update({
                'aggressive_memory_optimization': True,
                'minimal_model_loading': True,
                'fast_startup': True,
                'reduced_caching': True
            })
        elif env in ['railway', 'render', 'container']:
            config.update({
                'moderate_memory_optimization': True,
                'balanced_performance': True,
                'smart_caching': True
            })
        else:
            config.update({
                'full_feature_set': True,
                'extensive_caching': True,
                'parallel_processing': True
            })
        
        return config
    
    def select_optimal_model_config(self) -> Dict[str, Any]:
        """Select model configuration based on hardware capabilities"""
        hw = self.hardware_info
        env = hw['deployment_environment']
        
        # Base configuration
        config = {
            'model_name': 'HooshvareLab/bert-base-parsbert-uncased',
            'quantization': False,
            'use_dora': True,
            'use_qr_adaptor': False,
            'batch_size': 8,
            'max_length': 512,
            'device': 'cpu',
            'precision': 'fp32',
            'optimization_level': 'standard',
            'memory_efficient': False,
            'fast_inference': False
        }
        
        # High-end GPU configuration (8GB+ VRAM)
        if hw['gpu_available'] and hw['gpu_memory_gb'] >= 8:
            config.update({
                'model_name': 'HooshvareLab/bert-base-parsbert-uncased',
                'batch_size': 16,
                'use_dora': True,
                'use_qr_adaptor': True,
                'device': 'cuda',
                'precision': 'fp16',
                'optimization_level': 'high',
                'memory_efficient': False,
                'fast_inference': True
            })
            logger.info("Selected high-end GPU configuration")
            
        # Mid-range GPU configuration (4-8GB VRAM)
        elif hw['gpu_available'] and hw['gpu_memory_gb'] >= 4:
            config.update({
                'model_name': 'HooshvareLab/bert-base-parsbert-uncased',
                'batch_size': 8,
                'use_dora': True,
                'quantization': True,
                'device': 'cuda',
                'precision': 'fp16',
                'optimization_level': 'medium',
                'memory_efficient': True
            })
            logger.info("Selected mid-range GPU configuration")
            
        # Low-end GPU configuration (2-4GB VRAM)
        elif hw['gpu_available'] and hw['gpu_memory_gb'] >= 2:
            config.update({
                'model_name': 'distilbert-base-multilingual-cased',
                'batch_size': 4,
                'use_dora': False,
                'quantization': True,
                'device': 'cuda',
                'optimization_level': 'medium',
                'memory_efficient': True
            })
            logger.info("Selected low-end GPU configuration")
            
        # High RAM CPU configuration (8GB+ RAM)
        elif hw['ram_gb'] >= 8:
            config.update({
                'model_name': 'HooshvareLab/bert-base-parsbert-uncased',
                'batch_size': 4,
                'use_dora': False,
                'quantization': True,
                'device': 'cpu',
                'optimization_level': 'medium',
                'memory_efficient': True
            })
            logger.info("Selected high-RAM CPU configuration")
            
        # Medium RAM CPU configuration (4-8GB RAM)
        elif hw['ram_gb'] >= 4:
            config.update({
                'model_name': 'distilbert-base-multilingual-cased',
                'batch_size': 2,
                'use_dora': False,
                'quantization': True,
                'max_length': 256,
                'device': 'cpu',
                'optimization_level': 'low',
                'memory_efficient': True
            })
            logger.info("Selected medium-RAM CPU configuration")
            
        # Low memory configuration (Vercel/serverless)
        else:
            config.update({
                'model_name': 'distilbert-base-multilingual-cased',
                'batch_size': 1,
                'use_dora': False,
                'quantization': True,
                'max_length': 128,
                'device': 'cpu',
                'optimization_level': 'minimal',
                'memory_efficient': True,
                'fast_inference': True
            })
            logger.info("Selected low-memory serverless configuration")
        
        # Environment-specific adjustments
        if env in ['vercel', 'serverless', 'aws_lambda']:
            config.update({
                'batch_size': min(config['batch_size'], 1),
                'max_length': min(config['max_length'], 128),
                'quantization': True,
                'memory_efficient': True,
                'fast_inference': True
            })
        
        # Add hardware-specific metadata
        config['hardware_profile'] = {
            'cpu_cores': hw['cpu_cores'],
            'ram_gb': hw['ram_gb'],
            'gpu_available': hw['gpu_available'],
            'gpu_memory_gb': hw['gpu_memory_gb'],
            'deployment_environment': env,
            'optimization_applied': True
        }
        
        return config
    
    def get_hardware_summary(self) -> str:
        """Get human-readable hardware summary"""
        hw = self.hardware_info
        summary = f"CPU: {hw['cpu_cores']}c/{hw['cpu_logical_cores']}t, RAM: {hw['ram_gb']}GB"
        
        if hw['gpu_available']:
            gpu_info = f", GPU: {hw['gpu_count']}x ({hw['gpu_memory_gb']:.1f}GB total)"
            if hw['gpu_devices']:
                gpu_names = [gpu['name'] for gpu in hw['gpu_devices']]
                gpu_info += f" [{', '.join(set(gpu_names))}]"
            summary += gpu_info
        else:
            summary += ", GPU: None"
        
        summary += f", Env: {hw['deployment_environment']}"
        return summary
    
    def get_performance_recommendations(self) -> Dict[str, Any]:
        """Get performance recommendations based on hardware"""
        hw = self.hardware_info
        config = self.select_optimal_model_config()
        
        recommendations = {
            'hardware_optimal': True,
            'recommendations': [],
            'warnings': [],
            'optimizations': []
        }
        
        # Memory recommendations
        if hw['ram_gb'] < 4:
            recommendations['recommendations'].append(
                "Consider upgrading to at least 4GB RAM for better performance"
            )
            recommendations['optimizations'].append("Aggressive memory optimization enabled")
        
        # GPU recommendations
        if not hw['gpu_available']:
            recommendations['recommendations'].append(
                "GPU acceleration would significantly improve performance"
            )
        elif hw['gpu_memory_gb'] < 4:
            recommendations['warnings'].append(
                "Limited GPU memory may cause out-of-memory errors with large batches"
            )
        
        # Environment-specific recommendations
        env = hw['deployment_environment']
        if env in ['vercel', 'serverless']:
            recommendations['warnings'].append(
                "Serverless environment detected - model loading time may be high"
            )
            recommendations['optimizations'].append("Fast startup optimization enabled")
        
        # Model-specific recommendations
        if config['quantization']:
            recommendations['optimizations'].append("Model quantization enabled for memory efficiency")
        
        if config['memory_efficient']:
            recommendations['optimizations'].append("Memory-efficient inference enabled")
        
        return recommendations
    
    def is_production_ready(self) -> Dict[str, Any]:
        """Check if hardware configuration is production ready"""
        hw = self.hardware_info
        config = self.select_optimal_model_config()
        
        checks = {
            'memory_sufficient': hw['available_ram_gb'] >= 1.0,
            'model_loadable': True,  # Will be verified during model loading
            'performance_acceptable': hw['cpu_cores'] >= 1,
            'environment_supported': hw['deployment_environment'] != 'unknown',
            'optimization_enabled': config['optimization_level'] != 'none'
        }
        
        overall_ready = all(checks.values())
        
        return {
            'production_ready': overall_ready,
            'checks': checks,
            'hardware_score': self._calculate_hardware_score(),
            'recommended_config': config,
            'summary': self.get_hardware_summary()
        }
    
    def _calculate_hardware_score(self) -> float:
        """Calculate overall hardware capability score (0-100)"""
        hw = self.hardware_info
        score = 0
        
        # CPU score (0-25 points)
        cpu_score = min(25, hw['cpu_cores'] * 5)
        score += cpu_score
        
        # Memory score (0-25 points)
        memory_score = min(25, hw['ram_gb'] * 3)
        score += memory_score
        
        # GPU score (0-50 points)
        if hw['gpu_available']:
            gpu_score = min(50, hw['gpu_memory_gb'] * 6)
            score += gpu_score
        else:
            # CPU-only systems get partial GPU score based on CPU power
            score += min(20, hw['cpu_cores'] * 2)
        
        return round(score, 1)

# Global instance for easy access
hardware_detector = HardwareDetector()

# Convenience functions for backward compatibility
def get_hardware_info() -> Dict[str, Any]:
    """Get hardware information - convenience function"""
    return hardware_detector.hardware_info

def get_optimal_model_config() -> Dict[str, Any]:
    """Get optimal model configuration - convenience function"""
    return hardware_detector.select_optimal_model_config()

def get_hardware_summary() -> str:
    """Get hardware summary - convenience function"""
    return hardware_detector.get_hardware_summary()