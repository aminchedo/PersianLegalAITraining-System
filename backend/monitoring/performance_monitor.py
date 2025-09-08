import psutil
import torch
import asyncio
import time
from typing import Dict, List
from datetime import datetime, timedelta
import logging

class SystemPerformanceMonitor:
    """Comprehensive system performance monitoring"""
    
    def __init__(self):
        self.metrics_history: List[Dict] = []
        self.max_history_size = 1000
        self.logger = logging.getLogger(__name__)
    
    async def get_system_metrics(self) -> Dict:
        """Collect comprehensive system metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU metrics (if available)
            gpu_metrics = {}
            if torch.cuda.is_available():
                gpu_metrics = {
                    'gpu_available': True,
                    'gpu_count': torch.cuda.device_count(),
                    'gpu_memory_allocated': torch.cuda.memory_allocated(0) / 1024**3,  # GB
                    'gpu_memory_cached': torch.cuda.memory_reserved(0) / 1024**3,  # GB
                    'gpu_utilization': self._get_gpu_utilization(),
                }
            else:
                gpu_metrics = {'gpu_available': False}
            
            # AI Model metrics
            model_metrics = await self._get_model_metrics()
            
            # Performance metrics
            performance_metrics = {
                'api_response_time': await self._measure_api_performance(),
                'database_query_time': await self._measure_db_performance(),
                'classification_speed': await self._measure_classification_speed(),
            }
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / 1024**3,
                    'memory_total_gb': memory.total / 1024**3,
                    'disk_percent': disk.percent,
                    'disk_free_gb': disk.free / 1024**3,
                    'disk_total_gb': disk.total / 1024**3,
                },
                'gpu': gpu_metrics,
                'models': model_metrics,
                'performance': performance_metrics,
                'health_score': self._calculate_health_score()
            }
            
            # Store in history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except:
            return 0.0
    
    async def _get_model_metrics(self) -> Dict:
        """Get AI model performance metrics"""
        try:
            return {
                'models_loaded': True,
                'model_size_mb': 654,  # HooshvareLab BERT size
                'pytorch_version': torch.__version__,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'precision': 'fp16' if torch.cuda.is_available() else 'fp32',
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def _measure_api_performance(self) -> float:
        """Measure API response time"""
        try:
            import aiohttp
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8000/api/test/ping') as response:
                    await response.text()
            
            return round((time.time() - start_time) * 1000, 2)  # ms
        except:
            return -1
    
    async def _measure_db_performance(self) -> float:
        """Measure database query performance"""
        try:
            from backend.database.connection import PersianLegalDatabase
            db = PersianLegalDatabase()
            
            start_time = time.time()
            stats = await db.get_database_statistics()
            
            return round((time.time() - start_time) * 1000, 2)  # ms
        except:
            return -1
    
    async def _measure_classification_speed(self) -> float:
        """Measure AI classification speed"""
        try:
            from backend.models.persian_legal_classifier import PersianLegalAIClassifier
            classifier = PersianLegalAIClassifier()
            
            test_text = "این متن تستی برای سنجش سرعت طبقه‌بندی است."
            
            start_time = time.time()
            await classifier.classify_document(test_text)
            
            return round((time.time() - start_time) * 1000, 2)  # ms
        except:
            return -1
    
    def _calculate_health_score(self) -> int:
        """Calculate overall system health score (0-100)"""
        try:
            if not self.metrics_history:
                return 50
            
            latest = self.metrics_history[-1]
            score = 100
            
            # Penalize high CPU usage
            if latest['system']['cpu_percent'] > 80:
                score -= 20
            elif latest['system']['cpu_percent'] > 60:
                score -= 10
            
            # Penalize high memory usage
            if latest['system']['memory_percent'] > 90:
                score -= 30
            elif latest['system']['memory_percent'] > 75:
                score -= 15
            
            # Penalize slow API response
            if latest['performance']['api_response_time'] > 1000:
                score -= 20
            elif latest['performance']['api_response_time'] > 500:
                score -= 10
            
            # Penalize disk space issues
            if latest['system']['disk_percent'] > 95:
                score -= 25
            elif latest['system']['disk_percent'] > 85:
                score -= 10
            
            return max(0, min(100, score))
            
        except:
            return 50
    
    async def get_performance_summary(self, hours: int = 24) -> Dict:
        """Get performance summary over time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_metrics = [
                m for m in self.metrics_history
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
            
            if not recent_metrics:
                return {'error': 'No recent metrics available'}
            
            # Calculate averages
            avg_cpu = sum(m['system']['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m['system']['memory_percent'] for m in recent_metrics) / len(recent_metrics)
            avg_api_time = sum(m['performance']['api_response_time'] for m in recent_metrics if m['performance']['api_response_time'] > 0) / len(recent_metrics)
            avg_health = sum(m['health_score'] for m in recent_metrics) / len(recent_metrics)
            
            return {
                'period_hours': hours,
                'samples_count': len(recent_metrics),
                'averages': {
                    'cpu_percent': round(avg_cpu, 2),
                    'memory_percent': round(avg_memory, 2),
                    'api_response_time_ms': round(avg_api_time, 2),
                    'health_score': round(avg_health, 1),
                },
                'current_status': 'healthy' if avg_health > 70 else 'degraded' if avg_health > 40 else 'critical',
                'recommendations': self._generate_recommendations(recent_metrics[-1] if recent_metrics else {})
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_recommendations(self, latest_metrics: Dict) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        try:
            if latest_metrics.get('system', {}).get('cpu_percent', 0) > 80:
                recommendations.append("High CPU usage detected. Consider scaling horizontally or optimizing intensive operations.")
            
            if latest_metrics.get('system', {}).get('memory_percent', 0) > 85:
                recommendations.append("High memory usage. Consider increasing memory allocation or optimizing memory-intensive processes.")
            
            if latest_metrics.get('performance', {}).get('api_response_time', 0) > 500:
                recommendations.append("Slow API responses. Check database queries and consider adding caching.")
            
            if not latest_metrics.get('gpu', {}).get('gpu_available', False):
                recommendations.append("GPU not available. Consider enabling GPU acceleration for faster AI inference.")
            
            if latest_metrics.get('system', {}).get('disk_percent', 0) > 80:
                recommendations.append("Disk space running low. Clean up logs and temporary files.")
        
        except:
            recommendations.append("Unable to generate recommendations due to missing metrics.")
        
        return recommendations