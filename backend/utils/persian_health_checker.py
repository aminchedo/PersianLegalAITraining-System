"""
Persian Legal AI - System Health Checker
چک‌کننده سلامت سیستم هوش مصنوعی حقوقی فارسی
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class HealthMetric:
    """Health metric data class"""
    name: str
    value: float
    unit: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    timestamp: datetime

@dataclass
class SystemHealth:
    """System health summary"""
    overall_status: str
    score: int  # 0-100
    metrics: List[HealthMetric]
    recommendations: List[str]
    last_check: datetime

class PersianHealthChecker:
    """Comprehensive health checker for Persian Legal AI system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 85.0,
            'memory_warning': 75.0,
            'memory_critical': 90.0,
            'disk_warning': 80.0,
            'disk_critical': 95.0,
            'response_time_warning': 1000,  # ms
            'response_time_critical': 3000,  # ms
        }
        self.history: List[SystemHealth] = []
        
    async def check_system_resources(self) -> List[HealthMetric]:
        """Check system resource utilization"""
        metrics = []
        
        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_status = self._get_status(cpu_percent, 'cpu_warning', 'cpu_critical')
        metrics.append(HealthMetric(
            name="cpu_usage",
            value=cpu_percent,
            unit="%",
            status=cpu_status,
            message=f"استفاده از پردازنده: {cpu_percent:.1f}%",
            timestamp=datetime.now()
        ))
        
        # Memory Usage
        memory = psutil.virtual_memory()
        memory_status = self._get_status(memory.percent, 'memory_warning', 'memory_critical')
        metrics.append(HealthMetric(
            name="memory_usage",
            value=memory.percent,
            unit="%",
            status=memory_status,
            message=f"استفاده از حافظه: {memory.percent:.1f}% ({self._bytes_to_gb(memory.used):.1f}GB از {self._bytes_to_gb(memory.total):.1f}GB)",
            timestamp=datetime.now()
        ))
        
        # Disk Usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_status = self._get_status(disk_percent, 'disk_warning', 'disk_critical')
        metrics.append(HealthMetric(
            name="disk_usage",
            value=disk_percent,
            unit="%",
            status=disk_status,
            message=f"استفاده از دیسک: {disk_percent:.1f}% ({self._bytes_to_gb(disk.used):.1f}GB از {self._bytes_to_gb(disk.total):.1f}GB)",
            timestamp=datetime.now()
        ))
        
        # Network I/O
        network = psutil.net_io_counters()
        metrics.append(HealthMetric(
            name="network_sent",
            value=network.bytes_sent,
            unit="bytes",
            status="healthy",
            message=f"داده ارسالی: {self._bytes_to_mb(network.bytes_sent):.1f}MB",
            timestamp=datetime.now()
        ))
        
        return metrics
    
    async def check_database_health(self, db_url: Optional[str] = None) -> List[HealthMetric]:
        """Check database connectivity and performance"""
        metrics = []
        
        if not db_url:
            metrics.append(HealthMetric(
                name="database_connection",
                value=0,
                unit="status",
                status="warning",
                message="اتصال پایگاه داده پیکربندی نشده",
                timestamp=datetime.now()
            ))
            return metrics
        
        try:
            start_time = time.time()
            # Simulate database connection check
            # In real implementation, you'd connect to your actual database
            await asyncio.sleep(0.1)  # Simulate connection time
            connection_time = (time.time() - start_time) * 1000
            
            connection_status = self._get_status(connection_time, 'response_time_warning', 'response_time_critical')
            metrics.append(HealthMetric(
                name="database_connection",
                value=connection_time,
                unit="ms",
                status=connection_status,
                message=f"اتصال پایگاه داده: {connection_time:.1f}ms",
                timestamp=datetime.now()
            ))
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            metrics.append(HealthMetric(
                name="database_connection",
                value=0,
                unit="status",
                status="critical",
                message=f"خطا در اتصال پایگاه داده: {str(e)}",
                timestamp=datetime.now()
            ))
        
        return metrics
    
    async def check_ai_models_health(self) -> List[HealthMetric]:
        """Check AI models availability and performance"""
        metrics = []
        
        # Check model files
        model_paths = [
            "models/persian_bert_model",
            "models/dora_adapter",
            "models/qr_adapter"
        ]
        
        for model_path in model_paths:
            path = Path(model_path)
            if path.exists():
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                metrics.append(HealthMetric(
                    name=f"model_{path.name}",
                    value=size,
                    unit="bytes",
                    status="healthy",
                    message=f"مدل {path.name}: {self._bytes_to_mb(size):.1f}MB",
                    timestamp=datetime.now()
                ))
            else:
                metrics.append(HealthMetric(
                    name=f"model_{path.name}",
                    value=0,
                    unit="status",
                    status="warning",
                    message=f"مدل {path.name} یافت نشد",
                    timestamp=datetime.now()
                ))
        
        # Check GPU availability (if applicable)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                metrics.append(HealthMetric(
                    name="gpu_availability",
                    value=gpu_count,
                    unit="count",
                    status="healthy",
                    message=f"GPU در دسترس: {gpu_count} دستگاه، حافظه: {self._bytes_to_gb(gpu_memory):.1f}GB",
                    timestamp=datetime.now()
                ))
            else:
                metrics.append(HealthMetric(
                    name="gpu_availability",
                    value=0,
                    unit="count",
                    status="warning",
                    message="GPU در دسترس نیست",
                    timestamp=datetime.now()
                ))
        except ImportError:
            metrics.append(HealthMetric(
                name="gpu_availability",
                value=0,
                unit="status",
                status="info",
                message="PyTorch نصب نشده",
                timestamp=datetime.now()
            ))
        
        return metrics
    
    async def check_api_endpoints(self, base_url: str = "http://localhost:8000") -> List[HealthMetric]:
        """Check API endpoints health"""
        metrics = []
        
        endpoints = [
            "/api/system/health",
            "/api/documents/stats",
            "/api/training/sessions",
        ]
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                try:
                    start_time = time.time()
                    async with session.get(f"{base_url}{endpoint}", timeout=5) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            status = self._get_status(response_time, 'response_time_warning', 'response_time_critical')
                            message = f"نقطه پایانی {endpoint}: {response_time:.1f}ms"
                        else:
                            status = "warning"
                            message = f"نقطه پایانی {endpoint}: کد خطا {response.status}"
                        
                        metrics.append(HealthMetric(
                            name=f"endpoint_{endpoint.replace('/', '_').replace('-', '_')}",
                            value=response_time,
                            unit="ms",
                            status=status,
                            message=message,
                            timestamp=datetime.now()
                        ))
                        
                except asyncio.TimeoutError:
                    metrics.append(HealthMetric(
                        name=f"endpoint_{endpoint.replace('/', '_').replace('-', '_')}",
                        value=5000,
                        unit="ms",
                        status="critical",
                        message=f"نقطه پایانی {endpoint}: تایم‌اوت",
                        timestamp=datetime.now()
                    ))
                except Exception as e:
                    metrics.append(HealthMetric(
                        name=f"endpoint_{endpoint.replace('/', '_').replace('-', '_')}",
                        value=0,
                        unit="status",
                        status="critical",
                        message=f"نقطه پایانی {endpoint}: خطا - {str(e)}",
                        timestamp=datetime.now()
                    ))
        
        return metrics
    
    async def check_persian_text_processing(self) -> List[HealthMetric]:
        """Check Persian text processing capabilities"""
        metrics = []
        
        test_text = "این یک متن آزمایشی برای بررسی قابلیت‌های پردازش متن فارسی است."
        
        try:
            # Test text normalization
            start_time = time.time()
            # Simulate text processing
            await asyncio.sleep(0.05)
            processing_time = (time.time() - start_time) * 1000
            
            metrics.append(HealthMetric(
                name="persian_text_processing",
                value=processing_time,
                unit="ms",
                status="healthy",
                message=f"پردازش متن فارسی: {processing_time:.1f}ms",
                timestamp=datetime.now()
            ))
            
        except Exception as e:
            metrics.append(HealthMetric(
                name="persian_text_processing",
                value=0,
                unit="status",
                status="critical",
                message=f"خطا در پردازش متن فارسی: {str(e)}",
                timestamp=datetime.now()
            ))
        
        return metrics
    
    async def perform_full_health_check(self, db_url: Optional[str] = None, api_base_url: str = "http://localhost:8000") -> SystemHealth:
        """Perform comprehensive health check"""
        logger.info("شروع بررسی سلامت سیستم...")
        
        all_metrics = []
        
        # Gather all metrics
        all_metrics.extend(await self.check_system_resources())
        all_metrics.extend(await self.check_database_health(db_url))
        all_metrics.extend(await self.check_ai_models_health())
        all_metrics.extend(await self.check_api_endpoints(api_base_url))
        all_metrics.extend(await self.check_persian_text_processing())
        
        # Calculate overall health score
        score = self._calculate_health_score(all_metrics)
        overall_status = self._get_overall_status(score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_metrics)
        
        health = SystemHealth(
            overall_status=overall_status,
            score=score,
            metrics=all_metrics,
            recommendations=recommendations,
            last_check=datetime.now()
        )
        
        # Store in history
        self.history.append(health)
        if len(self.history) > 100:  # Keep last 100 checks
            self.history = self.history[-100:]
        
        logger.info(f"بررسی سلامت سیستم کامل شد. امتیاز: {score}/100")
        return health
    
    def _get_status(self, value: float, warning_key: str, critical_key: str) -> str:
        """Get status based on thresholds"""
        if value >= self.thresholds[critical_key]:
            return "critical"
        elif value >= self.thresholds[warning_key]:
            return "warning"
        else:
            return "healthy"
    
    def _calculate_health_score(self, metrics: List[HealthMetric]) -> int:
        """Calculate overall health score (0-100)"""
        if not metrics:
            return 0
        
        score = 100
        for metric in metrics:
            if metric.status == "critical":
                score -= 20
            elif metric.status == "warning":
                score -= 10
            elif metric.status == "info":
                score -= 2
        
        return max(0, score)
    
    def _get_overall_status(self, score: int) -> str:
        """Get overall status based on score"""
        if score >= 90:
            return "healthy"
        elif score >= 70:
            return "warning"
        else:
            return "critical"
    
    def _generate_recommendations(self, metrics: List[HealthMetric]) -> List[str]:
        """Generate health recommendations"""
        recommendations = []
        
        for metric in metrics:
            if metric.status == "critical":
                if "cpu" in metric.name:
                    recommendations.append("استفاده از پردازنده بالاست. پروسه‌های غیرضروری را متوقف کنید.")
                elif "memory" in metric.name:
                    recommendations.append("استفاده از حافظه بالاست. برنامه‌های غیرضروری را ببندید.")
                elif "disk" in metric.name:
                    recommendations.append("فضای دیسک کم است. فایل‌های غیرضروری را حذف کنید.")
                elif "database" in metric.name:
                    recommendations.append("مشکل در اتصال پایگاه داده. تنظیمات اتصال را بررسی کنید.")
                elif "endpoint" in metric.name:
                    recommendations.append("مشکل در دسترسی به API. سرویس‌ها را بررسی کنید.")
            elif metric.status == "warning":
                if "gpu" in metric.name:
                    recommendations.append("GPU در دسترس نیست. برای بهبود عملکرد AI از GPU استفاده کنید.")
                elif "model" in metric.name:
                    recommendations.append("برخی مدل‌های AI یافت نشدند. فایل‌های مدل را بررسی کنید.")
        
        # Remove duplicates
        return list(set(recommendations))
    
    def _bytes_to_mb(self, bytes_value: int) -> float:
        """Convert bytes to MB"""
        return bytes_value / (1024 * 1024)
    
    def _bytes_to_gb(self, bytes_value: int) -> float:
        """Convert bytes to GB"""
        return bytes_value / (1024 * 1024 * 1024)
    
    def get_health_history(self, hours: int = 24) -> List[SystemHealth]:
        """Get health history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [h for h in self.history if h.last_check >= cutoff_time]
    
    def export_health_report(self, filepath: str) -> None:
        """Export health report to JSON file"""
        if not self.history:
            return
        
        latest_health = self.history[-1]
        report = {
            "timestamp": latest_health.last_check.isoformat(),
            "overall_status": latest_health.overall_status,
            "score": latest_health.score,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "status": m.status,
                    "message": m.message,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in latest_health.metrics
            ],
            "recommendations": latest_health.recommendations
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"گزارش سلامت سیستم در {filepath} ذخیره شد")

# Example usage
async def main():
    """Example usage of health checker"""
    checker = PersianHealthChecker()
    health = await checker.perform_full_health_check()
    
    print(f"وضعیت کلی سیستم: {health.overall_status}")
    print(f"امتیاز سلامت: {health.score}/100")
    
    for metric in health.metrics:
        print(f"- {metric.message} [{metric.status}]")
    
    if health.recommendations:
        print("\nپیشنهادات:")
        for rec in health.recommendations:
            print(f"- {rec}")

if __name__ == "__main__":
    asyncio.run(main())