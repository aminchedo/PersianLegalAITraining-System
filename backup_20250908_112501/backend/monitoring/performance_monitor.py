"""
System Performance Monitor Module
Minimal implementation for system startup
"""

import psutil
import logging
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)

class SystemPerformanceMonitor:
    """Minimal System Performance Monitor for testing"""
    
    def __init__(self):
        self.start_time = datetime.now()
        logger.info("System Performance Monitor initialized")
    
    def get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "cpu": {
                    "percent": cpu_percent,
                    "cores": psutil.cpu_count()
                },
                "memory": {
                    "total_mb": memory.total / (1024 * 1024),
                    "used_mb": memory.used / (1024 * 1024),
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": disk.total / (1024 * 1024 * 1024),
                    "used_gb": disk.used / (1024 * 1024 * 1024),
                    "percent": disk.percent
                }
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        metrics = self.get_system_metrics()
        
        if "error" in metrics:
            return metrics
        
        # Add performance assessment
        cpu_status = "normal" if metrics["cpu"]["percent"] < 80 else "high"
        memory_status = "normal" if metrics["memory"]["percent"] < 80 else "high"
        disk_status = "normal" if metrics["disk"]["percent"] < 90 else "high"
        
        return {
            "metrics": metrics,
            "assessment": {
                "cpu_status": cpu_status,
                "memory_status": memory_status,
                "disk_status": disk_status,
                "overall": "healthy" if all(s == "normal" for s in [cpu_status, memory_status, disk_status]) else "attention_needed"
            }
        }
    
    def check_health(self) -> bool:
        """Simple health check"""
        try:
            metrics = self.get_system_metrics()
            return "error" not in metrics
        except:
            return False