"""
System API Endpoints for Persian Legal AI
نقاط پایانی API سیستم برای هوش مصنوعی حقوقی فارسی
"""

import logging
import psutil
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..database.connection import db_manager
from ..services.training_service import training_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])

class SystemInfoModel(BaseModel):
    cpu_cores: int
    memory_gb: float
    disk_space_gb: float
    os_info: str
    python_version: str

class SystemMetricsModel(BaseModel):
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_processes: int

@router.get("/info")
async def get_system_info():
    """Get system information"""
    try:
        import platform
        import sys
        
        # Get system information
        cpu_count = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return SystemInfoModel(
            cpu_cores=cpu_count,
            memory_gb=memory.total / (1024**3),
            disk_space_gb=disk.free / (1024**3),
            os_info=f"{platform.system()} {platform.release()}",
            python_version=sys.version
        ).dict()
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_system_metrics():
    """Get current system metrics"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return SystemMetricsModel(
            timestamp=datetime.utcnow(),
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            disk_usage=(disk.used / disk.total) * 100,
            network_io={
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            },
            active_processes=len(psutil.pids())
        ).dict()
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_system_performance():
    """Get comprehensive system performance"""
    try:
        # Get training service performance
        training_performance = await training_service.get_system_performance()
        
        # Get database info
        db_info = db_manager.get_database_info()
        
        # Get system metrics
        system_metrics = await get_system_metrics()
        
        return {
            "system_metrics": system_metrics,
            "training_performance": training_performance,
            "database_info": db_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def system_health_check():
    """Comprehensive system health check"""
    try:
        health_status = {
            "overall_status": "healthy",
            "components": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check database
        try:
            db_healthy = db_manager.test_connection()
            health_status["components"]["database"] = {
                "status": "healthy" if db_healthy else "unhealthy",
                "message": "Connected" if db_healthy else "Connection failed"
            }
        except Exception as e:
            health_status["components"]["database"] = {
                "status": "unhealthy",
                "message": str(e)
            }
        
        # Check system resources
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            memory_healthy = memory.percent < 90
            disk_healthy = (disk.used / disk.total) * 100 < 90
            
            health_status["components"]["system_resources"] = {
                "status": "healthy" if memory_healthy and disk_healthy else "warning",
                "memory_usage": memory.percent,
                "disk_usage": (disk.used / disk.total) * 100,
                "message": "Resources available" if memory_healthy and disk_healthy else "High resource usage"
            }
        except Exception as e:
            health_status["components"]["system_resources"] = {
                "status": "unhealthy",
                "message": str(e)
            }
        
        # Check training service
        try:
            training_sessions = await training_service.list_training_sessions()
            health_status["components"]["training_service"] = {
                "status": "healthy",
                "active_sessions": len([s for s in training_sessions if s.get('status') == 'running']),
                "total_sessions": len(training_sessions),
                "message": "Training service operational"
            }
        except Exception as e:
            health_status["components"]["training_service"] = {
                "status": "unhealthy",
                "message": str(e)
            }
        
        # Determine overall status
        component_statuses = [comp["status"] for comp in health_status["components"].values()]
        if "unhealthy" in component_statuses:
            health_status["overall_status"] = "unhealthy"
        elif "warning" in component_statuses:
            health_status["overall_status"] = "warning"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "overall_status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.post("/optimize")
async def optimize_system():
    """Optimize system for training"""
    try:
        optimization_results = await training_service.optimize_system_for_training()
        
        return {
            "message": "System optimization completed",
            "results": optimization_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to optimize system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs")
async def get_system_logs(lines: int = 100):
    """Get system logs"""
    try:
        log_file = "persian_ai_backend.log"
        
        if not os.path.exists(log_file):
            return {"message": "No log file found", "logs": []}
        
        # Read last N lines from log file
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "log_file": log_file,
            "lines_requested": lines,
            "lines_returned": len(recent_lines),
            "logs": [line.strip() for line in recent_lines]
        }
        
    except Exception as e:
        logger.error(f"Failed to get system logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/database/stats")
async def get_database_stats():
    """Get database statistics"""
    try:
        # Get table statistics
        table_stats = db_manager.get_table_stats()
        
        # Get database info
        db_info = db_manager.get_database_info()
        
        return {
            "database_info": db_info,
            "table_stats": table_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/database/cleanup")
async def cleanup_database(days: int = 30):
    """Cleanup old database records"""
    try:
        cleaned_records = db_manager.cleanup_old_data(days)
        
        return {
            "message": f"Database cleanup completed",
            "records_cleaned": cleaned_records,
            "days_threshold": days,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/database/backup")
async def backup_database():
    """Backup database"""
    try:
        backup_path = f"backups/persian_ai_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        os.makedirs("backups", exist_ok=True)
        
        success = db_manager.backup_database(backup_path)
        
        if success:
            return {
                "message": "Database backup completed",
                "backup_path": backup_path,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Backup failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to backup database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/database/optimize")
async def optimize_database():
    """Optimize database performance"""
    try:
        success = db_manager.optimize_database()
        
        if success:
            return {
                "message": "Database optimization completed",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Optimization failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to optimize database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/processes")
async def get_active_processes():
    """Get information about active processes"""
    try:
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                proc_info = proc.info
                processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort by CPU usage
        processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
        
        return {
            "total_processes": len(processes),
            "top_processes": processes[:20],  # Top 20 by CPU usage
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get active processes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/network")
async def get_network_info():
    """Get network information"""
    try:
        network_stats = psutil.net_io_counters()
        network_connections = psutil.net_connections()
        
        return {
            "network_io": {
                "bytes_sent": network_stats.bytes_sent,
                "bytes_recv": network_stats.bytes_recv,
                "packets_sent": network_stats.packets_sent,
                "packets_recv": network_stats.packets_recv,
                "errin": network_stats.errin,
                "errout": network_stats.errout,
                "dropin": network_stats.dropin,
                "dropout": network_stats.dropout
            },
            "active_connections": len(network_connections),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get network info: {e}")
        raise HTTPException(status_code=500, detail=str(e))