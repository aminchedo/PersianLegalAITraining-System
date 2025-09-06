from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
import psutil
import subprocess
import time
import json
from pathlib import Path
import torch
import GPUtil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Response models
class SystemMetrics(BaseModel):
    cpu_usage: float
    memory_usage: float
    memory_available: float
    memory_total: float
    disk_usage: float
    disk_available: float
    disk_total: float
    gpu_usage: Optional[float] = None
    gpu_memory_usage: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_temperature: Optional[float] = None
    timestamp: float

class ProcessInfo(BaseModel):
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    status: str
    create_time: float

class SystemInfo(BaseModel):
    platform: str
    architecture: str
    processor: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str] = None
    gpu_count: int
    gpu_names: List[str]

# Create router
router = APIRouter(prefix="/system", tags=["system"])

@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics():
    """
    Get current system metrics including CPU, memory, disk, and GPU usage.
    
    Returns:
        Current system metrics
    """
    try:
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_available = memory.available / (1024**3)  # GB
        memory_total = memory.total / (1024**3)  # GB
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        disk_available = disk.free / (1024**3)  # GB
        disk_total = disk.total / (1024**3)  # GB
        
        # GPU metrics (if available)
        gpu_usage = None
        gpu_memory_usage = None
        gpu_memory_total = None
        gpu_temperature = None
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_usage = gpu.load * 100
                gpu_memory_usage = gpu.memoryUsed
                gpu_memory_total = gpu.memoryTotal
                gpu_temperature = gpu.temperature
        except Exception as e:
            logger.warning(f"Could not get GPU metrics: {e}")
        
        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_available=memory_available,
            memory_total=memory_total,
            disk_usage=disk_usage,
            disk_available=disk_available,
            disk_total=disk_total,
            gpu_usage=gpu_usage,
            gpu_memory_usage=gpu_memory_usage,
            gpu_memory_total=gpu_memory_total,
            gpu_temperature=gpu_temperature,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/processes", response_model=List[ProcessInfo])
async def get_system_processes(limit: int = 20):
    """
    Get information about running processes.
    
    Args:
        limit: Maximum number of processes to return
    
    Returns:
        List of process information
    """
    try:
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info', 'status', 'create_time']):
            try:
                proc_info = proc.info
                processes.append(ProcessInfo(
                    pid=proc_info['pid'],
                    name=proc_info['name'],
                    cpu_percent=proc_info['cpu_percent'] or 0.0,
                    memory_percent=proc_info['memory_percent'] or 0.0,
                    memory_mb=proc_info['memory_info'].rss / (1024**2) if proc_info['memory_info'] else 0.0,
                    status=proc_info['status'],
                    create_time=proc_info['create_time']
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        # Sort by CPU usage and limit results
        processes.sort(key=lambda x: x.cpu_percent, reverse=True)
        return processes[:limit]
        
    except Exception as e:
        logger.error(f"Error getting processes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info", response_model=SystemInfo)
async def get_system_info():
    """
    Get system information including hardware and software details.
    
    Returns:
        System information
    """
    try:
        import platform
        import sys
        
        # Basic system info
        platform_name = platform.system()
        architecture = platform.machine()
        processor = platform.processor()
        python_version = sys.version
        
        # PyTorch info
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        
        # GPU names
        gpu_names = []
        if cuda_available:
            for i in range(gpu_count):
                gpu_names.append(torch.cuda.get_device_name(i))
        
        return SystemInfo(
            platform=platform_name,
            architecture=architecture,
            processor=processor,
            python_version=python_version,
            torch_version=torch_version,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            gpu_count=gpu_count,
            gpu_names=gpu_names
        )
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Perform a comprehensive health check of the system.
    
    Returns:
        Health status and diagnostics
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {}
        }
        
        # Check system resources
        memory = psutil.virtual_memory()
        cpu_usage = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        # Memory check
        if memory.percent > 90:
            health_status["checks"]["memory"] = {
                "status": "warning",
                "message": f"High memory usage: {memory.percent:.1f}%"
            }
        else:
            health_status["checks"]["memory"] = {
                "status": "ok",
                "usage": memory.percent
            }
        
        # CPU check
        if cpu_usage > 90:
            health_status["checks"]["cpu"] = {
                "status": "warning",
                "message": f"High CPU usage: {cpu_usage:.1f}%"
            }
        else:
            health_status["checks"]["cpu"] = {
                "status": "ok",
                "usage": cpu_usage
            }
        
        # Disk check
        disk_usage_percent = (disk.used / disk.total) * 100
        if disk_usage_percent > 90:
            health_status["checks"]["disk"] = {
                "status": "warning",
                "message": f"High disk usage: {disk_usage_percent:.1f}%"
            }
        else:
            health_status["checks"]["disk"] = {
                "status": "ok",
                "usage": disk_usage_percent
            }
        
        # GPU check
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                gpu_usage_percent = (gpu_memory / gpu_memory_total) * 100
                
                if gpu_usage_percent > 90:
                    health_status["checks"]["gpu"] = {
                        "status": "warning",
                        "message": f"High GPU memory usage: {gpu_usage_percent:.1f}%"
                    }
                else:
                    health_status["checks"]["gpu"] = {
                        "status": "ok",
                        "usage": gpu_usage_percent
                    }
            except Exception as e:
                health_status["checks"]["gpu"] = {
                    "status": "error",
                    "message": f"GPU check failed: {str(e)}"
                }
        else:
            health_status["checks"]["gpu"] = {
                "status": "not_available",
                "message": "CUDA not available"
            }
        
        # Check for any warnings or errors
        warnings = [check for check in health_status["checks"].values() 
                   if check["status"] in ["warning", "error"]]
        
        if warnings:
            health_status["status"] = "warning" if not any(c["status"] == "error" for c in warnings) else "unhealthy"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error performing health check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs/system")
async def get_system_logs(lines: int = 100):
    """
    Get system logs (if available).
    
    Args:
        lines: Number of log lines to return
    
    Returns:
        System logs
    """
    try:
        # Try to get system logs from common locations
        log_paths = [
            "/var/log/syslog",
            "/var/log/messages",
            "/var/log/kern.log"
        ]
        
        logs = []
        for log_path in log_paths:
            if Path(log_path).exists():
                try:
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        all_lines = f.readlines()
                        recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                        logs.extend([f"[{log_path}] {line.strip()}" for line in recent_lines])
                except Exception as e:
                    logger.warning(f"Could not read {log_path}: {e}")
        
        if not logs:
            return {
                "status": "success",
                "logs": [],
                "message": "No system logs found or accessible"
            }
        
        return {
            "status": "success",
            "logs": logs,
            "total_lines": len(logs)
        }
        
    except Exception as e:
        logger.error(f"Error getting system logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
async def cleanup_system():
    """
    Perform system cleanup operations.
    
    Returns:
        Cleanup results
    """
    try:
        cleanup_results = {
            "status": "success",
            "timestamp": time.time(),
            "operations": {}
        }
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            cleanup_results["operations"]["pytorch_cache"] = "cleared"
        
        # Clear temporary files (if any)
        temp_dirs = [
            Path("/tmp/persian-legal-ai"),
            Path("./tmp"),
            Path("./logs/temp")
        ]
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    cleanup_results["operations"][f"temp_dir_{temp_dir.name}"] = "cleared"
                except Exception as e:
                    cleanup_results["operations"][f"temp_dir_{temp_dir.name}"] = f"error: {str(e)}"
        
        return cleanup_results
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_performance_metrics():
    """
    Get detailed performance metrics for monitoring.
    
    Returns:
        Performance metrics
    """
    try:
        # Get system metrics
        metrics = await get_system_metrics()
        
        # Get process information
        processes = await get_system_processes(limit=10)
        
        # Calculate additional metrics
        performance_data = {
            "timestamp": time.time(),
            "system_metrics": metrics.dict(),
            "top_processes": [proc.dict() for proc in processes],
            "performance_score": 0.0
        }
        
        # Calculate performance score (0-100)
        cpu_score = max(0, 100 - metrics.cpu_usage)
        memory_score = max(0, 100 - metrics.memory_usage)
        disk_score = max(0, 100 - metrics.disk_usage)
        
        performance_score = (cpu_score + memory_score + disk_score) / 3
        
        if metrics.gpu_usage is not None:
            gpu_score = max(0, 100 - metrics.gpu_usage)
            performance_score = (performance_score + gpu_score) / 2
        
        performance_data["performance_score"] = round(performance_score, 2)
        
        # Add performance level
        if performance_score >= 80:
            performance_data["performance_level"] = "excellent"
        elif performance_score >= 60:
            performance_data["performance_level"] = "good"
        elif performance_score >= 40:
            performance_data["performance_level"] = "fair"
        else:
            performance_data["performance_level"] = "poor"
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))