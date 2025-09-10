"""
Real System API Endpoints for Persian Legal AI
نقاط پایانی API سیستم واقعی برای هوش مصنوعی حقوقی فارسی
"""

import logging
import psutil
import os
import platform
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])

class SystemInfoModel(BaseModel):
    cpu_cores: int
    memory_gb: float
    disk_space_gb: float
    os_info: str
    python_version: str
    torch_available: bool
    cuda_available: bool

class SystemMetricsModel(BaseModel):
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_processes: int
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None

@router.get("/health")
async def get_system_health():
    """Get real system health status"""
    try:
        # Get real system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # Get GPU info if available
        gpu_info = {}
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = {
                    "gpu_available": True,
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    "gpu_memory_used": torch.cuda.memory_allocated(0) / (1024**3),
                    "gpu_memory_free": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
                }
            else:
                gpu_info = {"gpu_available": False}
        except ImportError:
            gpu_info = {"gpu_available": False, "torch_not_available": True}
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "system_metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": (disk.used / disk.total) * 100,
                "disk_free_gb": disk.free / (1024**3),
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
                "active_processes": len(psutil.pids())
            },
            "gpu_info": gpu_info,
            "platform_info": {
                "os": platform.system(),
                "os_version": platform.release(),
                "python_version": sys.version,
                "architecture": platform.machine()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info")
async def get_system_info():
    """Get real system information"""
    try:
        # Get system information
        cpu_count = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check PyTorch availability
        torch_available = False
        cuda_available = False
        try:
            import torch
            torch_available = True
            cuda_available = torch.cuda.is_available()
        except ImportError:
            pass
        
        return SystemInfoModel(
            cpu_cores=cpu_count,
            memory_gb=memory.total / (1024**3),
            disk_space_gb=disk.free / (1024**3),
            os_info=f"{platform.system()} {platform.release()}",
            python_version=sys.version,
            torch_available=torch_available,
            cuda_available=cuda_available
        ).dict()
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_system_metrics():
    """Get current real system metrics"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # Get GPU metrics if available
        gpu_usage = None
        gpu_memory = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(0) / (1024**3)
                # GPU usage is not directly available in PyTorch, would need nvidia-ml-py
        except ImportError:
            pass
        
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
            active_processes=len(psutil.pids()),
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory
        ).dict()
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_system_performance():
    """Get comprehensive real system performance"""
    try:
        # Get system metrics
        system_metrics = await get_system_metrics()
        
        # Get process-specific metrics
        current_process = psutil.Process()
        process_metrics = {
            "cpu_percent": current_process.cpu_percent(),
            "memory_percent": current_process.memory_percent(),
            "memory_rss_mb": current_process.memory_info().rss / (1024**2),
            "memory_vms_mb": current_process.memory_info().vms / (1024**2),
            "num_threads": current_process.num_threads(),
            "create_time": datetime.fromtimestamp(current_process.create_time()).isoformat()
        }
        
        # Get network performance
        network_metrics = {
            "connections": len(current_process.connections()),
            "open_files": len(current_process.open_files())
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_metrics": system_metrics,
            "process_metrics": process_metrics,
            "network_metrics": network_metrics,
            "performance_score": _calculate_performance_score(system_metrics, process_metrics)
        }
        
    except Exception as e:
        logger.error(f"Failed to get system performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _calculate_performance_score(system_metrics: Dict, process_metrics: Dict) -> float:
    """Calculate overall performance score"""
    try:
        # CPU score (lower is better)
        cpu_score = max(0, 100 - system_metrics.get('cpu_usage', 0))
        
        # Memory score (lower is better)
        memory_score = max(0, 100 - system_metrics.get('memory_usage', 0))
        
        # Disk score (lower is better)
        disk_score = max(0, 100 - system_metrics.get('disk_usage', 0))
        
        # Process efficiency score
        process_cpu = process_metrics.get('cpu_percent', 0)
        process_memory = process_metrics.get('memory_percent', 0)
        process_score = max(0, 100 - (process_cpu + process_memory) / 2)
        
        # Overall score (weighted average)
        overall_score = (cpu_score * 0.3 + memory_score * 0.3 + 
                        disk_score * 0.2 + process_score * 0.2)
        
        return round(overall_score, 2)
        
    except Exception as e:
        logger.error(f"Failed to calculate performance score: {e}")
        return 0.0

@router.get("/resources")
async def get_system_resources():
    """Get detailed system resource information"""
    try:
        # CPU information
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "cpu_percent_per_core": psutil.cpu_percent(percpu=True, interval=1)
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        memory_info = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent,
            "swap_total_gb": swap.total / (1024**3),
            "swap_used_gb": swap.used / (1024**3),
            "swap_percent": swap.percent
        }
        
        # Disk information
        disk_info = {}
        for partition in psutil.disk_partitions():
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                disk_info[partition.mountpoint] = {
                    "device": partition.device,
                    "fstype": partition.fstype,
                    "total_gb": partition_usage.total / (1024**3),
                    "used_gb": partition_usage.used / (1024**3),
                    "free_gb": partition_usage.free / (1024**3),
                    "percent": (partition_usage.used / partition_usage.total) * 100
                }
            except PermissionError:
                continue
        
        # Network information
        network = psutil.net_io_counters()
        network_info = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv,
            "errin": network.errin,
            "errout": network.errout,
            "dropin": network.dropin,
            "dropout": network.dropout
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_info": cpu_info,
            "memory_info": memory_info,
            "disk_info": disk_info,
            "network_info": network_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get system resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# DEPLOYMENT ENDPOINTS - Integrated with existing system architecture
# ============================================================================

@router.get("/deployment/health")
async def get_deployment_health():
    """Get deployment health status - extends existing health endpoint"""
    try:
        from ..services.deployment_service import deployment_service
        return await deployment_service.check_deployment_health()
    except Exception as e:
        logger.error(f"Failed to get deployment health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/deployment/status")
async def get_deployment_status():
    """Get comprehensive deployment status"""
    try:
        from ..services.deployment_service import deployment_service
        status = await deployment_service.get_deployment_status()
        return status.dict()
    except Exception as e:
        logger.error(f"Failed to get deployment status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/deployment/validate")
async def validate_deployment_configuration():
    """Validate deployment configuration files"""
    try:
        from ..services.deployment_service import deployment_service
        return await deployment_service.validate_deployment_configuration()
    except Exception as e:
        logger.error(f"Failed to validate deployment configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/deployment/recommendations")
async def get_deployment_recommendations():
    """Get detailed deployment recommendations"""
    try:
        from ..services.deployment_service import deployment_service
        return await deployment_service.get_deployment_recommendations_detailed()
    except Exception as e:
        logger.error(f"Failed to get deployment recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))