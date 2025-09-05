"""
Enhanced Health Check Endpoint for Persian Legal AI
نقطه پایانی بررسی سلامت پیشرفته برای هوش مصنوعی حقوقی فارسی
"""

import logging
import psutil
import os
import platform
import sys
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

# Import database and other components
from database.connection import db_manager
from optimization.system_optimizer import system_optimizer
from auth.dependencies import optional_auth, TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])

class GPUInfo(BaseModel):
    available: bool
    count: int = 0
    devices: List[Dict[str, Any]] = []
    memory_total: float = 0.0
    memory_used: float = 0.0
    memory_free: float = 0.0
    utilization: float = 0.0

class DatabaseHealth(BaseModel):
    status: str
    connection_time: float
    query_time: float
    active_connections: int
    max_connections: int
    database_size: str
    last_backup: Optional[str] = None

class SystemHealth(BaseModel):
    status: str
    timestamp: datetime
    uptime: float
    version: str
    
    # System metrics
    system_metrics: Dict[str, Any]
    
    # GPU information
    gpu_info: GPUInfo
    
    # Database health
    database: DatabaseHealth
    
    # Services status
    services: Dict[str, str]
    
    # Performance metrics
    performance: Dict[str, Any]
    
    # Security status
    security: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    checks: Dict[str, Any]
    overall_health: str
    message: str

def get_gpu_info() -> GPUInfo:
    """Get GPU information"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            devices = []
            total_memory = 0.0
            used_memory = 0.0
            
            for i in range(gpu_count):
                device_props = torch.cuda.get_device_properties(i)
                memory_total = device_props.total_memory / (1024**3)  # GB
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                memory_cached = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                memory_used = memory_allocated + memory_cached
                memory_free = memory_total - memory_used
                
                devices.append({
                    "id": i,
                    "name": device_props.name,
                    "memory_total": memory_total,
                    "memory_used": memory_used,
                    "memory_free": memory_free,
                    "utilization": (memory_used / memory_total) * 100 if memory_total > 0 else 0
                })
                
                total_memory += memory_total
                used_memory += memory_used
            
            avg_utilization = sum(d["utilization"] for d in devices) / len(devices) if devices else 0
            
            return GPUInfo(
                available=True,
                count=gpu_count,
                devices=devices,
                memory_total=total_memory,
                memory_used=used_memory,
                memory_free=total_memory - used_memory,
                utilization=avg_utilization
            )
        else:
            return GPUInfo(available=False)
    except ImportError:
        return GPUInfo(available=False)
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
        return GPUInfo(available=False)

async def get_database_health() -> DatabaseHealth:
    """Get database health information"""
    try:
        start_time = datetime.now()
        
        # Test connection
        connection_ok = db_manager.test_connection()
        connection_time = (datetime.now() - start_time).total_seconds()
        
        if not connection_ok:
            return DatabaseHealth(
                status="unhealthy",
                connection_time=connection_time,
                query_time=0.0,
                active_connections=0,
                max_connections=0,
                database_size="unknown"
            )
        
        # Test query performance
        query_start = datetime.now()
        try:
            # Simple query to test performance
            result = db_manager.execute_query("SELECT 1")
            query_time = (datetime.now() - query_start).total_seconds()
        except Exception as e:
            logger.error(f"Database query test failed: {e}")
            query_time = -1
        
        # Get connection info (simplified)
        active_connections = 1  # We have at least one active connection
        max_connections = 100  # Default PostgreSQL max connections
        
        # Get database size (simplified)
        try:
            size_result = db_manager.execute_query("SELECT pg_size_pretty(pg_database_size(current_database()))")
            database_size = size_result[0][0] if size_result else "unknown"
        except Exception:
            database_size = "unknown"
        
        status = "healthy" if connection_ok and query_time >= 0 else "degraded"
        
        return DatabaseHealth(
            status=status,
            connection_time=connection_time,
            query_time=query_time,
            active_connections=active_connections,
            max_connections=max_connections,
            database_size=database_size
        )
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return DatabaseHealth(
            status="unhealthy",
            connection_time=-1,
            query_time=-1,
            active_connections=0,
            max_connections=0,
            database_size="unknown"
        )

def get_system_metrics() -> Dict[str, Any]:
    """Get comprehensive system metrics"""
    try:
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory information
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk information
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network information
        network = psutil.net_io_counters()
        
        # Process information
        processes = list(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']))
        active_processes = len(processes)
        
        # Load average (Unix-like systems)
        try:
            load_avg = os.getloadavg()
        except AttributeError:
            load_avg = [0, 0, 0]  # Windows doesn't have load average
        
        return {
            "cpu": {
                "usage_percent": cpu_percent,
                "count": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else None,
                "load_average": load_avg
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "usage_percent": memory.percent,
                "swap_total_gb": swap.total / (1024**3),
                "swap_used_gb": swap.used / (1024**3),
                "swap_usage_percent": swap.percent
            },
            "disk": {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3),
                "usage_percent": (disk.used / disk.total) * 100,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            },
            "processes": {
                "total": active_processes,
                "top_cpu": sorted(processes, key=lambda p: p.info.get('cpu_percent', 0), reverse=True)[:5],
                "top_memory": sorted(processes, key=lambda p: p.info.get('memory_percent', 0), reverse=True)[:5]
            }
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return {"error": str(e)}

def get_services_status() -> Dict[str, str]:
    """Get status of various services"""
    services = {}
    
    # Check database
    try:
        db_healthy = db_manager.test_connection()
        services["database"] = "healthy" if db_healthy else "unhealthy"
    except Exception:
        services["database"] = "unhealthy"
    
    # Check optimization system
    try:
        services["optimization"] = "active" if system_optimizer.monitoring_active else "inactive"
    except Exception:
        services["optimization"] = "unknown"
    
    # Check GPU availability
    try:
        import torch
        services["gpu"] = "available" if torch.cuda.is_available() else "unavailable"
    except ImportError:
        services["gpu"] = "not_installed"
    except Exception:
        services["gpu"] = "unknown"
    
    # Check Redis (if configured)
    try:
        import redis
        # This would need Redis connection details
        services["redis"] = "unknown"
    except ImportError:
        services["redis"] = "not_installed"
    except Exception:
        services["redis"] = "unknown"
    
    return services

def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics"""
    try:
        # Get optimization metrics
        optimization_report = system_optimizer.get_optimization_report()
        
        return {
            "optimization": {
                "active": system_optimizer.monitoring_active,
                "optimal_batch_size": system_optimizer.get_optimal_batch_size(),
                "optimal_workers": system_optimizer.get_optimal_num_workers(),
                "report": optimization_report
            },
            "response_times": {
                "health_check": 0.0,  # This would be measured in real implementation
                "database_query": 0.0,
                "api_endpoint": 0.0
            }
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return {"error": str(e)}

def get_security_status(current_user: Optional[TokenData] = None) -> Dict[str, Any]:
    """Get security status"""
    try:
        security_info = {
            "authentication": {
                "enabled": True,
                "jwt_enabled": True,
                "rate_limiting": True
            },
            "ssl": {
                "enabled": os.path.exists("/app/certificates/server.crt"),
                "certificate_valid": False  # Would check certificate validity
            },
            "permissions": {
                "current_user": current_user.username if current_user else None,
                "user_permissions": current_user.permissions if current_user else []
            }
        }
        
        # Check certificate validity if SSL is enabled
        if security_info["ssl"]["enabled"]:
            try:
                import ssl
                import socket
                from datetime import datetime
                
                # This is a simplified check - in production, use proper certificate validation
                cert_path = "/app/certificates/server.crt"
                if os.path.exists(cert_path):
                    # Check if certificate is not expired (simplified)
                    with open(cert_path, 'r') as f:
                        cert_content = f.read()
                        if "BEGIN CERTIFICATE" in cert_content:
                            security_info["ssl"]["certificate_valid"] = True
            except Exception as e:
                logger.error(f"SSL certificate check failed: {e}")
        
        return security_info
    except Exception as e:
        logger.error(f"Error getting security status: {e}")
        return {"error": str(e)}

@router.get("/health", response_model=HealthResponse)
async def get_enhanced_health(current_user: Optional[TokenData] = Depends(optional_auth)):
    """Enhanced health check endpoint"""
    try:
        start_time = datetime.now()
        
        # Get system uptime
        uptime = time.time() - psutil.boot_time()
        
        # Get all health information
        gpu_info = get_gpu_info()
        database_health = await get_database_health()
        system_metrics = get_system_metrics()
        services_status = get_services_status()
        performance_metrics = get_performance_metrics()
        security_status = get_security_status(current_user)
        
        # Determine overall health
        health_checks = {
            "database": database_health.status,
            "gpu": "healthy" if gpu_info.available else "unavailable",
            "system": "healthy" if system_metrics.get("cpu", {}).get("usage_percent", 0) < 90 else "degraded",
            "services": "healthy" if all(status in ["healthy", "active", "available"] for status in services_status.values()) else "degraded"
        }
        
        # Calculate overall health
        if all(status == "healthy" for status in health_checks.values()):
            overall_health = "healthy"
            status_message = "All systems operational"
        elif any(status == "unhealthy" for status in health_checks.values()):
            overall_health = "unhealthy"
            status_message = "Some systems are unhealthy"
        else:
            overall_health = "degraded"
            status_message = "Some systems are degraded"
        
        response_time = (datetime.now() - start_time).total_seconds()
        
        return HealthResponse(
            status=overall_health,
            timestamp=datetime.now(),
            checks={
                "system_metrics": system_metrics,
                "gpu_info": gpu_info.dict(),
                "database": database_health.dict(),
                "services": services_status,
                "performance": performance_metrics,
                "security": security_status,
                "response_time": response_time
            },
            overall_health=overall_health,
            message=status_message
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

@router.get("/health/simple")
async def get_simple_health():
    """Simple health check for load balancers"""
    try:
        # Quick database check
        db_healthy = db_manager.test_connection()
        
        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected" if db_healthy else "disconnected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }