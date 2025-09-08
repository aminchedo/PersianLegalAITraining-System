from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from models.system_metrics import SystemMetrics
from config.database import get_db
from pydantic import BaseModel
from datetime import datetime, timedelta
import psutil

router = APIRouter(prefix='/api/real/monitoring', tags=['monitoring'])

class SystemMetricsResponse(BaseModel):
    id: int
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    disk_usage: float
    network_in: float
    network_out: float
    temperature: float
    power_consumption: float
    active_connections: int
    queue_size: int
    error_count: int
    is_healthy: bool

    class Config:
        from_attributes = True

class CurrentSystemMetrics(BaseModel):
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    disk_usage: float
    network_in: float
    network_out: float
    temperature: float
    power_consumption: float
    active_connections: int
    queue_size: int
    error_count: int
    is_healthy: bool
    timestamp: datetime

@router.get('/system-metrics', response_model=CurrentSystemMetrics)
async def get_current_system_metrics():
    """Get current real-time system metrics"""
    try:
        # Get real system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # Get network stats
        network = psutil.net_io_counters()
        network_in = network.bytes_recv / 1024 / 1024  # MB
        network_out = network.bytes_sent / 1024 / 1024  # MB
        
        # Get temperature (if available)
        try:
            temps = psutil.sensors_temperatures()
            temperature = temps.get('coretemp', [{}])[0].get('current', 45.0)
        except:
            temperature = 45.0
        
        # Estimate power consumption based on CPU usage
        power_consumption = 150 + (cpu_usage * 2)
        
        # Check system health
        is_healthy = cpu_usage < 90 and memory_usage < 90 and disk_usage < 90
        
        return CurrentSystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=0.0,  # Would need GPU monitoring library
            disk_usage=disk_usage,
            network_in=network_in,
            network_out=network_out,
            temperature=temperature,
            power_consumption=power_consumption,
            active_connections=0,  # Would need to track active connections
            queue_size=0,  # Would need to track training queue
            error_count=0,  # Would need to track errors
            is_healthy=is_healthy,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")

@router.get('/system-metrics/history', response_model=List[SystemMetricsResponse])
async def get_system_metrics_history(
    hours: int = Query(24, ge=1, le=168),  # 1 hour to 1 week
    db: Session = Depends(get_db)
):
    """Get historical system metrics"""
    since = datetime.utcnow() - timedelta(hours=hours)
    metrics = db.query(SystemMetrics).filter(
        SystemMetrics.timestamp >= since
    ).order_by(SystemMetrics.timestamp.desc()).all()
    
    return metrics

@router.post('/system-metrics')
async def store_system_metrics(metrics_data: CurrentSystemMetrics, db: Session = Depends(get_db)):
    """Store current system metrics in database"""
    new_metrics = SystemMetrics(**metrics_data.dict())
    db.add(new_metrics)
    db.commit()
    db.refresh(new_metrics)
    return {"message": "Metrics stored successfully", "id": new_metrics.id}

@router.get('/health')
async def health_check():
    """System health check endpoint"""
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "checks": {
                "cpu": {
                    "usage": cpu_usage,
                    "status": "ok" if cpu_usage < 90 else "warning"
                },
                "memory": {
                    "usage": memory.percent,
                    "status": "ok" if memory.percent < 90 else "warning"
                },
                "disk": {
                    "usage": (disk.used / disk.total) * 100,
                    "status": "ok" if (disk.used / disk.total) * 100 < 90 else "warning"
                }
            }
        }
        
        # Overall status
        if any(check["status"] == "warning" for check in health_status["checks"].values()):
            health_status["status"] = "warning"
        
        return health_status
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.utcnow(),
            "error": str(e)
        }