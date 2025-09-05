"""
Training API Endpoints for Persian Legal AI
نقاط پایانی API آموزش برای هوش مصنوعی حقوقی فارسی
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import json

from ..services.training_service import training_service, TrainingRequest, TrainingResponse
from ..database.connection import db_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected for session: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")
                self.disconnect(session_id)

manager = ConnectionManager()

# Pydantic models for request/response
class TrainingRequestModel(BaseModel):
    model_name: str = Field(..., description="Name of the model to train")
    model_type: str = Field(..., description="Type of model (dora, qr_adaptor, hybrid)")
    base_model: str = Field(..., description="Base model to fine-tune")
    dataset_sources: List[str] = Field(..., description="List of data sources")
    training_config: Dict[str, Any] = Field(..., description="Training configuration")
    model_config: Dict[str, Any] = Field(default_factory=dict, description="Model configuration")
    priority: int = Field(default=1, ge=1, le=3, description="Training priority (1=high, 2=medium, 3=low)")

class TrainingResponseModel(BaseModel):
    session_id: str
    status: str
    message: str
    estimated_duration: Optional[str] = None

class TrainingStatusModel(BaseModel):
    session_id: str
    status: str
    progress: Dict[str, Any]
    metrics: Dict[str, Any]
    system_info: Dict[str, Any]

class TrainingControlModel(BaseModel):
    action: str = Field(..., description="Action to perform (pause, resume, cancel)")

@router.post("/sessions", response_model=TrainingResponseModel)
async def create_training_session(request: TrainingRequestModel, background_tasks: BackgroundTasks):
    """Create a new training session"""
    try:
        # Convert to training request
        training_request = TrainingRequest(
            model_name=request.model_name,
            model_type=request.model_type,
            base_model=request.base_model,
            dataset_sources=request.dataset_sources,
            training_config=request.training_config,
            model_config=request.model_config,
            priority=request.priority
        )
        
        # Submit training request
        response = await training_service.submit_training_request(training_request)
        
        # Start background monitoring if training started
        if response.status == "started":
            background_tasks.add_task(monitor_training_session, response.session_id)
        
        return TrainingResponseModel(
            session_id=response.session_id,
            status=response.status,
            message=response.message,
            estimated_duration=response.estimated_duration
        )
        
    except Exception as e:
        logger.error(f"Failed to create training session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/status", response_model=TrainingStatusModel)
async def get_training_status(session_id: str):
    """Get training session status"""
    try:
        # Get session status
        status = await training_service.get_training_status(session_id)
        if not status:
            raise HTTPException(status_code=404, detail="Training session not found")
        
        # Get recent metrics
        metrics = await training_service.get_training_metrics(session_id, limit=10)
        
        # Get system performance
        system_info = await training_service.get_system_performance()
        
        # Calculate progress
        progress = {
            'current_epoch': status.get('current_epoch', 0),
            'total_epochs': status.get('total_epochs', 0),
            'current_step': status.get('current_step', 0),
            'total_steps': status.get('total_steps', 0),
            'progress_percentage': 0
        }
        
        if progress['total_epochs'] > 0:
            progress['progress_percentage'] = (progress['current_epoch'] / progress['total_epochs']) * 100
        
        return TrainingStatusModel(
            session_id=session_id,
            status=status.get('status', 'unknown'),
            progress=progress,
            metrics={
                'current_loss': status.get('current_loss'),
                'best_loss': status.get('best_loss'),
                'current_accuracy': status.get('current_accuracy'),
                'best_accuracy': status.get('best_accuracy'),
                'learning_rate': status.get('learning_rate'),
                'recent_metrics': metrics
            },
            system_info=system_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/metrics")
async def get_training_metrics(session_id: str, limit: int = 100):
    """Get training metrics for a session"""
    try:
        metrics = await training_service.get_training_metrics(session_id, limit)
        return {"session_id": session_id, "metrics": metrics}
        
    except Exception as e:
        logger.error(f"Failed to get training metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/control")
async def control_training_session(session_id: str, control: TrainingControlModel):
    """Control training session (pause, resume, cancel)"""
    try:
        action = control.action.lower()
        
        if action == "pause":
            success = await training_service.pause_training(session_id)
            message = "Training paused successfully" if success else "Failed to pause training"
        elif action == "resume":
            success = await training_service.resume_training(session_id)
            message = "Training resumed successfully" if success else "Failed to resume training"
        elif action == "cancel":
            success = await training_service.cancel_training(session_id)
            message = "Training cancelled successfully" if success else "Failed to cancel training"
        else:
            raise HTTPException(status_code=400, detail="Invalid action. Use 'pause', 'resume', or 'cancel'")
        
        return {"session_id": session_id, "action": action, "success": success, "message": message}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to control training session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions")
async def list_training_sessions(status: Optional[str] = None):
    """List all training sessions"""
    try:
        sessions = await training_service.list_training_sessions(status)
        return {"sessions": sessions, "total": len(sessions)}
        
    except Exception as e:
        logger.error(f"Failed to list training sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/available")
async def get_available_models():
    """Get list of available models for training"""
    try:
        models = await training_service.get_available_models()
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_name}/recommendations")
async def get_training_recommendations(model_name: str, task_type: str = "text_generation"):
    """Get training recommendations for a model"""
    try:
        recommendations = await training_service.get_training_recommendations(model_name, task_type)
        return {
            "model_name": model_name,
            "task_type": task_type,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Failed to get training recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/prepare")
async def prepare_training_data(
    sources: List[str],
    task_type: str = "text_generation",
    max_documents: int = 1000
):
    """Prepare training data from sources"""
    try:
        data_info = await training_service.prepare_training_data(sources, task_type, max_documents)
        return data_info
        
    except Exception as e:
        logger.error(f"Failed to prepare training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/performance")
async def get_system_performance():
    """Get system performance metrics"""
    try:
        performance = await training_service.get_system_performance()
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get system performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/system/optimize")
async def optimize_system():
    """Optimize system for training"""
    try:
        optimization_results = await training_service.optimize_system_for_training()
        return {
            "message": "System optimization completed",
            "results": optimization_results
        }
        
    except Exception as e:
        logger.error(f"Failed to optimize system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/training/{session_id}")
async def websocket_training_monitor(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time training monitoring"""
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Send periodic updates
            status = await training_service.get_training_status(session_id)
            if status:
                await manager.send_message(session_id, {
                    "type": "status_update",
                    "session_id": session_id,
                    "status": status.get('status'),
                    "progress": {
                        "current_epoch": status.get('current_epoch', 0),
                        "total_epochs": status.get('total_epochs', 0),
                        "current_step": status.get('current_step', 0),
                        "total_steps": status.get('total_steps', 0)
                    },
                    "metrics": {
                        "current_loss": status.get('current_loss'),
                        "best_loss": status.get('best_loss'),
                        "learning_rate": status.get('learning_rate')
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Wait before next update
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        manager.disconnect(session_id)

async def monitor_training_session(session_id: str):
    """Background task to monitor training session"""
    try:
        while True:
            # Check if session is still active
            status = await training_service.get_training_status(session_id)
            if not status or status.get('status') in ['completed', 'failed', 'cancelled']:
                break
            
            # Send WebSocket update if connected
            if session_id in manager.active_connections:
                await manager.send_message(session_id, {
                    "type": "training_update",
                    "session_id": session_id,
                    "status": status.get('status'),
                    "metrics": {
                        "current_loss": status.get('current_loss'),
                        "current_epoch": status.get('current_epoch'),
                        "current_step": status.get('current_step')
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Wait before next check
            await asyncio.sleep(10)
            
    except Exception as e:
        logger.error(f"Error monitoring training session {session_id}: {e}")
    finally:
        # Cleanup WebSocket connection
        manager.disconnect(session_id)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db_healthy = db_manager.test_connection()
        
        # Check system performance
        system_perf = await training_service.get_system_performance()
        
        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "database": "connected" if db_healthy else "disconnected",
            "active_sessions": system_perf.get('training_queue', {}).get('active_sessions', 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )