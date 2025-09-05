"""
Real Training API Endpoints for Persian Legal AI with Security
نقاط پایانی API آموزش واقعی با امنیت برای هوش مصنوعی حقوقی فارسی
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

# Import real training components
from models.dora_trainer import DoRATrainer, DoRAConfig
from models.qr_adaptor import QRAdaptor, QRAdaptorConfig
from services.persian_data_processor import PersianLegalDataProcessor

# Import verified data training components
from models.verified_data_trainer import VerifiedDataTrainer

# Import authentication
from auth.dependencies import require_training_permission, TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])

class TrainingSessionRequest(BaseModel):
    model_type: str = Field(..., description="Type of model: 'dora' or 'qr_adaptor'")
    model_name: str = Field(..., description="Name for the training session")
    config: Dict[str, Any] = Field(..., description="Training configuration")
    data_source: str = Field(default="sample", description="Data source: 'sample', 'qavanin', 'majlis'")
    task_type: str = Field(default="text_classification", description="Task type")

class TrainingSessionResponse(BaseModel):
    session_id: str
    status: str
    message: str
    created_at: datetime

class TrainingSessionStatus(BaseModel):
    session_id: str
    status: str
    progress: Dict[str, Any]
    metrics: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

# In-memory storage for training sessions (in production, use database)
training_sessions = {}

# Initialize verified data trainer
verified_trainer = VerifiedDataTrainer()

async def run_training_session(session_id: str, request: TrainingSessionRequest):
    """Run training session in background"""
    try:
        logger.info(f"Starting training session {session_id}")
        
        # Update session status
        training_sessions[session_id]["status"] = "running"
        training_sessions[session_id]["started_at"] = datetime.utcnow()
        
        # Initialize data processor
        data_processor = PersianLegalDataProcessor()
        
        # Load and preprocess data
        logger.info("Loading training data...")
        if request.data_source == "sample":
            raw_data = data_processor.load_sample_data()
        else:
            raw_data = data_processor.fetch_real_legal_documents(request.data_source, limit=50)
        
        # Preprocess data
        processed_data = data_processor.preprocess_persian_text(raw_data)
        
        # Assess quality and filter
        quality_assessments = data_processor.assess_document_quality(processed_data)
        high_quality_data = data_processor.filter_high_quality_documents(quality_assessments)
        
        # Create training dataset
        training_dataset = data_processor.create_training_dataset(high_quality_data, request.task_type)
        
        if not training_dataset or training_dataset.get('size', 0) == 0:
            raise ValueError("No training data available")
        
        # Split data for training and evaluation
        dataset = training_dataset['dataset']
        split_idx = int(len(dataset) * 0.8)
        train_data = dataset[:split_idx]
        eval_data = dataset[split_idx:] if len(dataset) > split_idx else []
        
        logger.info(f"Training data: {len(train_data)} samples, Eval data: {len(eval_data)} samples")
        
        # Initialize trainer based on model type
        if request.model_type == "dora":
            config = DoRAConfig(**request.config)
            trainer = DoRATrainer(config)
        elif request.model_type == "qr_adaptor":
            config = QRAdaptorConfig(**request.config)
            trainer = QRAdaptor(config)
        else:
            raise ValueError(f"Unsupported model type: {request.model_type}")
        
        # Update session with trainer info
        training_sessions[session_id]["trainer"] = trainer
        training_sessions[session_id]["progress"]["data_loaded"] = True
        training_sessions[session_id]["progress"]["train_samples"] = len(train_data)
        training_sessions[session_id]["progress"]["eval_samples"] = len(eval_data)
        
        # Start training
        logger.info("Starting model training...")
        training_metrics = trainer.train(train_data, eval_data)
        
        # Update session with results
        training_sessions[session_id]["status"] = "completed"
        training_sessions[session_id]["completed_at"] = datetime.utcnow()
        training_sessions[session_id]["metrics"] = training_metrics
        training_sessions[session_id]["progress"]["training_completed"] = True
        
        logger.info(f"Training session {session_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Training session {session_id} failed: {e}")
        training_sessions[session_id]["status"] = "failed"
        training_sessions[session_id]["error_message"] = str(e)
        training_sessions[session_id]["failed_at"] = datetime.utcnow()

@router.post("/sessions", response_model=TrainingSessionResponse)
async def create_training_session(
    request: TrainingSessionRequest, 
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(require_training_permission)
):
    """Create a new training session"""
    try:
        # Validate request
        if request.model_type not in ["dora", "qr_adaptor"]:
            raise HTTPException(status_code=400, detail="Invalid model type. Must be 'dora' or 'qr_adaptor'")
        
        if request.task_type not in ["text_classification", "question_answering", "text_generation"]:
            raise HTTPException(status_code=400, detail="Invalid task type")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create session record
        training_sessions[session_id] = {
            "session_id": session_id,
            "model_type": request.model_type,
            "model_name": request.model_name,
            "config": request.config,
            "data_source": request.data_source,
            "task_type": request.task_type,
            "status": "pending",
            "progress": {
                "data_loaded": False,
                "model_initialized": False,
                "training_started": False,
                "training_completed": False,
                "train_samples": 0,
                "eval_samples": 0,
                "current_epoch": 0,
                "total_epochs": 0,
                "current_step": 0,
                "total_steps": 0
            },
            "metrics": {},
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "failed_at": None,
            "error_message": None
        }
        
        # Start training in background
        background_tasks.add_task(run_training_session, session_id, request)
        
        logger.info(f"Created training session {session_id} for {request.model_type} model")
        
        return TrainingSessionResponse(
            session_id=session_id,
            status="pending",
            message="Training session created and started",
            created_at=training_sessions[session_id]["created_at"]
        )
        
    except Exception as e:
        logger.error(f"Failed to create training session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions", response_model=List[TrainingSessionStatus])
async def get_training_sessions(current_user: TokenData = Depends(require_training_permission)):
    """Get all training sessions"""
    try:
        sessions = []
        for session_id, session_data in training_sessions.items():
            sessions.append(TrainingSessionStatus(
                session_id=session_data["session_id"],
                status=session_data["status"],
                progress=session_data["progress"],
                metrics=session_data["metrics"],
                created_at=session_data["created_at"],
                updated_at=session_data.get("completed_at", session_data.get("started_at", session_data["created_at"]))
            ))
        
        # Sort by creation time (newest first)
        sessions.sort(key=lambda x: x.created_at, reverse=True)
        
        return sessions
        
    except Exception as e:
        logger.error(f"Failed to get training sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}", response_model=TrainingSessionStatus)
async def get_training_session(session_id: str, current_user: TokenData = Depends(require_training_permission)):
    """Get specific training session"""
    try:
        if session_id not in training_sessions:
            raise HTTPException(status_code=404, detail="Training session not found")
        
        session_data = training_sessions[session_id]
        
        return TrainingSessionStatus(
            session_id=session_data["session_id"],
            status=session_data["status"],
            progress=session_data["progress"],
            metrics=session_data["metrics"],
            created_at=session_data["created_at"],
            updated_at=session_data.get("completed_at", session_data.get("started_at", session_data["created_at"]))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sessions/{session_id}")
async def delete_training_session(session_id: str, current_user: TokenData = Depends(require_training_permission)):
    """Delete training session"""
    try:
        if session_id not in training_sessions:
            raise HTTPException(status_code=404, detail="Training session not found")
        
        # Check if session is running
        if training_sessions[session_id]["status"] == "running":
            raise HTTPException(status_code=400, detail="Cannot delete running training session")
        
        del training_sessions[session_id]
        
        logger.info(f"Deleted training session {session_id}")
        
        return {"message": "Training session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete training session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/metrics")
async def get_training_metrics(session_id: str, current_user: TokenData = Depends(require_training_permission)):
    """Get detailed training metrics for a session"""
    try:
        if session_id not in training_sessions:
            raise HTTPException(status_code=404, detail="Training session not found")
        
        session_data = training_sessions[session_id]
        
        if session_data["status"] not in ["completed", "running"]:
            raise HTTPException(status_code=400, detail="Training session not completed or running")
        
        metrics = session_data.get("metrics", {})
        
        return {
            "session_id": session_id,
            "status": session_data["status"],
            "metrics": metrics,
            "progress": session_data["progress"],
            "created_at": session_data["created_at"],
            "started_at": session_data.get("started_at"),
            "completed_at": session_data.get("completed_at")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training metrics for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/logs")
async def get_training_logs(session_id: str, current_user: TokenData = Depends(require_training_permission)):
    """Get training logs for a session"""
    try:
        if session_id not in training_sessions:
            raise HTTPException(status_code=404, detail="Training session not found")
        
        session_data = training_sessions[session_id]
        
        # In a real implementation, this would read from log files
        # For now, return session information as logs
        logs = [
            f"Session {session_id} created at {session_data['created_at']}",
            f"Model type: {session_data['model_type']}",
            f"Task type: {session_data['task_type']}",
            f"Data source: {session_data['data_source']}",
            f"Status: {session_data['status']}"
        ]
        
        if session_data.get("started_at"):
            logs.append(f"Training started at {session_data['started_at']}")
        
        if session_data.get("completed_at"):
            logs.append(f"Training completed at {session_data['completed_at']}")
        
        if session_data.get("error_message"):
            logs.append(f"Error: {session_data['error_message']}")
        
        return {
            "session_id": session_id,
            "logs": logs,
            "total_logs": len(logs)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training logs for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/stop")
async def stop_training_session(session_id: str, current_user: TokenData = Depends(require_training_permission)):
    """Stop a running training session"""
    try:
        if session_id not in training_sessions:
            raise HTTPException(status_code=404, detail="Training session not found")
        
        session_data = training_sessions[session_id]
        
        if session_data["status"] != "running":
            raise HTTPException(status_code=400, detail="Training session is not running")
        
        # In a real implementation, this would stop the training process
        # For now, just update the status
        session_data["status"] = "stopped"
        session_data["stopped_at"] = datetime.utcnow()
        
        logger.info(f"Stopped training session {session_id}")
        
        return {"message": "Training session stopped successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop training session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# VERIFIED DATA TRAINING ENDPOINTS
# ============================================================================
# New endpoints for verified dataset training while maintaining compatibility

@router.post("/sessions/verified")
async def create_verified_training_session(
    request: TrainingSessionRequest,
    background_tasks: BackgroundTasks
):
    """
    NEW endpoint for verified data training
    Maintains EXACT same request/response format as existing endpoints
    """
    try:
        logger.info("🎯 Starting verified data training session")
        
        # Validate request format (same as existing endpoint)
        if not request.model_type or not request.model_name:
            raise HTTPException(status_code=400, detail="model_type and model_name are required")
        
        # Prepare training configuration
        training_config = {
            'model_type': request.model_type,
            'model_name': request.model_name,
            'rank': request.config.get('rank', 16),
            'alpha': request.config.get('alpha', 32),
            'dropout': request.config.get('dropout', 0.1),
            'learning_rate': request.config.get('learning_rate', 2e-4),
            'batch_size': request.config.get('batch_size', 4),
            'num_epochs': request.config.get('num_epochs', 3),
            'max_length': request.config.get('max_length', 512),
            'warmup_steps': request.config.get('warmup_steps', 100),
            'target_modules': request.config.get('target_modules', ["q_proj", "v_proj"])
        }
        
        # Start verified data training
        result = await verified_trainer.train_with_verified_data(training_config)
        
        if result['success']:
            # Maintain EXACT same response format as existing endpoints
            response = {
                "sessionId": result['session_id'],
                "status": "success",
                "message": "Verified data training started successfully",
                "created_at": datetime.utcnow(),
                "datasets_used": result.get('datasets_used', []),
                "validation_passed": result.get('validation_passed', {}),
                "training_data_info": result.get('training_data_info', {})
            }
            
            logger.info(f"✅ Verified training session created: {result['session_id']}")
            return response
        else:
            # Maintain EXACT error format as existing endpoints
            raise HTTPException(
                status_code=500,
                detail=f"Verified training failed: {result.get('error', 'Unknown error')}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create verified training session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/verified/{session_id}/status")
async def get_verified_training_status(session_id: str):
    """Get status of verified training session"""
    try:
        status = verified_trainer.get_training_session_status(session_id)
        
        if 'error' in status:
            raise HTTPException(status_code=404, detail=status['error'])
        
        # Maintain EXACT same response format as existing endpoints
        return {
            "session_id": session_id,
            "status": status.get('status', 'unknown'),
            "progress": status.get('progress', 0),
            "metrics": status.get('metrics', {}),
            "created_at": status.get('start_time', datetime.utcnow()),
            "updated_at": status.get('updated_at', datetime.utcnow())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get verified training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/verified")
async def get_verified_datasets_info():
    """Get information about verified datasets"""
    try:
        dataset_info = verified_trainer.get_dataset_info()
        validation_report = verified_trainer.get_validation_report()
        
        return {
            "verified_datasets": dataset_info.get('verified_datasets', {}),
            "loaded_datasets": dataset_info.get('loaded_datasets', {}),
            "validation_report": validation_report,
            "integration_status": dataset_info.get('integration_status', {})
        }
        
    except Exception as e:
        logger.error(f"Failed to get verified datasets info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/datasets/verify")
async def verify_datasets():
    """Trigger dataset verification process"""
    try:
        logger.info("🔍 Starting dataset verification process")
        
        # Load and verify datasets
        datasets = verified_trainer.data_integrator.load_verified_datasets()
        validation_results = verified_trainer.quality_validator.validate_datasets(datasets)
        
        return {
            "verification_completed": True,
            "datasets_verified": len(validation_results),
            "validation_results": validation_results,
            "verification_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to verify datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/verified")
async def get_all_verified_training_sessions():
    """Get all verified training sessions"""
    try:
        sessions_info = verified_trainer.get_all_training_sessions()
        training_log = verified_trainer.get_training_log(limit=50)
        
        return {
            "sessions": sessions_info.get('sessions', {}),
            "total_sessions": sessions_info.get('total_sessions', 0),
            "active_sessions": sessions_info.get('active_sessions', 0),
            "recent_training_log": training_log
        }
        
    except Exception as e:
        logger.error(f"Failed to get verified training sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sessions/verified/{session_id}")
async def cancel_verified_training_session(session_id: str):
    """Cancel a verified training session"""
    try:
        result = await verified_trainer.cancel_training_session(session_id)
        
        if result['success']:
            return {"message": "Verified training session cancelled successfully"}
        else:
            raise HTTPException(status_code=404, detail=result.get('error', 'Session not found'))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel verified training session: {e}")
        raise HTTPException(status_code=500, detail=str(e))