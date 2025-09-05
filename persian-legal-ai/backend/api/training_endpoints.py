from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Union
import logging
import subprocess
import threading
import time
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from training.train_legal_model import TrainingOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global training orchestrator instance
training_orchestrator = None

def get_training_orchestrator() -> TrainingOrchestrator:
    """Get or create training orchestrator instance."""
    global training_orchestrator
    if training_orchestrator is None:
        training_orchestrator = TrainingOrchestrator()
    return training_orchestrator

# Request/Response models
class TrainingRequest(BaseModel):
    model_type: str = "DoRA"  # or "QR-Adaptor"
    continuous: bool = False
    config_override: Optional[Dict] = None

class TrainingResponse(BaseModel):
    status: str
    message: str
    training_id: Optional[str] = None
    details: Optional[Dict] = None

class ModelInfo(BaseModel):
    model_type: str
    model_path: str
    training_time: str
    performance_metrics: Dict
    model_size: Optional[Dict] = None

# Create router
router = APIRouter(prefix="/training", tags=["training"])

@router.post("/start", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    orchestrator: TrainingOrchestrator = Depends(get_training_orchestrator)
):
    """
    Start training process.
    
    Args:
        request: Training configuration
        background_tasks: FastAPI background tasks
        orchestrator: Training orchestrator instance
    
    Returns:
        Training response with status and details
    """
    try:
        # Validate model type
        if request.model_type not in ["DoRA", "QR-Adaptor"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid model_type. Must be 'DoRA' or 'QR-Adaptor'"
            )
        
        # Update orchestrator config if provided
        if request.config_override:
            orchestrator.config.update(request.config_override)
        
        # Override model type
        orchestrator.config["model_type"] = request.model_type
        
        # Start training
        if request.continuous:
            background_tasks.add_task(
                orchestrator.start_training, 
                continuous=True
            )
            message = "Continuous training started in background"
        else:
            background_tasks.add_task(
                orchestrator.start_training, 
                continuous=False
            )
            message = "Single training run started in background"
        
        logger.info(f"Training started: {request.model_type}, continuous={request.continuous}")
        
        return TrainingResponse(
            status="started",
            message=message,
            training_id=f"training_{int(time.time())}",
            details={
                "model_type": request.model_type,
                "continuous": request.continuous,
                "config": orchestrator.config
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop", response_model=TrainingResponse)
async def stop_training(
    orchestrator: TrainingOrchestrator = Depends(get_training_orchestrator)
):
    """
    Stop ongoing training process.
    
    Returns:
        Training response with stop status
    """
    try:
        orchestrator.stop_training()
        
        logger.info("Training stopped by user request")
        
        return TrainingResponse(
            status="stopped",
            message="Training process stopped successfully",
            details={"stopped_at": time.time()}
        )
        
    except Exception as e:
        logger.error(f"Error stopping training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=Dict)
async def get_training_status(
    orchestrator: TrainingOrchestrator = Depends(get_training_orchestrator)
):
    """
    Get current training status and system information.
    
    Returns:
        Current training status and system metrics
    """
    try:
        status = orchestrator.get_status()
        
        return {
            "status": "success",
            "data": status,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=List[ModelInfo])
async def list_trained_models(
    orchestrator: TrainingOrchestrator = Depends(get_training_orchestrator)
):
    """
    List all trained models.
    
    Returns:
        List of trained models with metadata
    """
    try:
        models = []
        output_dir = Path(orchestrator.config["output_dir"])
        
        if output_dir.exists():
            for model_dir in output_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "training_results.json").exists():
                    try:
                        with open(model_dir / "training_results.json", "r") as f:
                            results = json.load(f)
                        
                        model_info = ModelInfo(
                            model_type=orchestrator.config["model_type"],
                            model_path=str(model_dir),
                            training_time=results.get("train_runtime", "unknown"),
                            performance_metrics=results.get("validation_results", {}),
                            model_size=results.get("model_size", None)
                        )
                        models.append(model_info)
                        
                    except Exception as e:
                        logger.warning(f"Error reading model info from {model_dir}: {e}")
        
        return models
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}/info", response_model=ModelInfo)
async def get_model_info(
    model_id: str,
    orchestrator: TrainingOrchestrator = Depends(get_training_orchestrator)
):
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: Model identifier (directory name)
    
    Returns:
        Detailed model information
    """
    try:
        output_dir = Path(orchestrator.config["output_dir"])
        model_dir = output_dir / model_id
        
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        results_file = model_dir / "training_results.json"
        if not results_file.exists():
            raise HTTPException(status_code=404, detail="Model results not found")
        
        with open(results_file, "r") as f:
            results = json.load(f)
        
        model_info = ModelInfo(
            model_type=orchestrator.config["model_type"],
            model_path=str(model_dir),
            training_time=results.get("train_runtime", "unknown"),
            performance_metrics=results.get("validation_results", {}),
            model_size=results.get("model_size", None)
        )
        
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_id}/predict")
async def predict_with_model(
    model_id: str,
    texts: List[str],
    orchestrator: TrainingOrchestrator = Depends(get_training_orchestrator)
):
    """
    Make predictions using a specific trained model.
    
    Args:
        model_id: Model identifier
        texts: List of texts to predict on
    
    Returns:
        Predictions for the input texts
    """
    try:
        output_dir = Path(orchestrator.config["output_dir"])
        model_dir = output_dir / model_id
        
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Load the appropriate trainer based on model type
        model_type = orchestrator.config["model_type"]
        
        if model_type == "DoRA":
            from models.dora_trainer import DoRATrainer
            trainer = DoRATrainer()
        elif model_type == "QR-Adaptor":
            from models.qr_adaptor import QRAdaptorTrainer
            trainer = QRAdaptorTrainer()
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
        
        # Load the model
        trainer.load_model(str(model_dir))
        
        # Make predictions
        predictions = trainer.predict(texts)
        
        return {
            "status": "success",
            "model_id": model_id,
            "predictions": predictions,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config", response_model=Dict)
async def get_training_config(
    orchestrator: TrainingOrchestrator = Depends(get_training_orchestrator)
):
    """
    Get current training configuration.
    
    Returns:
        Current training configuration
    """
    try:
        return {
            "status": "success",
            "config": orchestrator.config,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/config", response_model=Dict)
async def update_training_config(
    config_update: Dict,
    orchestrator: TrainingOrchestrator = Depends(get_training_orchestrator)
):
    """
    Update training configuration.
    
    Args:
        config_update: Configuration updates
    
    Returns:
        Updated configuration
    """
    try:
        # Validate configuration
        if "model_type" in config_update:
            if config_update["model_type"] not in ["DoRA", "QR-Adaptor"]:
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid model_type. Must be 'DoRA' or 'QR-Adaptor'"
                )
        
        # Update configuration
        orchestrator.config.update(config_update)
        
        logger.info(f"Configuration updated: {config_update}")
        
        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "config": orchestrator.config,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs")
async def get_training_logs(
    lines: int = 100,
    orchestrator: TrainingOrchestrator = Depends(get_training_orchestrator)
):
    """
    Get recent training logs.
    
    Args:
        lines: Number of log lines to return
    
    Returns:
        Recent training logs
    """
    try:
        log_file = Path("training.log")
        
        if not log_file.exists():
            return {
                "status": "success",
                "logs": [],
                "message": "No log file found"
            }
        
        # Read last N lines
        with open(log_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "status": "success",
            "logs": [line.strip() for line in recent_lines],
            "total_lines": len(all_lines),
            "returned_lines": len(recent_lines)
        }
        
    except Exception as e:
        logger.error(f"Error getting logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))