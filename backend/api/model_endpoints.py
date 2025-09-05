"""
Model API Endpoints for Persian Legal AI
نقاط پایانی API مدل برای هوش مصنوعی حقوقی فارسی
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
import os

from ..database.connection import db_manager
from ..database.models import ModelRegistry, TrainingSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])

class ModelInfoModel(BaseModel):
    name: str
    model_type: str
    base_model: str
    description: Optional[str] = None
    version: str = "1.0"
    is_active: bool = True
    is_public: bool = False

class ModelEvaluationModel(BaseModel):
    model_id: str
    test_dataset: str
    evaluation_metrics: Dict[str, float]

@router.get("/")
async def list_models():
    """List all registered models"""
    try:
        with db_manager.get_session_context() as session:
            models = session.query(ModelRegistry).filter_by(is_active=True).all()
            
            return {
                "models": [model.to_dict() for model in models],
                "total": len(models)
            }
            
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_id}")
async def get_model_details(model_id: str):
    """Get detailed information about a specific model"""
    try:
        with db_manager.get_session_context() as session:
            model = session.query(ModelRegistry).filter_by(id=model_id).first()
            
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Get training session details
            training_session = None
            if model.training_session_id:
                training_session = session.query(TrainingSession).filter_by(
                    id=model.training_session_id
                ).first()
            
            return {
                "model": model.to_dict(),
                "training_session": training_session.to_dict() if training_session else None
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/register")
async def register_model(model_info: ModelInfoModel):
    """Register a new model"""
    try:
        with db_manager.get_session_context() as session:
            # Check if model name already exists
            existing_model = session.query(ModelRegistry).filter_by(name=model_info.name).first()
            if existing_model:
                raise HTTPException(status_code=400, detail="Model name already exists")
            
            # Create new model registry entry
            model = ModelRegistry(
                name=model_info.name,
                model_type=model_info.model_type,
                base_model=model_info.base_model,
                description=model_info.description,
                version=model_info.version,
                is_active=model_info.is_active,
                is_public=model_info.is_public,
                model_path=f"models/{model_info.name}/model.pt"  # Default path
            )
            
            session.add(model)
            session.commit()
            
            return {
                "message": "Model registered successfully",
                "model_id": model.id,
                "model": model.to_dict()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{model_id}")
async def update_model(model_id: str, model_info: ModelInfoModel):
    """Update model information"""
    try:
        with db_manager.get_session_context() as session:
            model = session.query(ModelRegistry).filter_by(id=model_id).first()
            
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Update model information
            model.name = model_info.name
            model.model_type = model_info.model_type
            model.base_model = model_info.base_model
            model.description = model_info.description
            model.version = model_info.version
            model.is_active = model_info.is_active
            model.is_public = model_info.is_public
            
            session.commit()
            
            return {
                "message": "Model updated successfully",
                "model": model.to_dict()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete a model"""
    try:
        with db_manager.get_session_context() as session:
            model = session.query(ModelRegistry).filter_by(id=model_id).first()
            
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Soft delete by setting is_active to False
            model.is_active = False
            session.commit()
            
            return {"message": "Model deleted successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_id}/evaluate")
async def evaluate_model(model_id: str, evaluation: ModelEvaluationModel):
    """Evaluate a model"""
    try:
        with db_manager.get_session_context() as session:
            model = session.query(ModelRegistry).filter_by(id=model_id).first()
            
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Update evaluation metrics
            model.evaluation_metrics = evaluation.evaluation_metrics
            session.commit()
            
            return {
                "message": "Model evaluation completed",
                "model_id": model_id,
                "evaluation_metrics": evaluation.evaluation_metrics
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_id}/download")
async def download_model(model_id: str):
    """Download model files"""
    try:
        with db_manager.get_session_context() as session:
            model = session.query(ModelRegistry).filter_by(id=model_id).first()
            
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Check if model files exist
            if not os.path.exists(model.model_path):
                raise HTTPException(status_code=404, detail="Model files not found")
            
            # Return download information
            return {
                "model_id": model_id,
                "model_name": model.name,
                "download_url": f"/api/models/{model_id}/files",
                "file_size_mb": os.path.getsize(model.model_path) / (1024 * 1024) if os.path.exists(model.model_path) else 0
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get download info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_id}/upload")
async def upload_model_files(model_id: str, model_file: UploadFile = File(...)):
    """Upload model files"""
    try:
        with db_manager.get_session_context() as session:
            model = session.query(ModelRegistry).filter_by(id=model_id).first()
            
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Create model directory
            model_dir = f"models/{model.name}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Save uploaded file
            file_path = os.path.join(model_dir, model_file.filename)
            with open(file_path, "wb") as buffer:
                content = await model_file.read()
                buffer.write(content)
            
            # Update model path
            model.model_path = file_path
            session.commit()
            
            return {
                "message": "Model files uploaded successfully",
                "model_id": model_id,
                "file_path": file_path,
                "file_size_mb": len(content) / (1024 * 1024)
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload model files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_id}/checkpoints")
async def list_model_checkpoints(model_id: str):
    """List checkpoints for a model"""
    try:
        with db_manager.get_session_context() as session:
            model = session.query(ModelRegistry).filter_by(id=model_id).first()
            
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Get checkpoints from training session
            checkpoints = []
            if model.training_session_id:
                from ..database.models import ModelCheckpoint
                checkpoints = session.query(ModelCheckpoint).filter_by(
                    session_id=model.training_session_id
                ).order_by(ModelCheckpoint.created_at.desc()).all()
            
            return {
                "model_id": model_id,
                "checkpoints": [checkpoint.to_dict() for checkpoint in checkpoints],
                "total": len(checkpoints)
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list model checkpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search")
async def search_models(
    query: Optional[str] = None,
    model_type: Optional[str] = None,
    base_model: Optional[str] = None,
    is_public: Optional[bool] = None
):
    """Search models with filters"""
    try:
        with db_manager.get_session_context() as session:
            query_obj = session.query(ModelRegistry).filter_by(is_active=True)
            
            # Apply filters
            if query:
                query_obj = query_obj.filter(
                    ModelRegistry.name.contains(query) |
                    ModelRegistry.description.contains(query)
                )
            
            if model_type:
                query_obj = query_obj.filter_by(model_type=model_type)
            
            if base_model:
                query_obj = query_obj.filter_by(base_model=base_model)
            
            if is_public is not None:
                query_obj = query_obj.filter_by(is_public=is_public)
            
            models = query_obj.all()
            
            return {
                "models": [model.to_dict() for model in models],
                "total": len(models),
                "filters": {
                    "query": query,
                    "model_type": model_type,
                    "base_model": base_model,
                    "is_public": is_public
                }
            }
            
    except Exception as e:
        logger.error(f"Failed to search models: {e}")
        raise HTTPException(status_code=500, detail=str(e))