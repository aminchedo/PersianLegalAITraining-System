from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from models.model_training import ModelTraining
from config.database import get_db
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(prefix='/api/real/models', tags=['models'])

class ModelTrainingCreate(BaseModel):
    name: str
    description: Optional[str] = None
    framework: str
    epochs: int = 100
    parameters: Optional[dict] = None
    team_member_id: Optional[int] = None

class ModelTrainingUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    progress: Optional[float] = None
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    current_epoch: Optional[int] = None
    time_remaining: Optional[str] = None

class ModelTrainingResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    status: str
    progress: float
    accuracy: float
    loss: float
    parameters: Optional[dict]
    framework: str
    dora_rank: int
    epochs: int
    current_epoch: int
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    time_remaining: Optional[str]
    dataset_size: int
    model_size: float
    gpu_usage: float
    memory_usage: float
    created_at: datetime
    updated_at: datetime
    team_member_id: Optional[int]

    class Config:
        from_attributes = True

@router.get('/training', response_model=List[ModelTrainingResponse])
async def get_training_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    framework: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get all training jobs with optional filtering"""
    query = db.query(ModelTraining)
    
    if status:
        query = query.filter(ModelTraining.status == status)
    
    if framework:
        query = query.filter(ModelTraining.framework == framework)
    
    jobs = query.offset(skip).limit(limit).order_by(ModelTraining.created_at.desc()).all()
    return jobs

@router.get('/training/{job_id}', response_model=ModelTrainingResponse)
async def get_training_job(job_id: int, db: Session = Depends(get_db)):
    """Get a specific training job by ID"""
    job = db.query(ModelTraining).filter(ModelTraining.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail='Training job not found')
    return job

@router.post('/training', response_model=ModelTrainingResponse)
async def create_training_job(job_data: ModelTrainingCreate, db: Session = Depends(get_db)):
    """Create a new training job"""
    new_job = ModelTraining(**job_data.dict())
    db.add(new_job)
    db.commit()
    db.refresh(new_job)
    return new_job

@router.put('/training/{job_id}', response_model=ModelTrainingResponse)
async def update_training_job(
    job_id: int, 
    job_data: ModelTrainingUpdate, 
    db: Session = Depends(get_db)
):
    """Update a training job"""
    job = db.query(ModelTraining).filter(ModelTraining.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail='Training job not found')
    
    # Update only provided fields
    update_data = job_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(job, field, value)
    
    db.commit()
    db.refresh(job)
    return job

@router.post('/training/{job_id}/start')
async def start_training_job(job_id: int, db: Session = Depends(get_db)):
    """Start a training job"""
    job = db.query(ModelTraining).filter(ModelTraining.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail='Training job not found')
    
    if job.status not in ['pending', 'paused']:
        raise HTTPException(status_code=400, detail='Job cannot be started in current status')
    
    job.status = 'training'
    job.start_time = datetime.utcnow()
    job.progress = 0.0
    
    db.commit()
    db.refresh(job)
    return {"message": "Training job started successfully", "job": job}

@router.post('/training/{job_id}/pause')
async def pause_training_job(job_id: int, db: Session = Depends(get_db)):
    """Pause a training job"""
    job = db.query(ModelTraining).filter(ModelTraining.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail='Training job not found')
    
    if job.status != 'training':
        raise HTTPException(status_code=400, detail='Job is not currently training')
    
    job.status = 'paused'
    db.commit()
    db.refresh(job)
    return {"message": "Training job paused successfully", "job": job}

@router.post('/training/{job_id}/stop')
async def stop_training_job(job_id: int, db: Session = Depends(get_db)):
    """Stop a training job"""
    job = db.query(ModelTraining).filter(ModelTraining.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail='Training job not found')
    
    if job.status not in ['training', 'paused']:
        raise HTTPException(status_code=400, detail='Job cannot be stopped in current status')
    
    job.status = 'error'
    job.end_time = datetime.utcnow()
    
    db.commit()
    db.refresh(job)
    return {"message": "Training job stopped successfully", "job": job}

@router.get('/stats')
async def get_model_stats(db: Session = Depends(get_db)):
    """Get model training statistics"""
    total_jobs = db.query(ModelTraining).count()
    active_jobs = db.query(ModelTraining).filter(ModelTraining.status == 'training').count()
    completed_jobs = db.query(ModelTraining).filter(ModelTraining.status == 'completed').count()
    failed_jobs = db.query(ModelTraining).filter(ModelTraining.status == 'error').count()
    
    # Average accuracy for completed jobs
    avg_accuracy = db.query(ModelTraining.accuracy).filter(
        ModelTraining.status == 'completed',
        ModelTraining.accuracy > 0
    ).all()
    
    avg_accuracy_value = sum(acc[0] for acc in avg_accuracy) / len(avg_accuracy) if avg_accuracy else 0
    
    return {
        "total_jobs": total_jobs,
        "active_jobs": active_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "average_accuracy": round(avg_accuracy_value, 2)
    }