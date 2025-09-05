"""
Database Models for Persian Legal AI Training System
مدل‌های پایگاه داده برای سیستم آموزش هوش مصنوعی حقوقی فارسی
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class TrainingSession(Base):
    """Training session model"""
    __tablename__ = "training_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # dora, qr_adaptor, etc.
    status = Column(String, nullable=False, default="pending")  # pending, running, paused, completed, failed
    config = Column(JSON, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Training progress
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=0)
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    
    # Performance metrics
    current_loss = Column(Float)
    best_loss = Column(Float)
    current_accuracy = Column(Float)
    best_accuracy = Column(Float)
    learning_rate = Column(Float)
    
    # System metrics
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    training_speed = Column(Float)  # steps per second
    
    # Error handling
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    # Relationships
    checkpoints = relationship("ModelCheckpoint", back_populates="session", cascade="all, delete-orphan")
    metrics = relationship("TrainingMetrics", back_populates="session", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'status': self.status,
            'config': self.config,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'current_loss': self.current_loss,
            'best_loss': self.best_loss,
            'current_accuracy': self.current_accuracy,
            'best_accuracy': self.best_accuracy,
            'learning_rate': self.learning_rate,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'training_speed': self.training_speed,
            'error_message': self.error_message,
            'retry_count': self.retry_count
        }

class ModelCheckpoint(Base):
    """Model checkpoint model"""
    __tablename__ = "model_checkpoints"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("training_sessions.id"), nullable=False)
    
    # Checkpoint metadata
    epoch = Column(Integer, nullable=False)
    step = Column(Integer, nullable=False)
    checkpoint_type = Column(String, nullable=False)  # best, latest, epoch_end
    
    # Performance metrics at checkpoint
    loss = Column(Float, nullable=False)
    accuracy = Column(Float)
    learning_rate = Column(Float)
    
    # File information
    file_path = Column(String, nullable=False)
    file_size_mb = Column(Float)
    checksum = Column(String)  # For integrity verification
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    session = relationship("TrainingSession", back_populates="checkpoints")
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'epoch': self.epoch,
            'step': self.step,
            'checkpoint_type': self.checkpoint_type,
            'loss': self.loss,
            'accuracy': self.accuracy,
            'learning_rate': self.learning_rate,
            'file_path': self.file_path,
            'file_size_mb': self.file_size_mb,
            'checksum': self.checksum,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class TrainingMetrics(Base):
    """Training metrics model for detailed tracking"""
    __tablename__ = "training_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("training_sessions.id"), nullable=False)
    
    # Timestamp
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Training metrics
    epoch = Column(Integer, nullable=False)
    step = Column(Integer, nullable=False)
    loss = Column(Float, nullable=False)
    accuracy = Column(Float)
    learning_rate = Column(Float)
    
    # System metrics
    cpu_usage = Column(Float)
    memory_usage_mb = Column(Float)
    gpu_usage = Column(Float)  # If available
    gpu_memory_mb = Column(Float)  # If available
    
    # Training speed
    steps_per_second = Column(Float)
    samples_per_second = Column(Float)
    
    # Additional metrics
    gradient_norm = Column(Float)
    weight_norm = Column(Float)
    
    # Relationships
    session = relationship("TrainingSession", back_populates="metrics")
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'epoch': self.epoch,
            'step': self.step,
            'loss': self.loss,
            'accuracy': self.accuracy,
            'learning_rate': self.learning_rate,
            'cpu_usage': self.cpu_usage,
            'memory_usage_mb': self.memory_usage_mb,
            'gpu_usage': self.gpu_usage,
            'gpu_memory_mb': self.gpu_memory_mb,
            'steps_per_second': self.steps_per_second,
            'samples_per_second': self.samples_per_second,
            'gradient_norm': self.gradient_norm,
            'weight_norm': self.weight_norm
        }

class DataSource(Base):
    """Data source model for tracking data collection"""
    __tablename__ = "data_sources"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    source_type = Column(String, nullable=False)  # naab, qavanin, majles, etc.
    url = Column(String)
    
    # Configuration
    config = Column(JSON)
    api_key = Column(String)  # Encrypted in production
    
    # Status
    is_active = Column(Boolean, default=True)
    last_sync = Column(DateTime)
    sync_frequency_hours = Column(Integer, default=24)
    
    # Statistics
    total_documents = Column(Integer, default=0)
    last_document_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'source_type': self.source_type,
            'url': self.url,
            'config': self.config,
            'is_active': self.is_active,
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'sync_frequency_hours': self.sync_frequency_hours,
            'total_documents': self.total_documents,
            'last_document_count': self.last_document_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Dataset(Base):
    """Dataset model for training data"""
    __tablename__ = "datasets"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    task_type = Column(String, nullable=False)  # qa, ner, classification, generation
    
    # Dataset information
    description = Column(Text)
    language = Column(String, default="persian")
    
    # File information
    file_path = Column(String, nullable=False)
    file_size_mb = Column(Float)
    checksum = Column(String)
    
    # Statistics
    total_samples = Column(Integer, default=0)
    train_samples = Column(Integer, default=0)
    validation_samples = Column(Integer, default=0)
    test_samples = Column(Integer, default=0)
    
    # Quality metrics
    quality_score = Column(Float)
    avg_length = Column(Float)
    language_confidence = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'task_type': self.task_type,
            'description': self.description,
            'language': self.language,
            'file_path': self.file_path,
            'file_size_mb': self.file_size_mb,
            'checksum': self.checksum,
            'total_samples': self.total_samples,
            'train_samples': self.train_samples,
            'validation_samples': self.validation_samples,
            'test_samples': self.test_samples,
            'quality_score': self.quality_score,
            'avg_length': self.avg_length,
            'language_confidence': self.language_confidence,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class ModelRegistry(Base):
    """Model registry for tracking trained models"""
    __tablename__ = "model_registry"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True)
    model_type = Column(String, nullable=False)
    base_model = Column(String, nullable=False)
    
    # Model information
    description = Column(Text)
    version = Column(String, default="1.0")
    
    # Training information
    training_session_id = Column(String, ForeignKey("training_sessions.id"))
    dataset_id = Column(String, ForeignKey("datasets.id"))
    
    # Performance metrics
    best_accuracy = Column(Float)
    best_loss = Column(Float)
    evaluation_metrics = Column(JSON)
    
    # File information
    model_path = Column(String, nullable=False)
    config_path = Column(String)
    tokenizer_path = Column(String)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'model_type': self.model_type,
            'base_model': self.base_model,
            'description': self.description,
            'version': self.version,
            'training_session_id': self.training_session_id,
            'dataset_id': self.dataset_id,
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
            'evaluation_metrics': self.evaluation_metrics,
            'model_path': self.model_path,
            'config_path': self.config_path,
            'tokenizer_path': self.tokenizer_path,
            'is_active': self.is_active,
            'is_public': self.is_public,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }