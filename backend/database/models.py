"""
Real Database Models for Persian Legal AI Training System
مدل‌های واقعی پایگاه داده برای سیستم آموزش هوش مصنوعی حقوقی فارسی
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class TrainingSession(Base):
    """Real training session model"""
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
    
    # Data information
    data_source = Column(String)
    task_type = Column(String)
    train_samples = Column(Integer, default=0)
    eval_samples = Column(Integer, default=0)
    
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
            'retry_count': self.retry_count,
            'data_source': self.data_source,
            'task_type': self.task_type,
            'train_samples': self.train_samples,
            'eval_samples': self.eval_samples
        }

class ModelCheckpoint(Base):
    """Real model checkpoint model"""
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
    file_size_bytes = Column(Integer)
    
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
            'file_size_bytes': self.file_size_bytes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class TrainingMetrics(Base):
    """Real training metrics model"""
    __tablename__ = "training_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("training_sessions.id"), nullable=False)
    
    # Training metrics
    epoch = Column(Integer, nullable=False)
    step = Column(Integer, nullable=False)
    loss = Column(Float, nullable=False)
    accuracy = Column(Float)
    learning_rate = Column(Float)
    
    # System metrics
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    gpu_usage = Column(Float)
    gpu_memory = Column(Float)
    training_speed = Column(Float)  # steps per second
    
    # Timestamps
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    session = relationship("TrainingSession", back_populates="metrics")
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'epoch': self.epoch,
            'step': self.step,
            'loss': self.loss,
            'accuracy': self.accuracy,
            'learning_rate': self.learning_rate,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'gpu_memory': self.gpu_memory,
            'training_speed': self.training_speed,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class DataSource(Base):
    """Real data source model"""
    __tablename__ = "data_sources"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True)
    source_type = Column(String, nullable=False)  # api, web_scraping, file, etc.
    url = Column(String)
    config = Column(JSON)
    
    # Data statistics
    total_documents = Column(Integer, default=0)
    processed_documents = Column(Integer, default=0)
    quality_score = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'source_type': self.source_type,
            'url': self.url,
            'config': self.config,
            'total_documents': self.total_documents,
            'processed_documents': self.processed_documents,
            'quality_score': self.quality_score,
            'is_active': self.is_active,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class LegalDocument(Base):
    """Real legal document model"""
    __tablename__ = "legal_documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String, nullable=False)
    category = Column(String)
    
    # Processing information
    word_count = Column(Integer)
    char_count = Column(Integer)
    language_confidence = Column(Float)
    legal_relevance = Column(Float)
    quality_score = Column(Float)
    
    # Metadata
    document_metadata = Column(JSON)
    processed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'source': self.source,
            'category': self.category,
            'word_count': self.word_count,
            'char_count': self.char_count,
            'language_confidence': self.language_confidence,
            'legal_relevance': self.legal_relevance,
            'quality_score': self.quality_score,
            'metadata': self.document_metadata,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class SystemLog(Base):
    """Real system log model"""
    __tablename__ = "system_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    level = Column(String, nullable=False)  # INFO, WARNING, ERROR, DEBUG
    message = Column(Text, nullable=False)
    component = Column(String)  # training, api, data_processing, etc.
    session_id = Column(String, ForeignKey("training_sessions.id"))
    
    # Additional context
    context = Column(JSON)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'level': self.level,
            'message': self.message,
            'component': self.component,
            'session_id': self.session_id,
            'context': self.context,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }