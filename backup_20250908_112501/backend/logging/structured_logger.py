"""
Structured Logging System for Persian Legal AI
سیستم لاگ‌گیری ساختاریافته برای هوش مصنوعی حقوقی فارسی
"""

import logging
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import traceback
import psutil
import threading
from contextvars import ContextVar

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')
session_id_var: ContextVar[str] = ContextVar('session_id', default='')

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record):
        # Base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_id': threading.get_ident(),
            'process_id': os.getpid()
        }
        
        # Add context information
        if request_id_var.get():
            log_entry['request_id'] = request_id_var.get()
        if user_id_var.get():
            log_entry['user_id'] = user_id_var.get()
        if session_id_var.get():
            log_entry['session_id'] = session_id_var.get()
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)

class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, logger_name: str = 'performance'):
        self.logger = logging.getLogger(logger_name)
        self.start_times = {}
    
    def start_timer(self, operation: str, **context):
        """Start timing an operation"""
        self.start_times[operation] = {
            'start_time': datetime.utcnow(),
            'context': context
        }
    
    def end_timer(self, operation: str, **metrics):
        """End timing an operation and log performance"""
        if operation not in self.start_times:
            self.logger.warning(f"Timer for operation '{operation}' was not started")
            return
        
        start_data = self.start_times.pop(operation)
        duration = (datetime.utcnow() - start_data['start_time']).total_seconds()
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        self.logger.info(
            f"Operation '{operation}' completed",
            extra={
                'operation': operation,
                'duration_seconds': duration,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                **start_data['context'],
                **metrics
            }
        )
    
    def log_metric(self, metric_name: str, value: float, **context):
        """Log a performance metric"""
        self.logger.info(
            f"Performance metric: {metric_name}",
            extra={
                'metric_name': metric_name,
                'metric_value': value,
                'timestamp': datetime.utcnow().isoformat(),
                **context
            }
        )

class SecurityLogger:
    """Logger for security events"""
    
    def __init__(self, logger_name: str = 'security'):
        self.logger = logging.getLogger(logger_name)
    
    def log_login_attempt(self, username: str, success: bool, ip_address: str = None, user_agent: str = None):
        """Log login attempt"""
        self.logger.info(
            f"Login attempt: {'success' if success else 'failed'}",
            extra={
                'event_type': 'login_attempt',
                'username': username,
                'success': success,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_authentication_failure(self, reason: str, ip_address: str = None, user_agent: str = None):
        """Log authentication failure"""
        self.logger.warning(
            f"Authentication failure: {reason}",
            extra={
                'event_type': 'auth_failure',
                'reason': reason,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_rate_limit_exceeded(self, endpoint: str, ip_address: str, limit: int):
        """Log rate limit exceeded"""
        self.logger.warning(
            f"Rate limit exceeded for endpoint {endpoint}",
            extra={
                'event_type': 'rate_limit_exceeded',
                'endpoint': endpoint,
                'ip_address': ip_address,
                'limit': limit,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_permission_denied(self, user_id: str, resource: str, action: str, reason: str = None):
        """Log permission denied"""
        self.logger.warning(
            f"Permission denied: {user_id} attempted {action} on {resource}",
            extra={
                'event_type': 'permission_denied',
                'user_id': user_id,
                'resource': resource,
                'action': action,
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_suspicious_activity(self, activity_type: str, details: Dict[str, Any], ip_address: str = None):
        """Log suspicious activity"""
        self.logger.error(
            f"Suspicious activity detected: {activity_type}",
            extra={
                'event_type': 'suspicious_activity',
                'activity_type': activity_type,
                'details': details,
                'ip_address': ip_address,
                'timestamp': datetime.utcnow().isoformat()
            }
        )

class TrainingLogger:
    """Logger for training events"""
    
    def __init__(self, logger_name: str = 'training'):
        self.logger = logging.getLogger(logger_name)
    
    def log_training_start(self, session_id: str, model_type: str, config: Dict[str, Any], user_id: str = None):
        """Log training session start"""
        self.logger.info(
            f"Training session started: {session_id}",
            extra={
                'event_type': 'training_start',
                'session_id': session_id,
                'model_type': model_type,
                'config': config,
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_training_progress(self, session_id: str, epoch: int, step: int, loss: float, metrics: Dict[str, Any] = None):
        """Log training progress"""
        self.logger.info(
            f"Training progress: {session_id} - Epoch {epoch}, Step {step}",
            extra={
                'event_type': 'training_progress',
                'session_id': session_id,
                'epoch': epoch,
                'step': step,
                'loss': loss,
                'metrics': metrics or {},
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_training_completion(self, session_id: str, final_metrics: Dict[str, Any], duration_seconds: float):
        """Log training completion"""
        self.logger.info(
            f"Training session completed: {session_id}",
            extra={
                'event_type': 'training_completion',
                'session_id': session_id,
                'final_metrics': final_metrics,
                'duration_seconds': duration_seconds,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_training_error(self, session_id: str, error: Exception, context: Dict[str, Any] = None):
        """Log training error"""
        self.logger.error(
            f"Training session error: {session_id} - {str(error)}",
            extra={
                'event_type': 'training_error',
                'session_id': session_id,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context or {},
                'timestamp': datetime.utcnow().isoformat()
            },
            exc_info=True
        )
    
    def log_model_save(self, session_id: str, model_path: str, model_size_mb: float):
        """Log model save"""
        self.logger.info(
            f"Model saved: {session_id}",
            extra={
                'event_type': 'model_save',
                'session_id': session_id,
                'model_path': model_path,
                'model_size_mb': model_size_mb,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_checkpoint_save(self, session_id: str, checkpoint_path: str, epoch: int, step: int):
        """Log checkpoint save"""
        self.logger.info(
            f"Checkpoint saved: {session_id}",
            extra={
                'event_type': 'checkpoint_save',
                'session_id': session_id,
                'checkpoint_path': checkpoint_path,
                'epoch': epoch,
                'step': step,
                'timestamp': datetime.utcnow().isoformat()
            }
        )

class SystemLogger:
    """Logger for system events"""
    
    def __init__(self, logger_name: str = 'system'):
        self.logger = logging.getLogger(logger_name)
    
    def log_system_startup(self, version: str, config: Dict[str, Any]):
        """Log system startup"""
        self.logger.info(
            "System startup",
            extra={
                'event_type': 'system_startup',
                'version': version,
                'config': config,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_system_shutdown(self, reason: str = None):
        """Log system shutdown"""
        self.logger.info(
            "System shutdown",
            extra={
                'event_type': 'system_shutdown',
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_health_check(self, status: str, metrics: Dict[str, Any]):
        """Log health check"""
        self.logger.info(
            f"Health check: {status}",
            extra={
                'event_type': 'health_check',
                'status': status,
                'metrics': metrics,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_resource_usage(self, cpu_percent: float, memory_percent: float, gpu_usage: Dict[str, Any] = None):
        """Log resource usage"""
        self.logger.info(
            "Resource usage update",
            extra={
                'event_type': 'resource_usage',
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'gpu_usage': gpu_usage or {},
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_database_event(self, event_type: str, details: Dict[str, Any]):
        """Log database event"""
        self.logger.info(
            f"Database event: {event_type}",
            extra={
                'event_type': 'database_event',
                'db_event_type': event_type,
                'details': details,
                'timestamp': datetime.utcnow().isoformat()
            }
        )

def setup_structured_logging(log_level: str = 'INFO', log_dir: str = './logs'):
    """Setup structured logging for the application"""
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with structured formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler for all logs
    file_handler = logging.FileHandler(log_path / 'application.log', encoding='utf-8')
    file_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(file_handler)
    
    # Separate file handlers for different log types
    loggers_config = [
        ('security', 'security.log'),
        ('training', 'training.log'),
        ('performance', 'performance.log'),
        ('system', 'system.log'),
        ('api', 'api.log')
    ]
    
    for logger_name, filename in loggers_config:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # File handler for specific logger
        handler = logging.FileHandler(log_path / filename, encoding='utf-8')
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        
        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False
    
    # Configure third-party loggers
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    
    logging.info("Structured logging setup complete", extra={
        'log_level': log_level,
        'log_directory': str(log_path)
    })

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)

def set_request_context(request_id: str, user_id: str = None, session_id: str = None):
    """Set request context for logging"""
    request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if session_id:
        session_id_var.set(session_id)

def clear_request_context():
    """Clear request context"""
    request_id_var.set('')
    user_id_var.set('')
    session_id_var.set('')

# Global logger instances
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()
training_logger = TrainingLogger()
system_logger = SystemLogger()