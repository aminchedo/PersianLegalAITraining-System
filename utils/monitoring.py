"""
System Monitoring Module
Comprehensive monitoring with Windows Event Log integration and performance tracking
"""

import os
import sys
import time
import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import json
import sqlite3

import psutil
import numpy as np
from loguru import logger

# Windows-specific imports
try:
    import win32evtlog
    import win32evtlogutil
    import win32api
    import wmi
    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False

# PyTorch monitoring
import torch
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

# Local imports
from config.training_config import MonitoringConfig


@dataclass
class SystemMetrics:
    """System performance metrics snapshot"""
    
    timestamp: datetime
    cpu_usage: float
    cpu_per_core: List[float]
    memory_usage: float
    memory_available: float
    disk_usage: Dict[str, float]
    network_io: Dict[str, int]
    process_count: int
    thread_count: int
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None


@dataclass
class TrainingMetrics:
    """Training-specific metrics"""
    
    timestamp: datetime
    training_active: bool
    current_epoch: int
    current_step: int
    loss: float
    learning_rate: float
    batch_size: int
    throughput: float  # samples/second
    memory_usage: float
    gradient_norm: float
    model_parameters: int


@dataclass
class Alert:
    """System alert information"""
    
    id: str
    timestamp: datetime
    level: str  # INFO, WARNING, ERROR, CRITICAL
    category: str
    title: str
    message: str
    metrics: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False


class PerformanceTracker:
    """Track and analyze performance metrics over time"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics_history: List[SystemMetrics] = []
        self.training_history: List[TrainingMetrics] = []
        
        # Performance statistics
        self.stats = {
            'cpu_avg': 0.0,
            'cpu_max': 0.0,
            'memory_avg': 0.0,
            'memory_max': 0.0,
            'uptime': 0.0,
            'training_throughput_avg': 0.0
        }
    
    def add_system_metrics(self, metrics: SystemMetrics) -> None:
        """Add system metrics to history"""
        self.metrics_history.append(metrics)
        
        # Limit history size
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
        
        # Update statistics
        self._update_stats()
    
    def add_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Add training metrics to history"""
        self.training_history.append(metrics)
        
        # Limit history size
        if len(self.training_history) > self.max_history:
            self.training_history = self.training_history[-self.max_history:]
    
    def _update_stats(self) -> None:
        """Update performance statistics"""
        if not self.metrics_history:
            return
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 measurements
        
        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_usage for m in recent_metrics]
        
        self.stats.update({
            'cpu_avg': np.mean(cpu_values),
            'cpu_max': np.max(cpu_values),
            'memory_avg': np.mean(memory_values),
            'memory_max': np.max(memory_values),
            'uptime': (datetime.now() - self.metrics_history[0].timestamp).total_seconds()
        })
        
        # Training throughput
        if self.training_history:
            recent_training = self.training_history[-50:]
            throughput_values = [t.throughput for t in recent_training if t.throughput > 0]
            if throughput_values:
                self.stats['training_throughput_avg'] = np.mean(throughput_values)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        return {
            'current_metrics': asdict(self.metrics_history[-1]) if self.metrics_history else None,
            'statistics': self.stats.copy(),
            'history_length': len(self.metrics_history),
            'training_history_length': len(self.training_history)
        }
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies"""
        anomalies = []
        
        if len(self.metrics_history) < 10:
            return anomalies
        
        recent_metrics = self.metrics_history[-10:]
        
        # CPU spike detection
        cpu_values = [m.cpu_usage for m in recent_metrics]
        if np.mean(cpu_values) > 90:
            anomalies.append({
                'type': 'cpu_spike',
                'severity': 'high',
                'value': np.mean(cpu_values),
                'message': f'High CPU usage detected: {np.mean(cpu_values):.1f}%'
            })
        
        # Memory spike detection
        memory_values = [m.memory_usage for m in recent_metrics]
        if np.mean(memory_values) > 90:
            anomalies.append({
                'type': 'memory_spike',
                'severity': 'high',
                'value': np.mean(memory_values),
                'message': f'High memory usage detected: {np.mean(memory_values):.1f}%'
            })
        
        # Training performance degradation
        if len(self.training_history) >= 10:
            recent_training = self.training_history[-10:]
            throughput_values = [t.throughput for t in recent_training if t.throughput > 0]
            
            if throughput_values and len(throughput_values) >= 5:
                recent_avg = np.mean(throughput_values[-5:])
                historical_avg = self.stats.get('training_throughput_avg', 0)
                
                if historical_avg > 0 and recent_avg < historical_avg * 0.7:
                    anomalies.append({
                        'type': 'training_slowdown',
                        'severity': 'medium',
                        'value': recent_avg,
                        'message': f'Training throughput decreased: {recent_avg:.1f} vs {historical_avg:.1f} samples/sec'
                    })
        
        return anomalies


class AlertManager:
    """Manage system alerts and notifications"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Alert thresholds
        self.thresholds = {
            'cpu_warning': config.cpu_usage_threshold,
            'memory_warning': config.memory_usage_threshold,
            'disk_warning': config.disk_usage_threshold
        }
        
        # Alert cooldown to prevent spam
        self.alert_cooldown = {}
        self.cooldown_duration = 300  # 5 minutes
        
        logger.info("Alert manager initialized")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def create_alert(
        self,
        level: str,
        category: str,
        title: str,
        message: str,
        metrics: Dict[str, Any] = None
    ) -> Alert:
        """Create a new alert"""
        
        alert_id = f"{category}_{int(time.time())}"
        
        # Check cooldown
        cooldown_key = f"{category}_{level}"
        current_time = time.time()
        
        if cooldown_key in self.alert_cooldown:
            if current_time - self.alert_cooldown[cooldown_key] < self.cooldown_duration:
                return None  # Skip alert due to cooldown
        
        self.alert_cooldown[cooldown_key] = current_time
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            level=level,
            category=category,
            title=title,
            message=message,
            metrics=metrics or {}
        )
        
        self.alerts.append(alert)
        
        # Limit alert history
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        return alert
    
    def check_system_thresholds(self, metrics: SystemMetrics) -> None:
        """Check system metrics against thresholds"""
        
        # CPU usage check
        if metrics.cpu_usage > self.thresholds['cpu_warning']:
            self.create_alert(
                level='WARNING' if metrics.cpu_usage < 95 else 'ERROR',
                category='system',
                title='High CPU Usage',
                message=f'CPU usage is {metrics.cpu_usage:.1f}%',
                metrics={'cpu_usage': metrics.cpu_usage, 'per_core': metrics.cpu_per_core}
            )
        
        # Memory usage check
        if metrics.memory_usage > self.thresholds['memory_warning']:
            self.create_alert(
                level='WARNING' if metrics.memory_usage < 95 else 'ERROR',
                category='system',
                title='High Memory Usage',
                message=f'Memory usage is {metrics.memory_usage:.1f}%',
                metrics={'memory_usage': metrics.memory_usage, 'available': metrics.memory_available}
            )
        
        # Disk usage check
        for drive, usage in metrics.disk_usage.items():
            if usage > self.thresholds['disk_warning']:
                self.create_alert(
                    level='WARNING' if usage < 95 else 'ERROR',
                    category='system',
                    title='High Disk Usage',
                    message=f'Disk {drive} usage is {usage:.1f}%',
                    metrics={'disk': drive, 'usage': usage}
                )
    
    def get_active_alerts(self, level: Optional[str] = None) -> List[Alert]:
        """Get active (unresolved) alerts"""
        alerts = [a for a in self.alerts if not a.resolved]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False


class WindowsEventLogger:
    """Windows Event Log integration"""
    
    def __init__(self, source_name: str = "Persian Legal AI"):
        self.source_name = source_name
        self.available = WINDOWS_AVAILABLE
        
        if not self.available:
            logger.warning("Windows Event Log not available on this platform")
            return
        
        try:
            # Register event source if needed
            win32evtlogutil.AddSourceToRegistry(
                appName=source_name,
                msgDLL=win32evtlogutil.GetEventLogMessageExe(),
                eventLogType="Application"
            )
            logger.info(f"Windows Event Log source registered: {source_name}")
        except Exception as e:
            logger.warning(f"Could not register Windows Event Log source: {e}")
            self.available = False
    
    def log_event(self, level: str, message: str, event_id: int = 1000) -> None:
        """Log event to Windows Event Log"""
        
        if not self.available:
            return
        
        try:
            # Map log levels to Windows event types
            event_type_map = {
                'INFO': win32evtlog.EVENTLOG_INFORMATION_TYPE,
                'WARNING': win32evtlog.EVENTLOG_WARNING_TYPE,
                'ERROR': win32evtlog.EVENTLOG_ERROR_TYPE,
                'CRITICAL': win32evtlog.EVENTLOG_ERROR_TYPE
            }
            
            event_type = event_type_map.get(level, win32evtlog.EVENTLOG_INFORMATION_TYPE)
            
            win32evtlogutil.ReportEvent(
                self.source_name,
                event_id,
                eventCategory=0,
                eventType=event_type,
                strings=[message]
            )
            
        except Exception as e:
            logger.error(f"Failed to log Windows event: {e}")


class SystemMonitor:
    """Main system monitoring class"""
    
    def __init__(
        self,
        update_interval: int = 5,
        log_metrics: bool = True,
        enable_alerts: bool = True,
        config: Optional[MonitoringConfig] = None
    ):
        self.update_interval = update_interval
        self.log_metrics = log_metrics
        self.enable_alerts = enable_alerts
        self.config = config or MonitoringConfig()
        
        # Initialize components
        self.performance_tracker = PerformanceTracker()
        self.alert_manager = AlertManager(self.config)
        self.windows_logger = WindowsEventLogger()
        
        # Monitoring state
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Database for metrics storage
        self.db_path = Path("./logs/monitoring.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        
        # Setup alert handlers
        self._setup_alert_handlers()
        
        logger.info("System monitor initialized")
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database for metrics storage"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # System metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    timestamp TEXT PRIMARY KEY,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage TEXT,
                    network_io TEXT,
                    process_count INTEGER,
                    thread_count INTEGER
                )
            ''')
            
            # Training metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_metrics (
                    timestamp TEXT PRIMARY KEY,
                    training_active BOOLEAN,
                    current_epoch INTEGER,
                    current_step INTEGER,
                    loss REAL,
                    learning_rate REAL,
                    throughput REAL,
                    memory_usage REAL
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    level TEXT,
                    category TEXT,
                    title TEXT,
                    message TEXT,
                    metrics TEXT,
                    acknowledged BOOLEAN,
                    resolved BOOLEAN
                )
            ''')
            
            conn.commit()
    
    def _setup_alert_handlers(self) -> None:
        """Setup alert handlers"""
        
        # Log alert handler
        def log_alert_handler(alert: Alert) -> None:
            level_map = {
                'INFO': logger.info,
                'WARNING': logger.warning,
                'ERROR': logger.error,
                'CRITICAL': logger.critical
            }
            
            log_func = level_map.get(alert.level, logger.info)
            log_func(f"ALERT [{alert.category}] {alert.title}: {alert.message}")
            
            # Log to Windows Event Log
            if self.config.enable_windows_event_log:
                self.windows_logger.log_event(alert.level, f"{alert.title}: {alert.message}")
        
        # Database alert handler
        def db_alert_handler(alert: Alert) -> None:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO alerts 
                        (id, timestamp, level, category, title, message, metrics, acknowledged, resolved)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        alert.id,
                        alert.timestamp.isoformat(),
                        alert.level,
                        alert.category,
                        alert.title,
                        alert.message,
                        json.dumps(alert.metrics),
                        alert.acknowledged,
                        alert.resolved
                    ))
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to save alert to database: {e}")
        
        self.alert_manager.add_alert_handler(log_alert_handler)
        self.alert_manager.add_alert_handler(db_alert_handler)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        
        # CPU metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_per_core = psutil.cpu_percent(percpu=True)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_available = memory.available / (1024**3)  # GB
        
        # Disk metrics
        disk_usage = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.device] = (usage.used / usage.total) * 100
            except (PermissionError, OSError):
                continue
        
        # Network metrics
        network_io = psutil.net_io_counters()._asdict()
        
        # Process metrics
        process_count = len(psutil.pids())
        
        # Thread count for current process
        current_process = psutil.Process()
        thread_count = current_process.num_threads()
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            cpu_per_core=cpu_per_core,
            memory_usage=memory_usage,
            memory_available=memory_available,
            disk_usage=disk_usage,
            network_io=network_io,
            process_count=process_count,
            thread_count=thread_count
        )
    
    def collect_training_metrics(
        self,
        training_active: bool = False,
        current_epoch: int = 0,
        current_step: int = 0,
        loss: float = 0.0,
        learning_rate: float = 0.0,
        batch_size: int = 1,
        throughput: float = 0.0
    ) -> TrainingMetrics:
        """Collect training-specific metrics"""
        
        # Get model parameter count if available
        model_parameters = 0
        try:
            if torch.cuda.is_available():
                # This would be customized based on actual model
                pass
        except Exception:
            pass
        
        # Calculate gradient norm
        gradient_norm = 0.0
        # This would be calculated from actual gradients during training
        
        # Memory usage specific to training
        training_memory = psutil.virtual_memory().percent
        
        return TrainingMetrics(
            timestamp=datetime.now(),
            training_active=training_active,
            current_epoch=current_epoch,
            current_step=current_step,
            loss=loss,
            learning_rate=learning_rate,
            batch_size=batch_size,
            throughput=throughput,
            memory_usage=training_memory,
            gradient_norm=gradient_norm,
            model_parameters=model_parameters
        )
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        
        while self.is_running:
            try:
                # Collect system metrics
                system_metrics = self.collect_system_metrics()
                self.performance_tracker.add_system_metrics(system_metrics)
                
                # Check thresholds and create alerts
                if self.enable_alerts:
                    self.alert_manager.check_system_thresholds(system_metrics)
                
                # Detect anomalies
                anomalies = self.performance_tracker.detect_anomalies()
                for anomaly in anomalies:
                    self.alert_manager.create_alert(
                        level='WARNING',
                        category='performance',
                        title=f'Performance Anomaly: {anomaly["type"]}',
                        message=anomaly['message'],
                        metrics=anomaly
                    )
                
                # Save to database
                if self.log_metrics:
                    self._save_metrics_to_db(system_metrics)
                
                # Log performance summary periodically
                if len(self.performance_tracker.metrics_history) % 60 == 0:  # Every 5 minutes
                    summary = self.performance_tracker.get_performance_summary()
                    logger.info(f"Performance Summary - CPU: {summary['statistics']['cpu_avg']:.1f}%, "
                               f"Memory: {summary['statistics']['memory_avg']:.1f}%, "
                               f"Uptime: {summary['statistics']['uptime']:.0f}s")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _save_metrics_to_db(self, metrics: SystemMetrics) -> None:
        """Save metrics to database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO system_metrics 
                    (timestamp, cpu_usage, memory_usage, disk_usage, network_io, process_count, thread_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp.isoformat(),
                    metrics.cpu_usage,
                    metrics.memory_usage,
                    json.dumps(metrics.disk_usage),
                    json.dumps(metrics.network_io),
                    metrics.process_count,
                    metrics.thread_count
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save metrics to database: {e}")
    
    async def start(self) -> None:
        """Start monitoring"""
        
        if self.is_running:
            logger.warning("Monitor is already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("System monitoring started")
    
    async def stop(self) -> None:
        """Stop monitoring"""
        
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        logger.info("System monitoring stopped")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        
        if not self.performance_tracker.metrics_history:
            return {}
        
        latest_metrics = self.performance_tracker.metrics_history[-1]
        summary = self.performance_tracker.get_performance_summary()
        
        return {
            'timestamp': latest_metrics.timestamp.isoformat(),
            'cpu_usage': latest_metrics.cpu_usage,
            'memory_usage': latest_metrics.memory_usage,
            'disk_usage': latest_metrics.disk_usage,
            'process_count': latest_metrics.process_count,
            'thread_count': latest_metrics.thread_count,
            'statistics': summary['statistics'],
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'uptime': summary['statistics']['uptime']
        }
    
    def get_performance_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance history for specified hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = []
        for metrics in self.performance_tracker.metrics_history:
            if metrics.timestamp >= cutoff_time:
                history.append({
                    'timestamp': metrics.timestamp.isoformat(),
                    'cpu_usage': metrics.cpu_usage,
                    'memory_usage': metrics.memory_usage,
                    'process_count': metrics.process_count
                })
        
        return history
    
    def get_alerts(self, level: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        
        alerts = self.alert_manager.get_active_alerts(level)[:limit]
        
        return [
            {
                'id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level,
                'category': alert.category,
                'title': alert.title,
                'message': alert.message,
                'acknowledged': alert.acknowledged,
                'resolved': alert.resolved
            }
            for alert in alerts
        ]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        return self.alert_manager.acknowledge_alert(alert_id)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        return self.alert_manager.resolve_alert(alert_id)