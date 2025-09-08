#!/usr/bin/env python3
"""
Persian Legal AI Training System - Main Entry Point
Advanced 2025 implementation with DoRA, QR-Adaptor, and Intel CPU optimization
"""

import os
import sys
import asyncio
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any
import multiprocessing as mp

# Windows-specific imports
try:
    import win32event
    import win32api
    import win32con
    import wmi
    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False

# Core ML imports
import torch
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

import transformers
from transformers import AutoTokenizer, AutoModel
import numpy as np
from loguru import logger
import psutil
import cpuinfo

# Local imports
from config.training_config import TrainingConfig, WindowsOptimizationConfig
from models.dora_trainer import DoRATrainer
from models.qr_adaptor import QRAdaptor
from data.persian_legal_collector import PersianLegalDataCollector
from optimization.windows_cpu import WindowsCPUOptimizer
from utils.monitoring import SystemMonitor
from services.windows_service import WindowsServiceManager

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class PersianLegalAISystem:
    """
    Main orchestrator for the Persian Legal AI Training System
    Handles initialization, training coordination, and system management
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Persian Legal AI System with comprehensive setup"""
        self.config = TrainingConfig(config_path)
        self.windows_config = WindowsOptimizationConfig()
        
        # System components
        self.cpu_optimizer: Optional[WindowsCPUOptimizer] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.data_collector: Optional[PersianLegalDataCollector] = None
        self.dora_trainer: Optional[DoRATrainer] = None
        self.qr_adaptor: Optional[QRAdaptor] = None
        
        # System state
        self.is_initialized = False
        self.training_active = False
        self.shutdown_event = asyncio.Event()
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Persian Legal AI System initializing...")
        
    def _setup_logging(self) -> None:
        """Configure comprehensive logging system with Windows Event Log integration"""
        # Remove default logger
        logger.remove()
        
        # Console logging with colors
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
            colorize=True
        )
        
        # File logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger.add(
            log_dir / "persian_legal_ai_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="30 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            compression="zip"
        )
        
        # Windows Event Log integration
        if WINDOWS_AVAILABLE:
            try:
                logger.add(
                    self._windows_event_log_sink,
                    level="WARNING",
                    format="{level}: {message}"
                )
            except Exception as e:
                logger.warning(f"Could not setup Windows Event Log: {e}")
    
    def _windows_event_log_sink(self, message) -> None:
        """Custom sink for Windows Event Log integration"""
        if not WINDOWS_AVAILABLE:
            return
            
        try:
            import win32evtlog
            import win32evtlogutil
            
            event_id = 1000
            if "ERROR" in message.record["level"].name:
                event_type = win32evtlog.EVENTLOG_ERROR_TYPE
            elif "WARNING" in message.record["level"].name:
                event_type = win32evtlog.EVENTLOG_WARNING_TYPE
            else:
                event_type = win32evtlog.EVENTLOG_INFORMATION_TYPE
            
            win32evtlogutil.ReportEvent(
                "Persian Legal AI",
                event_id,
                eventCategory=0,
                eventType=event_type,
                strings=[message.record["message"]]
            )
        except Exception:
            pass  # Fail silently for event log issues
    
    async def initialize_system(self) -> bool:
        """
        Initialize all system components with comprehensive error handling
        Returns True if successful, False otherwise
        """
        try:
            logger.info("Starting system initialization...")
            
            # 1. Initialize CPU optimization first
            await self._initialize_cpu_optimization()
            
            # 2. Initialize system monitoring
            await self._initialize_monitoring()
            
            # 3. Initialize data collection system
            await self._initialize_data_collector()
            
            # 4. Initialize training models
            await self._initialize_training_models()
            
            # 5. Verify system readiness
            if await self._verify_system_readiness():
                self.is_initialized = True
                logger.success("Persian Legal AI System initialized successfully!")
                return True
            else:
                logger.error("System initialization failed verification")
                return False
                
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    async def _initialize_cpu_optimization(self) -> None:
        """Initialize Windows CPU optimization with Intel Extension"""
        logger.info("Initializing CPU optimization...")
        
        # Check system capabilities
        cpu_info = cpuinfo.get_cpu_info()
        logger.info(f"CPU: {cpu_info.get('brand_raw', 'Unknown')}")
        logger.info(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        
        # Initialize Intel Extension for PyTorch if available
        if IPEX_AVAILABLE:
            logger.info("Intel Extension for PyTorch detected - enabling optimizations")
            
            # Configure Intel optimizations
            torch.backends.mkldnn.enabled = True
            torch.backends.mkldnn.verbose = 0
            
            # Set optimal thread configuration for 24-core CPU
            torch.set_num_threads(self.config.cpu_threads)
            torch.set_num_interop_threads(self.config.interop_threads)
            
            # Enable Intel Extension optimizations
            ipex.optimize(level="O1", auto_kernel_selection=True)
            
            logger.success("Intel Extension for PyTorch optimizations enabled")
        else:
            logger.warning("Intel Extension for PyTorch not available - using standard PyTorch")
        
        # Initialize Windows-specific CPU optimizer
        self.cpu_optimizer = WindowsCPUOptimizer(self.windows_config)
        await self.cpu_optimizer.initialize()
        
        logger.success("CPU optimization initialized")
    
    async def _initialize_monitoring(self) -> None:
        """Initialize comprehensive system monitoring"""
        logger.info("Initializing system monitoring...")
        
        self.system_monitor = SystemMonitor(
            update_interval=self.config.monitoring_interval,
            log_metrics=True,
            enable_alerts=True
        )
        
        await self.system_monitor.start()
        logger.success("System monitoring initialized")
    
    async def _initialize_data_collector(self) -> None:
        """Initialize Persian legal data collection system"""
        logger.info("Initializing Persian legal data collector...")
        
        self.data_collector = PersianLegalDataCollector(
            cache_dir=Path(self.config.data_cache_dir),
            max_workers=self.config.data_collection_workers,
            enable_caching=True
        )
        
        # Test connectivity to data sources
        connectivity_status = await self.data_collector.test_connectivity()
        
        available_sources = sum(1 for status in connectivity_status.values() if status)
        total_sources = len(connectivity_status)
        
        logger.info(f"Data sources available: {available_sources}/{total_sources}")
        
        if available_sources == 0:
            logger.error("No data sources available - check internet connectivity")
            raise RuntimeError("No data sources available")
        
        logger.success("Persian legal data collector initialized")
    
    async def _initialize_training_models(self) -> None:
        """Initialize DoRA trainer and QR-Adaptor models"""
        logger.info("Initializing training models...")
        
        # Initialize DoRA trainer
        self.dora_trainer = DoRATrainer(
            model_name=self.config.base_model_name,
            rank=self.config.dora_rank,
            alpha=self.config.dora_alpha,
            target_modules=self.config.target_modules,
            enable_decomposition=True,
            cpu_optimization=True
        )
        
        # Initialize QR-Adaptor
        self.qr_adaptor = QRAdaptor(
            base_model=self.config.base_model_name,
            quantization_bits=self.config.quantization_bits,
            adaptive_rank=True,
            optimization_target="cpu"
        )
        
        # Load base models
        logger.info("Loading Persian language models...")
        
        try:
            # Load primary Persian model
            await self._load_persian_models()
            logger.success("Persian language models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Persian models: {e}")
            raise
        
        logger.success("Training models initialized")
    
    async def _load_persian_models(self) -> None:
        """Load verified Persian language models"""
        models_to_load = [
            "universitytehran/PersianMind-v1.0",
            "HooshvareLab/bert-base-parsbert-uncased",
            "myrkur/sentence-transformer-parsbert-fa-2.0",
            "mansoorhamidzadeh/parsbert-persian-QA",
            "HooshvareLab/bert-base-parsbert-ner-uncased"
        ]
        
        loaded_models = {}
        
        for model_name in models_to_load:
            try:
                logger.info(f"Loading model: {model_name}")
                
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.config.model_cache_dir,
                    trust_remote_code=True
                )
                
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=self.config.model_cache_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,  # CPU optimization
                    low_cpu_mem_usage=True
                )
                
                # Apply Intel optimizations if available
                if IPEX_AVAILABLE:
                    model = ipex.optimize(model, level="O1")
                
                loaded_models[model_name] = {
                    'tokenizer': tokenizer,
                    'model': model
                }
                
                logger.success(f"Model loaded: {model_name}")
                
            except Exception as e:
                logger.warning(f"Could not load model {model_name}: {e}")
                continue
        
        if not loaded_models:
            raise RuntimeError("No Persian models could be loaded")
        
        # Store loaded models in trainers
        self.dora_trainer.loaded_models = loaded_models
        self.qr_adaptor.loaded_models = loaded_models
        
        logger.info(f"Successfully loaded {len(loaded_models)} Persian models")
    
    async def _verify_system_readiness(self) -> bool:
        """Verify all system components are ready for operation"""
        logger.info("Verifying system readiness...")
        
        checks = {
            "CPU Optimizer": self.cpu_optimizer is not None,
            "System Monitor": self.system_monitor is not None and self.system_monitor.is_running,
            "Data Collector": self.data_collector is not None,
            "DoRA Trainer": self.dora_trainer is not None,
            "QR-Adaptor": self.qr_adaptor is not None,
            "Persian Models": hasattr(self.dora_trainer, 'loaded_models') and len(self.dora_trainer.loaded_models) > 0
        }
        
        all_ready = True
        for component, status in checks.items():
            if status:
                logger.success(f"✓ {component}: Ready")
            else:
                logger.error(f"✗ {component}: Not ready")
                all_ready = False
        
        return all_ready
    
    async def start_training(self, training_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start the Persian Legal AI training process
        Returns True if training started successfully
        """
        if not self.is_initialized:
            logger.error("System not initialized - call initialize_system() first")
            return False
        
        if self.training_active:
            logger.warning("Training already active")
            return False
        
        try:
            logger.info("Starting Persian Legal AI training...")
            
            # Merge training parameters
            params = self.config.training_params.copy()
            if training_params:
                params.update(training_params)
            
            # Start data collection in background
            collection_task = asyncio.create_task(
                self.data_collector.start_collection()
            )
            
            # Start DoRA training
            dora_task = asyncio.create_task(
                self.dora_trainer.start_training(params)
            )
            
            # Start QR-Adaptor optimization
            qr_task = asyncio.create_task(
                self.qr_adaptor.start_optimization(params)
            )
            
            self.training_active = True
            
            # Monitor training progress
            await self._monitor_training_progress([collection_task, dora_task, qr_task])
            
            logger.success("Training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
        finally:
            self.training_active = False
    
    async def _monitor_training_progress(self, tasks: list) -> None:
        """Monitor training progress and handle errors"""
        try:
            # Wait for all tasks to complete or until shutdown
            done, pending = await asyncio.wait(
                tasks + [asyncio.create_task(self.shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks if shutdown requested
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested - cancelling training tasks...")
                for task in pending:
                    task.cancel()
                    
                # Wait for cancellation to complete
                await asyncio.gather(*pending, return_exceptions=True)
                
        except Exception as e:
            logger.error(f"Error monitoring training progress: {e}")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the Persian Legal AI system"""
        logger.info("Shutting down Persian Legal AI system...")
        
        # Signal shutdown to all components
        self.shutdown_event.set()
        
        # Stop system monitoring
        if self.system_monitor:
            await self.system_monitor.stop()
        
        # Stop data collection
        if self.data_collector:
            await self.data_collector.stop()
        
        # Save model checkpoints
        if self.dora_trainer and self.training_active:
            await self.dora_trainer.save_checkpoint()
        
        if self.qr_adaptor and self.training_active:
            await self.qr_adaptor.save_checkpoint()
        
        # Cleanup CPU optimizer
        if self.cpu_optimizer:
            await self.cpu_optimizer.cleanup()
        
        logger.success("Persian Legal AI system shutdown complete")

async def main():
    """Main entry point for the Persian Legal AI system"""
    
    # Initialize system
    ai_system = PersianLegalAISystem()
    
    try:
        # Initialize all components
        if not await ai_system.initialize_system():
            logger.error("Failed to initialize system")
            return 1
        
        # Start the training process
        success = await ai_system.start_training()
        
        if not success:
            logger.error("Training failed")
            return 1
        
        logger.success("Persian Legal AI training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        await ai_system.shutdown()

def run_as_service():
    """Run the system as a Windows service"""
    if not WINDOWS_AVAILABLE:
        logger.error("Windows service mode requires Windows OS")
        return 1
    
    service_manager = WindowsServiceManager()
    return service_manager.run()

if __name__ == "__main__":
    # Set multiprocessing start method for Windows compatibility
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    # Check if running as Windows service
    if len(sys.argv) > 1 and sys.argv[1] in ['service', 'install', 'remove', 'start', 'stop']:
        exit_code = run_as_service()
    else:
        # Run as regular application
        exit_code = asyncio.run(main())
    
    sys.exit(exit_code)