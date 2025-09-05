import os
import sys
import json
import logging
import time
import signal
import threading
from pathlib import Path
from typing import Dict, List, Optional, Union
import argparse
from datetime import datetime, timedelta
import psutil
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dora_trainer import DoRATrainer
from models.qr_adaptor import QRAdaptorTrainer
from datasets.setup_datasets import PersianLegalDatasetProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingOrchestrator:
    """Orchestrates continuous training with 24/7 capabilities."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.is_training = False
        self.training_thread = None
        self.stop_event = threading.Event()
        self.current_model = None
        self.training_stats = {
            "total_training_cycles": 0,
            "last_training_time": None,
            "best_model_path": None,
            "training_history": []
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Training Orchestrator initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load training configuration."""
        default_config = {
            "model_type": "DoRA",  # or "QR-Adaptor"
            "base_model": "HooshvareLab/bert-base-parsbert-uncased",
            "num_labels": 2,
            "output_dir": "./models/legal_model",
            "data_dir": "./data/processed",
            "training": {
                "num_epochs": 3,
                "batch_size": 16,
                "learning_rate": 2e-5,
                "warmup_steps": 100,
                "save_steps": 500,
                "eval_steps": 100,
                "gradient_accumulation_steps": 1
            },
            "continuous_training": {
                "enabled": True,
                "retrain_interval_hours": 24,
                "min_data_threshold": 100,
                "performance_threshold": 0.8
            },
            "monitoring": {
                "check_interval_minutes": 30,
                "max_memory_usage_percent": 85,
                "max_cpu_usage_percent": 90
            },
            "quantization": {
                "rank": 16,
                "dropout": 0.1,
                "alpha": 1.0,
                "quantization_bits": 8
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop_training()
        sys.exit(0)
    
    def _check_system_resources(self) -> bool:
        """Check if system resources are sufficient for training."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if memory.percent > self.config["monitoring"]["max_memory_usage_percent"]:
            logger.warning(f"Memory usage too high: {memory.percent}%")
            return False
        
        if cpu_percent > self.config["monitoring"]["max_cpu_usage_percent"]:
            logger.warning(f"CPU usage too high: {cpu_percent}%")
            return False
        
        return True
    
    def _load_training_data(self) -> tuple:
        """Load and prepare training data."""
        data_dir = Path(self.config["data_dir"])
        
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return None, None, None
        
        # Try to load combined dataset first
        combined_path = data_dir / "combined_processed"
        if combined_path.exists():
            from datasets import load_from_disk
            dataset = load_from_disk(str(combined_path))
            
            train_data = dataset.get("train", [])
            val_data = dataset.get("validation", [])
            test_data = dataset.get("test", [])
            
            logger.info(f"Loaded combined dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
            return train_data, val_data, test_data
        
        # Fallback to individual datasets
        persets_path = data_dir / "persets_processed"
        hamshahri_path = data_dir / "hamshahri_processed"
        
        train_data, val_data, test_data = [], [], []
        
        if persets_path.exists():
            from datasets import load_from_disk
            persets_ds = load_from_disk(str(persets_path))
            train_data.extend(persets_ds.get("train", []))
            val_data.extend(persets_ds.get("validation", []))
            test_data.extend(persets_ds.get("test", []))
        
        if hamshahri_path.exists():
            from datasets import load_from_disk
            hamshahri_ds = load_from_disk(str(hamshahri_path))
            train_data.extend(hamshahri_ds.get("train", []))
            val_data.extend(hamshahri_ds.get("validation", []))
            test_data.extend(hamshahri_ds.get("test", []))
        
        if not train_data:
            logger.error("No training data found")
            return None, None, None
        
        logger.info(f"Loaded datasets: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        return train_data, val_data, test_data
    
    def _create_trainer(self) -> Union[DoRATrainer, QRAdaptorTrainer]:
        """Create the appropriate trainer based on configuration."""
        model_type = self.config["model_type"]
        
        if model_type == "DoRA":
            trainer = DoRATrainer(
                model_name=self.config["base_model"],
                num_labels=self.config["num_labels"]
            )
        elif model_type == "QR-Adaptor":
            trainer = QRAdaptorTrainer(
                model_name=self.config["base_model"],
                num_labels=self.config["num_labels"],
                quantize_config=self.config["quantization"]
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return trainer
    
    def _train_model(self) -> Dict:
        """Train the model with current configuration."""
        logger.info("Starting model training...")
        
        # Check system resources
        if not self._check_system_resources():
            logger.warning("System resources insufficient, skipping training")
            return {"status": "skipped", "reason": "insufficient_resources"}
        
        # Load training data
        train_data, val_data, test_data = self._load_training_data()
        if not train_data:
            logger.error("No training data available")
            return {"status": "failed", "reason": "no_data"}
        
        # Check minimum data threshold
        if len(train_data) < self.config["continuous_training"]["min_data_threshold"]:
            logger.warning(f"Training data below threshold: {len(train_data)} < {self.config['continuous_training']['min_data_threshold']}")
            return {"status": "skipped", "reason": "insufficient_data"}
        
        try:
            # Create trainer
            trainer = self._create_trainer()
            
            # Prepare output directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config["output_dir"]) / f"{self.config['model_type']}_{timestamp}"
            
            # Train the model
            training_config = self.config["training"]
            results = trainer.train(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                output_dir=str(output_dir),
                num_epochs=training_config["num_epochs"],
                batch_size=training_config["batch_size"],
                learning_rate=training_config["learning_rate"],
                warmup_steps=training_config["warmup_steps"],
                save_steps=training_config["save_steps"],
                eval_steps=training_config["eval_steps"]
            )
            
            # Update training stats
            self.training_stats["total_training_cycles"] += 1
            self.training_stats["last_training_time"] = datetime.now().isoformat()
            
            # Check if this is the best model
            val_f1 = results.get("validation_results", {}).get("eval_f1", 0)
            if val_f1 > self.config["continuous_training"]["performance_threshold"]:
                self.training_stats["best_model_path"] = str(output_dir)
                logger.info(f"New best model saved: {output_dir} (F1: {val_f1})")
            
            # Record training history
            self.training_stats["training_history"].append({
                "timestamp": datetime.now().isoformat(),
                "model_path": str(output_dir),
                "validation_f1": val_f1,
                "validation_accuracy": results.get("validation_results", {}).get("eval_accuracy", 0),
                "training_loss": results.get("train_loss", 0)
            })
            
            # Keep only last 10 training records
            if len(self.training_stats["training_history"]) > 10:
                self.training_stats["training_history"] = self.training_stats["training_history"][-10:]
            
            # Save training stats
            self._save_training_stats()
            
            logger.info("Model training completed successfully!")
            return {"status": "success", "results": results}
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {"status": "failed", "reason": str(e)}
    
    def _save_training_stats(self):
        """Save training statistics to file."""
        stats_path = Path(self.config["output_dir"]) / "training_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
    
    def _load_training_stats(self):
        """Load training statistics from file."""
        stats_path = Path(self.config["output_dir"]) / "training_stats.json"
        
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.training_stats = json.load(f)
    
    def _should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if not self.config["continuous_training"]["enabled"]:
            return False
        
        # Check if enough time has passed
        if self.training_stats["last_training_time"]:
            last_training = datetime.fromisoformat(self.training_stats["last_training_time"])
            interval = timedelta(hours=self.config["continuous_training"]["retrain_interval_hours"])
            
            if datetime.now() - last_training < interval:
                return False
        
        return True
    
    def _training_loop(self):
        """Main training loop for continuous training."""
        logger.info("Starting continuous training loop...")
        
        while not self.stop_event.is_set():
            try:
                if self._should_retrain():
                    logger.info("Starting scheduled retraining...")
                    result = self._train_model()
                    logger.info(f"Retraining result: {result['status']}")
                
                # Wait for next check
                check_interval = self.config["monitoring"]["check_interval_minutes"] * 60
                if self.stop_event.wait(check_interval):
                    break
                    
            except Exception as e:
                logger.error(f"Error in training loop: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying
        
        logger.info("Training loop stopped")
    
    def start_training(self, continuous: bool = False):
        """Start training process."""
        if self.is_training:
            logger.warning("Training already in progress")
            return
        
        self.is_training = True
        self.stop_event.clear()
        
        if continuous:
            # Start continuous training in background thread
            self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
            self.training_thread.start()
            logger.info("Continuous training started")
        else:
            # Single training run
            result = self._train_model()
            self.is_training = False
            return result
    
    def stop_training(self):
        """Stop training process."""
        if not self.is_training:
            return
        
        logger.info("Stopping training...")
        self.stop_event.set()
        
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=30)
        
        self.is_training = False
        logger.info("Training stopped")
    
    def get_status(self) -> Dict:
        """Get current training status."""
        return {
            "is_training": self.is_training,
            "training_stats": self.training_stats,
            "system_resources": {
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent(),
                "disk_percent": psutil.disk_usage('/').percent
            },
            "config": self.config
        }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Persian Legal AI Training System")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model-type", choices=["DoRA", "QR-Adaptor"], 
                       help="Model type to train")
    parser.add_argument("--continuous", action="store_true", 
                       help="Enable continuous training")
    parser.add_argument("--data-dir", type=str, help="Path to training data directory")
    parser.add_argument("--output-dir", type=str, help="Path to output directory")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(args.config)
    
    # Override config with command line arguments
    if args.model_type:
        orchestrator.config["model_type"] = args.model_type
    if args.data_dir:
        orchestrator.config["data_dir"] = args.data_dir
    if args.output_dir:
        orchestrator.config["output_dir"] = args.output_dir
    if args.epochs:
        orchestrator.config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        orchestrator.config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        orchestrator.config["training"]["learning_rate"] = args.learning_rate
    
    # Load existing training stats
    orchestrator._load_training_stats()
    
    try:
        if args.continuous:
            logger.info("Starting continuous training mode...")
            orchestrator.start_training(continuous=True)
            
            # Keep main thread alive
            while True:
                time.sleep(60)
                status = orchestrator.get_status()
                logger.info(f"Status: Training={status['is_training']}, "
                          f"Memory={status['system_resources']['memory_percent']:.1f}%, "
                          f"CPU={status['system_resources']['cpu_percent']:.1f}%")
        else:
            logger.info("Starting single training run...")
            result = orchestrator.start_training(continuous=False)
            logger.info(f"Training completed: {result}")
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        orchestrator.stop_training()
        logger.info("Training system shutdown complete")

if __name__ == "__main__":
    main()