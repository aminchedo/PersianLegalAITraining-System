import pytest
import tempfile
import json
import time
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from training.train_legal_model import TrainingOrchestrator

class TestTrainingOrchestrator:
    """Test cases for TrainingOrchestrator."""
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = TrainingOrchestrator()
            
            assert orchestrator.config is not None
            assert orchestrator.is_training == False
            assert orchestrator.training_thread is None
            assert orchestrator.stop_event is not None
            assert orchestrator.current_model is None
            assert "total_training_cycles" in orchestrator.training_stats
    
    def test_load_config(self):
        """Test configuration loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with default config
            orchestrator = TrainingOrchestrator()
            
            assert "model_type" in orchestrator.config
            assert "base_model" in orchestrator.config
            assert "training" in orchestrator.config
            assert "continuous_training" in orchestrator.config
    
    def test_load_config_with_file(self):
        """Test configuration loading from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test config file
            config_data = {
                "model_type": "QR-Adaptor",
                "training": {
                    "num_epochs": 5,
                    "batch_size": 32
                }
            }
            
            config_file = Path(temp_dir) / "test_config.json"
            with open(config_file, "w") as f:
                json.dump(config_data, f)
            
            orchestrator = TrainingOrchestrator(str(config_file))
            
            assert orchestrator.config["model_type"] == "QR-Adaptor"
            assert orchestrator.config["training"]["num_epochs"] == 5
            assert orchestrator.config["training"]["batch_size"] == 32
    
    def test_check_system_resources(self):
        """Test system resource checking."""
        orchestrator = TrainingOrchestrator()
        
        # Mock psutil to return low usage
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value.percent = 50.0
            mock_cpu.return_value = 30.0
            
            result = orchestrator._check_system_resources()
            assert result == True
    
    def test_check_system_resources_high_usage(self):
        """Test system resource checking with high usage."""
        orchestrator = TrainingOrchestrator()
        
        # Mock psutil to return high usage
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value.percent = 95.0  # High memory usage
            mock_cpu.return_value = 30.0
            
            result = orchestrator._check_system_resources()
            assert result == False
    
    def test_should_retrain(self):
        """Test retrain decision logic."""
        orchestrator = TrainingOrchestrator()
        
        # Test with continuous training disabled
        orchestrator.config["continuous_training"]["enabled"] = False
        assert orchestrator._should_retrain() == False
        
        # Test with continuous training enabled but no previous training
        orchestrator.config["continuous_training"]["enabled"] = True
        orchestrator.training_stats["last_training_time"] = None
        assert orchestrator._should_retrain() == True
        
        # Test with recent training
        from datetime import datetime, timedelta
        recent_time = (datetime.now() - timedelta(hours=1)).isoformat()
        orchestrator.training_stats["last_training_time"] = recent_time
        orchestrator.config["continuous_training"]["retrain_interval_hours"] = 24
        assert orchestrator._should_retrain() == False
    
    def test_save_and_load_training_stats(self):
        """Test training statistics saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = TrainingOrchestrator()
            orchestrator.config["output_dir"] = temp_dir
            
            # Set some test stats
            orchestrator.training_stats = {
                "total_training_cycles": 5,
                "last_training_time": "2024-01-01T00:00:00",
                "best_model_path": "/path/to/best/model",
                "training_history": []
            }
            
            # Save stats
            orchestrator._save_training_stats()
            
            # Create new orchestrator and load stats
            new_orchestrator = TrainingOrchestrator()
            new_orchestrator.config["output_dir"] = temp_dir
            new_orchestrator._load_training_stats()
            
            assert new_orchestrator.training_stats["total_training_cycles"] == 5
            assert new_orchestrator.training_stats["last_training_time"] == "2024-01-01T00:00:00"
            assert new_orchestrator.training_stats["best_model_path"] == "/path/to/best/model"
    
    def test_get_status(self):
        """Test status retrieval."""
        orchestrator = TrainingOrchestrator()
        
        status = orchestrator.get_status()
        
        assert "is_training" in status
        assert "training_stats" in status
        assert "system_resources" in status
        assert "config" in status
        
        assert status["is_training"] == False
        assert "memory_percent" in status["system_resources"]
        assert "cpu_percent" in status["system_resources"]
        assert "disk_percent" in status["system_resources"]
    
    @patch('training.train_legal_model.DoRATrainer')
    def test_train_model_mock(self, mock_trainer_class):
        """Test model training with mocked trainer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = TrainingOrchestrator()
            orchestrator.config["output_dir"] = temp_dir
            orchestrator.config["data_dir"] = temp_dir
            
            # Mock trainer
            mock_trainer = Mock()
            mock_trainer.train.return_value = {
                "train_loss": 0.5,
                "validation_results": {"eval_f1": 0.8, "eval_accuracy": 0.85}
            }
            mock_trainer_class.return_value = mock_trainer
            
            # Create mock data directory
            data_dir = Path(temp_dir)
            data_dir.mkdir(exist_ok=True)
            
            # Mock data loading
            with patch.object(orchestrator, '_load_training_data') as mock_load_data:
                mock_load_data.return_value = (
                    [{"question": "test", "answer": "test", "context": "test", "category": "test"}],
                    [{"question": "test", "answer": "test", "context": "test", "category": "test"}],
                    None
                )
                
                # Mock system resource check
                with patch.object(orchestrator, '_check_system_resources', return_value=True):
                    result = orchestrator._train_model()
                    
                    assert result["status"] == "success"
                    assert "results" in result
                    assert orchestrator.training_stats["total_training_cycles"] == 1
    
    def test_start_and_stop_training(self):
        """Test training start and stop."""
        orchestrator = TrainingOrchestrator()
        
        # Test starting training
        assert orchestrator.is_training == False
        orchestrator.start_training(continuous=False)
        # Note: In a real test, you'd need to mock the training process
        
        # Test stopping training
        orchestrator.stop_training()
        assert orchestrator.is_training == False

if __name__ == "__main__":
    pytest.main([__file__])