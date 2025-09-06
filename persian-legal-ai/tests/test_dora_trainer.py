import pytest
import tempfile
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dora_trainer import DoRATrainer, LegalDataset, create_sample_data

class TestLegalDataset:
    """Test cases for LegalDataset."""
    
    def test_dataset_creation(self):
        """Test dataset creation with sample data."""
        from transformers import BertTokenizer
        
        # Create sample data
        train_data, val_data = create_sample_data()
        
        # Initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
        
        # Create dataset
        dataset = LegalDataset(train_data, tokenizer)
        
        assert len(dataset) == len(train_data)
        
        # Test getting an item
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        
        # Check tensor shapes
        assert item["input_ids"].shape == torch.Size([512])  # max_length
        assert item["attention_mask"].shape == torch.Size([512])
        assert isinstance(item["labels"], torch.Tensor)

class TestDoRATrainer:
    """Test cases for DoRATrainer."""
    
    def test_trainer_initialization(self):
        """Test DoRA trainer initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = DoRATrainer()
            
            assert trainer.model_name == "HooshvareLab/bert-base-parsbert-uncased"
            assert trainer.num_labels == 2
            assert trainer.tokenizer is not None
            assert trainer.model is not None
    
    def test_prepare_datasets(self):
        """Test dataset preparation."""
        trainer = DoRATrainer()
        train_data, val_data = create_sample_data()
        
        train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(
            train_data, val_data, None
        )
        
        assert isinstance(train_dataset, LegalDataset)
        assert isinstance(val_dataset, LegalDataset)
        assert test_dataset is None
        assert len(train_dataset) == len(train_data)
        assert len(val_dataset) == len(val_data)
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        trainer = DoRATrainer()
        
        # Create mock predictions and labels
        import numpy as np
        predictions = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        labels = np.array([0, 1, 0])
        
        metrics = trainer.compute_metrics((predictions, labels))
        
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        
        # All metrics should be between 0 and 1
        for metric_name, metric_value in metrics.items():
            assert 0 <= metric_value <= 1
    
    def test_predict(self):
        """Test prediction functionality."""
        trainer = DoRATrainer()
        
        # Test with sample texts
        texts = [
            "قانون مجازات اسلامی چیست؟",
            "شرایط طلاق در ایران چیست؟"
        ]
        
        predictions = trainer.predict(texts, batch_size=1)
        
        assert len(predictions) == len(texts)
        
        for pred in predictions:
            assert "text" in pred
            assert "predicted_label" in pred
            assert "confidence" in pred
            assert "probabilities" in pred
            assert isinstance(pred["predicted_label"], int)
            assert 0 <= pred["confidence"] <= 1

class TestSampleData:
    """Test cases for sample data creation."""
    
    def test_create_sample_data(self):
        """Test sample data creation."""
        train_data, val_data = create_sample_data()
        
        assert isinstance(train_data, list)
        assert isinstance(val_data, list)
        assert len(train_data) > 0
        assert len(val_data) > 0
        
        # Check data structure
        for item in train_data + val_data:
            assert "question" in item
            assert "answer" in item
            assert "context" in item
            assert "category" in item
            
            # Check that text is in Persian
            assert any('\u0600' <= char <= '\u06FF' for char in item["question"])
            assert any('\u0600' <= char <= '\u06FF' for char in item["answer"])

if __name__ == "__main__":
    pytest.main([__file__])