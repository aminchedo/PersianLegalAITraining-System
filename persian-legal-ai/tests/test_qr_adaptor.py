import pytest
import tempfile
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.qr_adaptor import QRAdaptorTrainer, QRAdaptorLayer, create_sample_data

class TestQRAdaptorLayer:
    """Test cases for QRAdaptorLayer."""
    
    def test_layer_initialization(self):
        """Test QR-Adaptor layer initialization."""
        layer = QRAdaptorLayer(in_features=768, out_features=768, rank=16)
        
        assert layer.rank == 16
        assert layer.Q.shape == (768, 16)
        assert layer.R.shape == (16, 768)
        assert layer.alpha == 1.0
    
    def test_layer_forward(self):
        """Test QR-Adaptor layer forward pass."""
        layer = QRAdaptorLayer(in_features=768, out_features=768, rank=16)
        
        # Create input tensor
        batch_size = 2
        seq_len = 128
        input_tensor = torch.randn(batch_size, seq_len, 768)
        
        # Forward pass
        output = layer(input_tensor)
        
        assert output.shape == input_tensor.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestQRAdaptorTrainer:
    """Test cases for QRAdaptorTrainer."""
    
    def test_trainer_initialization(self):
        """Test QR-Adaptor trainer initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            quantize_config = {
                'rank': 8,
                'dropout': 0.1,
                'alpha': 1.0,
                'quantization_bits': 8
            }
            
            trainer = QRAdaptorTrainer(quantize_config=quantize_config)
            
            assert trainer.model_name == "HooshvareLab/bert-base-parsbert-uncased"
            assert trainer.num_labels == 2
            assert trainer.tokenizer is not None
            assert trainer.model is not None
            assert trainer.quantize_config == quantize_config
    
    def test_prepare_datasets(self):
        """Test dataset preparation."""
        trainer = QRAdaptorTrainer()
        train_data, val_data = create_sample_data()
        
        train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(
            train_data, val_data, None
        )
        
        assert train_dataset is not None
        assert val_dataset is not None
        assert test_dataset is None
        assert len(train_dataset) == len(train_data)
        assert len(val_dataset) == len(val_data)
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        trainer = QRAdaptorTrainer()
        
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
    
    def test_get_model_size(self):
        """Test model size calculation."""
        trainer = QRAdaptorTrainer()
        
        size_info = trainer.get_model_size()
        
        assert "total_parameters" in size_info
        assert "trainable_parameters" in size_info
        assert "model_size_mb" in size_info
        assert "compression_ratio" in size_info
        
        assert size_info["total_parameters"] > 0
        assert size_info["trainable_parameters"] > 0
        assert size_info["model_size_mb"] > 0
        assert size_info["compression_ratio"] > 0
    
    def test_predict(self):
        """Test prediction functionality."""
        trainer = QRAdaptorTrainer()
        
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

class TestQuantization:
    """Test cases for quantization functionality."""
    
    def test_quantize_model(self):
        """Test model quantization."""
        trainer = QRAdaptorTrainer()
        
        # Check that model has been quantized
        # This is a basic check - in practice, you'd verify specific quantization
        assert trainer.model is not None
        
        # Test that model can perform forward pass
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
        
        test_text = "قانون مجازات اسلامی چیست؟"
        encoding = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = trainer.model(**encoding)
            assert outputs.logits is not None
            assert outputs.logits.shape[1] == trainer.num_labels

if __name__ == "__main__":
    pytest.main([__file__])