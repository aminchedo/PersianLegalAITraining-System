import pytest
import tempfile
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_data():
    """Create sample training data."""
    return [
        {
            "question": "قانون مجازات اسلامی در مورد سرقت چیست؟",
            "answer": "سرقت در قانون مجازات اسلامی جرم محسوب می‌شود و مجازات آن حبس است.",
            "context": "قانون مجازات اسلامی",
            "category": "criminal_law"
        },
        {
            "question": "شرایط طلاق در ایران چیست؟",
            "answer": "طلاق در ایران باید در دادگاه خانواده انجام شود.",
            "context": "قانون خانواده",
            "category": "family_law"
        },
        {
            "question": "حقوق کارگر در ایران چیست؟",
            "answer": "کارگران در ایران حق دریافت حداقل دستمزد و بیمه دارند.",
            "context": "قانون کار",
            "category": "labor_law"
        }
    ]

@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return {
        "model_type": "DoRA",
        "base_model": "HooshvareLab/bert-base-parsbert-uncased",
        "num_labels": 2,
        "output_dir": "./models/legal_model",
        "data_dir": "./data/processed",
        "training": {
            "num_epochs": 1,  # Reduced for testing
            "batch_size": 2,  # Small batch for testing
            "learning_rate": 2e-5,
            "warmup_steps": 10,
            "save_steps": 100,
            "eval_steps": 50
        },
        "continuous_training": {
            "enabled": False,  # Disabled for testing
            "retrain_interval_hours": 24,
            "min_data_threshold": 1,
            "performance_threshold": 0.5
        },
        "monitoring": {
            "check_interval_minutes": 1,
            "max_memory_usage_percent": 95,
            "max_cpu_usage_percent": 95
        },
        "quantization": {
            "rank": 8,
            "dropout": 0.1,
            "alpha": 1.0,
            "quantization_bits": 8
        }
    }