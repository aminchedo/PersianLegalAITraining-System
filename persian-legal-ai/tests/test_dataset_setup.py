import pytest
import tempfile
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from datasets.setup_datasets import PersianLegalDatasetProcessor, download_and_prepare

class TestPersianLegalDatasetProcessor:
    """Test cases for PersianLegalDatasetProcessor."""
    
    def test_init(self):
        """Test processor initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = PersianLegalDatasetProcessor(temp_dir)
            assert processor.output_dir == Path(temp_dir)
            assert processor.stats == {}
    
    def test_normalize_text(self):
        """Test text normalization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = PersianLegalDatasetProcessor(temp_dir)
            
            # Test basic normalization
            text = "  سوال   قانونی   "
            normalized = processor.normalize_text(text)
            assert normalized == "سوال قانونی"
            
            # Test Persian character normalization
            text = "قانون مجازات اسلامی"
            normalized = processor.normalize_text(text)
            assert "ي" not in normalized  # Should be normalized to ی
            
            # Test empty text
            assert processor.normalize_text("") == ""
            assert processor.normalize_text(None) == ""
    
    def test_save_stats(self):
        """Test statistics saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = PersianLegalDatasetProcessor(temp_dir)
            processor.stats = {"test": "data"}
            processor.save_stats()
            
            stats_file = Path(temp_dir) / "stats.json"
            assert stats_file.exists()
            
            with open(stats_file, "r", encoding="utf-8") as f:
                saved_stats = json.load(f)
            assert saved_stats == {"test": "data"}

class TestDatasetDownload:
    """Test cases for dataset download and preparation."""
    
    def test_download_and_prepare_with_skip_flags(self):
        """Test download_and_prepare with skip flags."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with both datasets skipped
            stats = download_and_prepare(temp_dir, skip_perSets=True, skip_hamshahri=True)
            
            # Should still create stats file
            stats_file = Path(temp_dir) / "stats.json"
            assert stats_file.exists()
            
            # Stats should be empty or minimal
            assert isinstance(stats, dict)
    
    def test_download_and_prepare_output_structure(self):
        """Test that download_and_prepare creates proper output structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            stats = download_and_prepare(temp_dir, skip_perSets=True, skip_hamshahri=True)
            
            # Check that output directory exists
            output_path = Path(temp_dir)
            assert output_path.exists()
            
            # Check that stats.json exists
            stats_file = output_path / "stats.json"
            assert stats_file.exists()

if __name__ == "__main__":
    pytest.main([__file__])