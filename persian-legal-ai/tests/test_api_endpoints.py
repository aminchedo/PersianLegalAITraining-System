import pytest
import tempfile
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backend.main import app

class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
    
    @patch('backend.api.training_endpoints.get_training_orchestrator')
    def test_start_training_endpoint(self, mock_get_orchestrator, client):
        """Test start training endpoint."""
        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_get_orchestrator.return_value = mock_orchestrator
        
        # Test request
        request_data = {
            "model_type": "DoRA",
            "continuous": False
        }
        
        response = client.post("/training/start", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "started"
        assert "training_id" in data
    
    @patch('backend.api.training_endpoints.get_training_orchestrator')
    def test_start_training_invalid_model_type(self, mock_get_orchestrator, client):
        """Test start training with invalid model type."""
        mock_orchestrator = Mock()
        mock_get_orchestrator.return_value = mock_orchestrator
        
        request_data = {
            "model_type": "InvalidModel",
            "continuous": False
        }
        
        response = client.post("/training/start", json=request_data)
        assert response.status_code == 400
    
    @patch('backend.api.training_endpoints.get_training_orchestrator')
    def test_stop_training_endpoint(self, mock_get_orchestrator, client):
        """Test stop training endpoint."""
        mock_orchestrator = Mock()
        mock_get_orchestrator.return_value = mock_orchestrator
        
        response = client.post("/training/stop")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "stopped"
    
    @patch('backend.api.training_endpoints.get_training_orchestrator')
    def test_training_status_endpoint(self, mock_get_orchestrator, client):
        """Test training status endpoint."""
        mock_orchestrator = Mock()
        mock_orchestrator.get_status.return_value = {
            "is_training": False,
            "training_stats": {},
            "system_resources": {},
            "config": {}
        }
        mock_get_orchestrator.return_value = mock_orchestrator
        
        response = client.get("/training/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
    
    @patch('backend.api.training_endpoints.get_training_orchestrator')
    def test_list_models_endpoint(self, mock_get_orchestrator, client):
        """Test list models endpoint."""
        mock_orchestrator = Mock()
        mock_orchestrator.config = {"output_dir": "/tmp"}
        mock_get_orchestrator.return_value = mock_orchestrator
        
        with patch('pathlib.Path.exists', return_value=False):
            response = client.get("/training/models")
            assert response.status_code == 200
            
            data = response.json()
            assert isinstance(data, list)
    
    @patch('backend.api.training_endpoints.get_training_orchestrator')
    def test_get_training_config_endpoint(self, mock_get_orchestrator, client):
        """Test get training config endpoint."""
        mock_orchestrator = Mock()
        mock_orchestrator.config = {"model_type": "DoRA"}
        mock_get_orchestrator.return_value = mock_orchestrator
        
        response = client.get("/training/config")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "config" in data
    
    @patch('backend.api.training_endpoints.get_training_orchestrator')
    def test_update_training_config_endpoint(self, mock_get_orchestrator, client):
        """Test update training config endpoint."""
        mock_orchestrator = Mock()
        mock_orchestrator.config = {"model_type": "DoRA"}
        mock_get_orchestrator.return_value = mock_orchestrator
        
        config_update = {"model_type": "QR-Adaptor"}
        
        response = client.put("/training/config", json=config_update)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
    
    def test_system_metrics_endpoint(self, client):
        """Test system metrics endpoint."""
        response = client.get("/system/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "cpu_usage" in data
        assert "memory_usage" in data
        assert "disk_usage" in data
        assert "timestamp" in data
    
    def test_system_health_endpoint(self, client):
        """Test system health endpoint."""
        response = client.get("/system/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "checks" in data
        assert "timestamp" in data
    
    def test_system_processes_endpoint(self, client):
        """Test system processes endpoint."""
        response = client.get("/system/processes")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_system_info_endpoint(self, client):
        """Test system info endpoint."""
        response = client.get("/system/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "platform" in data
        assert "architecture" in data
        assert "python_version" in data
        assert "torch_version" in data
        assert "cuda_available" in data
    
    def test_system_performance_endpoint(self, client):
        """Test system performance endpoint."""
        response = client.get("/system/performance")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "system_metrics" in data
        assert "performance_score" in data
        assert "performance_level" in data
    
    def test_system_cleanup_endpoint(self, client):
        """Test system cleanup endpoint."""
        response = client.post("/system/cleanup")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "operations" in data

class TestAPIErrorHandling:
    """Test cases for API error handling."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_invalid_endpoint(self, client):
        """Test invalid endpoint returns 404."""
        response = client.get("/invalid/endpoint")
        assert response.status_code == 404
    
    def test_invalid_json_request(self, client):
        """Test invalid JSON request."""
        response = client.post("/training/start", data="invalid json")
        assert response.status_code == 422  # Unprocessable Entity
    
    @patch('backend.api.training_endpoints.get_training_orchestrator')
    def test_training_error_handling(self, mock_get_orchestrator, client):
        """Test training error handling."""
        mock_orchestrator = Mock()
        mock_orchestrator.start_training.side_effect = Exception("Training error")
        mock_get_orchestrator.return_value = mock_orchestrator
        
        request_data = {
            "model_type": "DoRA",
            "continuous": False
        }
        
        response = client.post("/training/start", json=request_data)
        assert response.status_code == 500

if __name__ == "__main__":
    pytest.main([__file__])