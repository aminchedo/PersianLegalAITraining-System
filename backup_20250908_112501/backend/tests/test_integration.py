"""
Integration Tests for Persian Legal AI System
تست‌های یکپارچگی برای سیستم هوش مصنوعی حقوقی فارسی
"""

import pytest
import asyncio
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

# Import the main app
from main import app

client = TestClient(app)

class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""
    
    def test_complete_training_workflow(self):
        """Test complete training workflow from frontend to backend"""
        # Step 1: Login
        login_response = client.post("/api/auth/login", json={
            "username": "trainer",
            "password": "trainer123"
        })
        
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Step 2: Check system health
        health_response = client.get("/api/system/health", headers=headers)
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["overall_health"] in ["healthy", "degraded", "unhealthy"]
        
        # Step 3: Create training session
        training_request = {
            "model_type": "dora",
            "model_name": "integration_test_model",
            "config": {
                "learning_rate": 0.001,
                "batch_size": 4,
                "epochs": 1,
                "max_length": 256
            },
            "data_source": "sample",
            "task_type": "text_classification"
        }
        
        create_response = client.post("/api/training/sessions", 
            json=training_request, 
            headers=headers
        )
        
        assert create_response.status_code == 200
        session_data = create_response.json()
        session_id = session_data["session_id"]
        assert session_data["status"] == "pending"
        
        # Step 4: Check training session status
        status_response = client.get(f"/api/training/sessions/{session_id}", headers=headers)
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["session_id"] == session_id
        
        # Step 5: Get training sessions list
        sessions_response = client.get("/api/training/sessions", headers=headers)
        assert sessions_response.status_code == 200
        sessions_list = sessions_response.json()
        assert len(sessions_list) >= 1
        assert any(session["session_id"] == session_id for session in sessions_list)
        
        # Step 6: Get training metrics (if available)
        metrics_response = client.get(f"/api/training/sessions/{session_id}/metrics", headers=headers)
        # Should return 200 even if no metrics yet
        assert metrics_response.status_code == 200
        
        # Step 7: Get training logs
        logs_response = client.get(f"/api/training/sessions/{session_id}/logs", headers=headers)
        assert logs_response.status_code == 200
        logs_data = logs_response.json()
        assert isinstance(logs_data, list)
        
        # Step 8: Stop training session (if running)
        stop_response = client.post(f"/api/training/sessions/{session_id}/stop", headers=headers)
        assert stop_response.status_code == 200
        
        # Step 9: Delete training session
        delete_response = client.delete(f"/api/training/sessions/{session_id}", headers=headers)
        assert delete_response.status_code == 200
        
        # Step 10: Verify session is deleted
        final_status_response = client.get(f"/api/training/sessions/{session_id}", headers=headers)
        assert final_status_response.status_code == 404
    
    def test_verified_training_workflow(self):
        """Test verified training workflow"""
        # Login
        login_response = client.post("/api/auth/login", json={
            "username": "trainer",
            "password": "trainer123"
        })
        
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Get verified datasets
        datasets_response = client.get("/api/training/datasets/verified", headers=headers)
        assert datasets_response.status_code == 200
        datasets = datasets_response.json()
        assert isinstance(datasets, list)
        
        # Start verified training
        training_request = {
            "model_type": "dora",
            "model_name": "verified_test_model",
            "config": {
                "learning_rate": 0.001,
                "batch_size": 4,
                "epochs": 1
            },
            "data_source": "sample",
            "task_type": "text_classification"
        }
        
        create_response = client.post("/api/training/sessions/verified", 
            json=training_request, 
            headers=headers
        )
        
        assert create_response.status_code == 200
        session_data = create_response.json()
        session_id = session_data["session_id"]
        
        # Check verified training status
        status_response = client.get(f"/api/training/sessions/verified/{session_id}/status", headers=headers)
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["session_id"] == session_id
        
        # Get verified training sessions
        sessions_response = client.get("/api/training/sessions/verified", headers=headers)
        assert sessions_response.status_code == 200
        sessions_list = sessions_response.json()
        assert isinstance(sessions_list, list)
        
        # Cancel verified training
        cancel_response = client.delete(f"/api/training/sessions/verified/{session_id}", headers=headers)
        assert cancel_response.status_code == 200

class TestDatabaseIntegration:
    """Test database integration"""
    
    def test_database_connection_health(self):
        """Test database connection in health check"""
        response = client.get("/api/system/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert "checks" in health_data
        assert "database" in health_data["checks"]
        
        database_health = health_data["checks"]["database"]
        assert "status" in database_health
        assert "connection_time" in database_health
        assert "query_time" in database_health
    
    def test_database_persistence(self):
        """Test database persistence across requests"""
        # Login and create a training session
        login_response = client.post("/api/auth/login", json={
            "username": "trainer",
            "password": "trainer123"
        })
        
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Create training session
        training_request = {
            "model_type": "dora",
            "model_name": "persistence_test_model",
            "config": {"learning_rate": 0.001},
            "data_source": "sample",
            "task_type": "text_classification"
        }
        
        create_response = client.post("/api/training/sessions", 
            json=training_request, 
            headers=headers
        )
        
        assert create_response.status_code == 200
        session_id = create_response.json()["session_id"]
        
        # Wait a moment
        time.sleep(1)
        
        # Check if session persists
        status_response = client.get(f"/api/training/sessions/{session_id}", headers=headers)
        assert status_response.status_code == 200
        
        # Clean up
        client.delete(f"/api/training/sessions/{session_id}", headers=headers)

class TestWebSocketIntegration:
    """Test WebSocket integration"""
    
    def test_websocket_connection(self):
        """Test WebSocket connection"""
        with client.websocket_connect("/ws") as websocket:
            # Should be able to connect
            assert websocket is not None
            
            # Should receive system metrics
            data = websocket.receive_json()
            assert "type" in data
            assert data["type"] == "system_metrics"
            assert "timestamp" in data

class TestMultiGPUIntegration:
    """Test multi-GPU integration"""
    
    @patch('backend.training.multi_gpu_trainer.MultiGPUTrainer')
    def test_multi_gpu_training_setup(self, mock_trainer):
        """Test multi-GPU training setup"""
        # Mock the trainer
        mock_instance = MagicMock()
        mock_trainer.return_value = mock_instance
        mock_instance.get_training_status.return_value = {
            "is_training": False,
            "world_size": 2,
            "gpu_count": 2,
            "current_epoch": 0,
            "current_step": 0
        }
        
        # Login
        login_response = client.post("/api/auth/login", json={
            "username": "trainer",
            "password": "trainer123"
        })
        
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Create training session (this would trigger multi-GPU setup in real implementation)
        training_request = {
            "model_type": "dora",
            "model_name": "multi_gpu_test_model",
            "config": {
                "learning_rate": 0.001,
                "batch_size": 8,
                "epochs": 1,
                "use_multi_gpu": True
            },
            "data_source": "sample",
            "task_type": "text_classification"
        }
        
        create_response = client.post("/api/training/sessions", 
            json=training_request, 
            headers=headers
        )
        
        assert create_response.status_code == 200
        
        # Clean up
        session_id = create_response.json()["session_id"]
        client.delete(f"/api/training/sessions/{session_id}", headers=headers)

class TestPerformanceIntegration:
    """Test performance and load handling"""
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            try:
                response = client.get("/api/system/health")
                results.put(("health", response.status_code))
            except Exception as e:
                results.put(("health", str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        success_count = 0
        while not results.empty():
            endpoint, result = results.get()
            if result == 200:
                success_count += 1
        
        # At least 80% of requests should succeed
        assert success_count >= 8
    
    def test_large_payload_handling(self):
        """Test handling of large payloads"""
        # Login
        login_response = client.post("/api/auth/login", json={
            "username": "trainer",
            "password": "trainer123"
        })
        
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Create training session with large config
        large_config = {
            "learning_rate": 0.001,
            "batch_size": 8,
            "epochs": 100,
            "max_length": 2048,
            "large_data": "x" * 10000  # 10KB of data
        }
        
        training_request = {
            "model_type": "dora",
            "model_name": "large_payload_test",
            "config": large_config,
            "data_source": "sample",
            "task_type": "text_classification"
        }
        
        response = client.post("/api/training/sessions", 
            json=training_request, 
            headers=headers
        )
        
        # Should handle large payloads gracefully
        assert response.status_code in [200, 413]  # 413 if payload too large
        
        if response.status_code == 200:
            # Clean up
            session_id = response.json()["session_id"]
            client.delete(f"/api/training/sessions/{session_id}", headers=headers)

class TestErrorRecovery:
    """Test error recovery and resilience"""
    
    def test_service_recovery(self):
        """Test service recovery after errors"""
        # Make a request that might cause an error
        response = client.get("/api/training/sessions/nonexistent")
        assert response.status_code == 404
        
        # Service should still be responsive
        health_response = client.get("/api/system/health")
        assert health_response.status_code == 200
    
    def test_authentication_recovery(self):
        """Test authentication recovery"""
        # Try to access protected endpoint without auth
        response = client.get("/api/training/sessions")
        assert response.status_code == 401
        
        # Login and retry
        login_response = client.post("/api/auth/login", json={
            "username": "trainer",
            "password": "trainer123"
        })
        
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        
        # Should work now
        response = client.get("/api/training/sessions", 
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])