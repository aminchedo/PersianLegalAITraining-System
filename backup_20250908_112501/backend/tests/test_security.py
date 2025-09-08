"""
Security Tests for Persian Legal AI Backend
تست‌های امنیتی برای Backend هوش مصنوعی حقوقی فارسی
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

# Import the main app
from main import app

client = TestClient(app)

class TestAuthentication:
    """Test authentication endpoints"""
    
    def test_login_success(self):
        """Test successful login"""
        response = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "user" in data
        assert data["user"]["username"] == "admin"
        assert "admin" in data["user"]["permissions"]
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        response = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "wrongpassword"
        })
        
        assert response.status_code == 401
        assert "detail" in response.json()
    
    def test_login_missing_fields(self):
        """Test login with missing fields"""
        response = client.post("/api/auth/login", json={
            "username": "admin"
            # Missing password
        })
        
        assert response.status_code == 422
    
    def test_get_current_user_without_token(self):
        """Test getting current user without token"""
        response = client.get("/api/auth/me")
        
        assert response.status_code == 401
    
    def test_get_current_user_with_valid_token(self):
        """Test getting current user with valid token"""
        # First login to get token
        login_response = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        token = login_response.json()["access_token"]
        
        # Use token to get user info
        response = client.get("/api/auth/me", headers={
            "Authorization": f"Bearer {token}"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "admin"
    
    def test_refresh_token(self):
        """Test token refresh"""
        # First login to get token
        login_response = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        token = login_response.json()["access_token"]
        
        # Refresh token
        response = client.post("/api/auth/refresh", headers={
            "Authorization": f"Bearer {token}"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["access_token"] != token  # Should be different token
    
    def test_logout(self):
        """Test logout"""
        # First login to get token
        login_response = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        token = login_response.json()["access_token"]
        
        # Logout
        response = client.post("/api/auth/logout", headers={
            "Authorization": f"Bearer {token}"
        })
        
        assert response.status_code == 200
        assert "message" in response.json()

class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limiting_login(self):
        """Test rate limiting on login endpoint"""
        # Make multiple requests to trigger rate limiting
        for i in range(6):  # Should trigger rate limit (5 requests per 5 minutes)
            response = client.post("/api/auth/login", json={
                "username": "admin",
                "password": "wrongpassword"  # Wrong password to avoid success
            })
            
            if i < 5:
                assert response.status_code == 401  # Wrong credentials
            else:
                assert response.status_code == 429  # Rate limited
                assert "X-RateLimit-Limit" in response.headers
                assert "X-RateLimit-Remaining" in response.headers
    
    def test_rate_limiting_headers(self):
        """Test rate limiting headers are present"""
        response = client.get("/api/system/health")
        
        # Should have rate limiting headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

class TestTrainingEndpointsSecurity:
    """Test training endpoints security"""
    
    def test_create_training_session_without_auth(self):
        """Test creating training session without authentication"""
        response = client.post("/api/training/sessions", json={
            "model_type": "dora",
            "model_name": "test_model",
            "config": {"learning_rate": 0.001},
            "data_source": "sample",
            "task_type": "text_classification"
        })
        
        assert response.status_code == 401
    
    def test_create_training_session_with_auth(self):
        """Test creating training session with authentication"""
        # Login to get token
        login_response = client.post("/api/auth/login", json={
            "username": "trainer",
            "password": "trainer123"
        })
        
        token = login_response.json()["access_token"]
        
        # Create training session
        response = client.post("/api/training/sessions", 
            json={
                "model_type": "dora",
                "model_name": "test_model",
                "config": {"learning_rate": 0.001},
                "data_source": "sample",
                "task_type": "text_classification"
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
    
    def test_create_training_session_without_permission(self):
        """Test creating training session without training permission"""
        # Login with viewer account (no training permission)
        login_response = client.post("/api/auth/login", json={
            "username": "viewer",
            "password": "viewer123"
        })
        
        token = login_response.json()["access_token"]
        
        # Try to create training session
        response = client.post("/api/training/sessions", 
            json={
                "model_type": "dora",
                "model_name": "test_model",
                "config": {"learning_rate": 0.001},
                "data_source": "sample",
                "task_type": "text_classification"
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 403
        assert "Permission 'training' required" in response.json()["detail"]
    
    def test_get_training_sessions_without_auth(self):
        """Test getting training sessions without authentication"""
        response = client.get("/api/training/sessions")
        
        assert response.status_code == 401
    
    def test_get_training_sessions_with_auth(self):
        """Test getting training sessions with authentication"""
        # Login to get token
        login_response = client.post("/api/auth/login", json={
            "username": "trainer",
            "password": "trainer123"
        })
        
        token = login_response.json()["access_token"]
        
        # Get training sessions
        response = client.get("/api/training/sessions", 
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        assert isinstance(response.json(), list)

class TestSystemEndpointsSecurity:
    """Test system endpoints security"""
    
    def test_health_endpoint_public(self):
        """Test health endpoint is publicly accessible"""
        response = client.get("/api/system/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_health_endpoint_with_auth(self):
        """Test health endpoint with authentication provides more info"""
        # Login to get token
        login_response = client.post("/api/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        token = login_response.json()["access_token"]
        
        # Get health with auth
        response = client.get("/api/system/health", 
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "checks" in data
        assert "security" in data["checks"]

class TestCORS:
    """Test CORS configuration"""
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.options("/api/auth/login")
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers

class TestInputValidation:
    """Test input validation and sanitization"""
    
    def test_malicious_input_login(self):
        """Test login with malicious input"""
        malicious_inputs = [
            {"username": "<script>alert('xss')</script>", "password": "test"},
            {"username": "admin'; DROP TABLE users; --", "password": "test"},
            {"username": "admin", "password": "../../etc/passwd"},
        ]
        
        for malicious_input in malicious_inputs:
            response = client.post("/api/auth/login", json=malicious_input)
            # Should not crash and should return 401 (invalid credentials)
            assert response.status_code == 401
    
    def test_training_session_validation(self):
        """Test training session input validation"""
        # Login to get token
        login_response = client.post("/api/auth/login", json={
            "username": "trainer",
            "password": "trainer123"
        })
        
        token = login_response.json()["access_token"]
        
        # Test invalid model type
        response = client.post("/api/training/sessions", 
            json={
                "model_type": "invalid_model",
                "model_name": "test_model",
                "config": {"learning_rate": 0.001},
                "data_source": "sample",
                "task_type": "text_classification"
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 400
        assert "Invalid model type" in response.json()["detail"]
        
        # Test invalid task type
        response = client.post("/api/training/sessions", 
            json={
                "model_type": "dora",
                "model_name": "test_model",
                "config": {"learning_rate": 0.001},
                "data_source": "sample",
                "task_type": "invalid_task"
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 400
        assert "Invalid task type" in response.json()["detail"]

class TestErrorHandling:
    """Test error handling and information disclosure"""
    
    def test_error_response_format(self):
        """Test error responses don't leak sensitive information"""
        # Test with invalid endpoint
        response = client.get("/api/nonexistent")
        
        assert response.status_code == 404
        # Should not contain internal paths or sensitive info
        error_detail = response.json()["detail"]
        assert "traceback" not in error_detail.lower()
        assert "/app/" not in error_detail
    
    def test_database_error_handling(self):
        """Test database error handling"""
        with patch('database.connection.db_manager.test_connection') as mock_db:
            mock_db.side_effect = Exception("Database connection failed")
            
            response = client.get("/api/system/health")
            
            # Should handle database errors gracefully
            assert response.status_code == 200  # Health endpoint should still work
            # But database status should reflect the error
            data = response.json()
            assert data["checks"]["database"]["status"] == "unhealthy"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])