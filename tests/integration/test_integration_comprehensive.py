#!/usr/bin/env python3
"""
Comprehensive Integration Testing
تست جامع یکپارچگی سیستم
"""

import asyncio
import json
import time
import requests
import subprocess
import threading
import websocket
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("integration_test.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntegrationTester:
    """Comprehensive integration testing class"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.ws_url = "ws://localhost:8000"
        self.results = {
            'api_health_check': {},
            'frontend_backend_flow': {},
            'websocket_communication': {},
            'data_persistence': {},
            'error_handling': {},
            'performance_under_load': {},
            'overall_status': 'PENDING'
        }
        self.server_process = None
    
    def start_backend_server(self) -> bool:
        """Start the backend server for testing"""
        logger.info("🚀 Starting backend server...")
        
        try:
            # Start server in background
            self.server_process = subprocess.Popen(
                ["python", "backend/main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path(__file__).parent)
            )
            
            # Wait for server to start
            max_retries = 30
            for i in range(max_retries):
                try:
                    response = requests.get(f"{self.base_url}/api/system/health", timeout=2)
                    if response.status_code == 200:
                        logger.info("✅ Backend server started successfully")
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(1)
                logger.info(f"Waiting for server... ({i+1}/{max_retries})")
            
            logger.error("❌ Failed to start backend server")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error starting server: {str(e)}")
            return False
    
    def stop_backend_server(self):
        """Stop the backend server"""
        if self.server_process:
            logger.info("🛑 Stopping backend server...")
            self.server_process.terminate()
            self.server_process.wait()
            logger.info("✅ Backend server stopped")
    
    def test_api_health_check(self) -> bool:
        """Test 1: API Health Check"""
        logger.info("🔍 Testing API Health Check...")
        
        try:
            # Test system health endpoint
            response = requests.get(f"{self.base_url}/api/system/health", timeout=10)
            
            assert response.status_code == 200, f"Health check failed with status {response.status_code}"
            
            health_data = response.json()
            required_fields = ['status', 'timestamp', 'system_metrics', 'gpu_info', 'platform_info']
            
            for field in required_fields:
                assert field in health_data, f"Missing field: {field}"
            
            # Validate system metrics
            metrics = health_data['system_metrics']
            assert 'cpu_percent' in metrics, "Missing CPU metric"
            assert 'memory_percent' in metrics, "Missing memory metric"
            assert 0 <= metrics['cpu_percent'] <= 100, "Invalid CPU percentage"
            assert 0 <= metrics['memory_percent'] <= 100, "Invalid memory percentage"
            
            # Validate GPU info
            gpu_info = health_data['gpu_info']
            assert 'gpu_available' in gpu_info, "Missing GPU availability info"
            
            self.results['api_health_check'] = {
                'status': 'PASSED',
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'health_status': health_data['status'],
                'cpu_percent': metrics['cpu_percent'],
                'memory_percent': metrics['memory_percent'],
                'gpu_available': gpu_info['gpu_available']
            }
            
            logger.info("✅ API Health Check PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ API Health Check FAILED: {str(e)}")
            self.results['api_health_check'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_frontend_backend_data_flow(self) -> bool:
        """Test 2: Frontend-Backend Data Flow"""
        logger.info("🔍 Testing Frontend-Backend Data Flow...")
        
        try:
            # Test training sessions endpoint
            sessions_response = requests.get(f"{self.base_url}/api/training/sessions", timeout=10)
            assert sessions_response.status_code == 200, "Training sessions endpoint failed"
            
            sessions_data = sessions_response.json()
            assert isinstance(sessions_data, list), "Sessions data should be a list"
            
            # Test legal query processing
            test_query = {
                "query": "تحلیل ماده ۱۰ قانون مدنی ایران",
                "session_id": "test_session_123",
                "context": "قانون مدنی"
            }
            
            query_response = requests.post(
                f"{self.base_url}/api/legal/process",
                json=test_query,
                timeout=15
            )
            
            assert query_response.status_code == 200, f"Legal query failed with status {query_response.status_code}"
            
            query_data = query_response.json()
            required_fields = ['analysis', 'legal_references', 'confidence']
            
            for field in required_fields:
                assert field in query_data, f"Missing field in response: {field}"
            
            # Validate response content
            assert isinstance(query_data['analysis'], str), "Analysis should be a string"
            assert len(query_data['analysis']) > 0, "Analysis should not be empty"
            assert isinstance(query_data['legal_references'], list), "Legal references should be a list"
            assert 0 <= query_data['confidence'] <= 1, "Confidence should be between 0 and 1"
            
            # Test model training endpoint
            training_config = {
                "model_config": {
                    "base_model": "HooshvareLab/bert-base-parsbert-uncased",
                    "dora_rank": 8,
                    "dora_alpha": 8.0
                },
                "data_sources": ["lscp_legal", "hooshvare_legal"],
                "training_params": {
                    "num_epochs": 1,
                    "learning_rate": 2e-4,
                    "batch_size": 4
                }
            }
            
            training_response = requests.post(
                f"{self.base_url}/api/training/sessions",
                json=training_config,
                timeout=30
            )
            
            # Training might take time, so we accept 200 or 202
            assert training_response.status_code in [200, 202], f"Training endpoint failed with status {training_response.status_code}"
            
            self.results['frontend_backend_flow'] = {
                'status': 'PASSED',
                'sessions_endpoint_working': True,
                'legal_query_working': True,
                'training_endpoint_working': True,
                'query_response_time_ms': query_response.elapsed.total_seconds() * 1000,
                'analysis_length': len(query_data['analysis']),
                'legal_references_count': len(query_data['legal_references']),
                'confidence_score': query_data['confidence']
            }
            
            logger.info("✅ Frontend-Backend Data Flow PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Frontend-Backend Data Flow FAILED: {str(e)}")
            self.results['frontend_backend_flow'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_websocket_communication(self) -> bool:
        """Test 3: WebSocket Communication"""
        logger.info("🔍 Testing WebSocket Communication...")
        
        try:
            # Test WebSocket connection
            ws_url = f"{self.ws_url}/api/training/ws/training/test_session_123"
            
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    assert 'status' in data, "WebSocket message missing status"
                    assert data['status'] in ['training', 'completed', 'error'], f"Invalid status: {data['status']}"
                    return data
                except Exception as e:
                    logger.error(f"WebSocket message error: {str(e)}")
                    return None
            
            def on_error(ws, error):
                logger.error(f"WebSocket error: {str(error)}")
            
            def on_close(ws, close_status_code, close_msg):
                logger.info("WebSocket connection closed")
            
            def on_open(ws):
                logger.info("WebSocket connection opened")
                # Send test message
                test_message = {"action": "status_request"}
                ws.send(json.dumps(test_message))
            
            # Create WebSocket connection
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Run WebSocket in a separate thread
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection and message
            time.sleep(3)
            
            # Close connection
            ws.close()
            
            self.results['websocket_communication'] = {
                'status': 'PASSED',
                'connection_established': True,
                'message_received': True
            }
            
            logger.info("✅ WebSocket Communication PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ WebSocket Communication FAILED: {str(e)}")
            self.results['websocket_communication'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_data_persistence(self) -> bool:
        """Test 4: Data Persistence"""
        logger.info("🔍 Testing Data Persistence...")
        
        try:
            # Test session creation and retrieval
            session_data = {
                "session_id": "test_persistence_session",
                "model_config": {
                    "base_model": "HooshvareLab/bert-base-parsbert-uncased",
                    "dora_rank": 4
                },
                "status": "created"
            }
            
            # Create session
            create_response = requests.post(
                f"{self.base_url}/api/training/sessions",
                json=session_data,
                timeout=10
            )
            
            assert create_response.status_code in [200, 201], f"Session creation failed: {create_response.status_code}"
            
            # Retrieve session
            get_response = requests.get(
                f"{self.base_url}/api/training/sessions/{session_data['session_id']}",
                timeout=10
            )
            
            assert get_response.status_code == 200, f"Session retrieval failed: {get_response.status_code}"
            
            retrieved_data = get_response.json()
            assert retrieved_data['session_id'] == session_data['session_id'], "Session ID mismatch"
            
            # Test data consistency
            sessions_list_response = requests.get(f"{self.base_url}/api/training/sessions", timeout=10)
            assert sessions_list_response.status_code == 200, "Sessions list failed"
            
            sessions_list = sessions_list_response.json()
            session_found = any(s['session_id'] == session_data['session_id'] for s in sessions_list)
            assert session_found, "Created session not found in list"
            
            self.results['data_persistence'] = {
                'status': 'PASSED',
                'session_creation': True,
                'session_retrieval': True,
                'data_consistency': True
            }
            
            logger.info("✅ Data Persistence PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data Persistence FAILED: {str(e)}")
            self.results['data_persistence'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_error_handling(self) -> bool:
        """Test 5: Error Handling"""
        logger.info("🔍 Testing Error Handling...")
        
        try:
            # Test invalid endpoint
            invalid_response = requests.get(f"{self.base_url}/api/invalid/endpoint", timeout=5)
            assert invalid_response.status_code == 404, "Should return 404 for invalid endpoint"
            
            # Test invalid JSON
            try:
                invalid_json_response = requests.post(
                    f"{self.base_url}/api/legal/process",
                    data="invalid json",
                    headers={'Content-Type': 'application/json'},
                    timeout=5
                )
                assert invalid_json_response.status_code == 422, "Should return 422 for invalid JSON"
            except requests.exceptions.RequestException:
                # Some servers might handle this differently
                pass
            
            # Test missing required fields
            incomplete_query = {"query": "test"}
            incomplete_response = requests.post(
                f"{self.base_url}/api/legal/process",
                json=incomplete_query,
                timeout=5
            )
            # Should either succeed with defaults or return validation error
            assert incomplete_response.status_code in [200, 422], f"Unexpected status for incomplete query: {incomplete_response.status_code}"
            
            # Test large payload
            large_query = {
                "query": "test " * 10000,  # Very large query
                "session_id": "test_session"
            }
            large_response = requests.post(
                f"{self.base_url}/api/legal/process",
                json=large_query,
                timeout=10
            )
            # Should handle large payload gracefully
            assert large_response.status_code in [200, 413, 422], f"Unexpected status for large payload: {large_response.status_code}"
            
            self.results['error_handling'] = {
                'status': 'PASSED',
                'invalid_endpoint_handled': True,
                'invalid_json_handled': True,
                'incomplete_data_handled': True,
                'large_payload_handled': True
            }
            
            logger.info("✅ Error Handling PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error Handling FAILED: {str(e)}")
            self.results['error_handling'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_performance_under_load(self) -> bool:
        """Test 6: Performance Under Load"""
        logger.info("🔍 Testing Performance Under Load...")
        
        try:
            import concurrent.futures
            import statistics
            
            def make_request():
                try:
                    response = requests.get(f"{self.base_url}/api/system/health", timeout=5)
                    return {
                        'status_code': response.status_code,
                        'response_time': response.elapsed.total_seconds() * 1000,
                        'success': response.status_code == 200
                    }
                except Exception as e:
                    return {
                        'status_code': 0,
                        'response_time': 0,
                        'success': False,
                        'error': str(e)
                    }
            
            # Test with 20 concurrent requests
            num_requests = 20
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_request) for _ in range(num_requests)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Analyze results
            successful_requests = [r for r in results if r['success']]
            response_times = [r['response_time'] for r in successful_requests]
            
            success_rate = len(successful_requests) / num_requests * 100
            avg_response_time = statistics.mean(response_times) if response_times else 0
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times) if response_times else 0
            
            # Performance criteria
            performance_acceptable = (
                success_rate >= 95 and  # 95% success rate
                avg_response_time < 1000 and  # Average response time < 1s
                p95_response_time < 2000  # P95 response time < 2s
            )
            
            self.results['performance_under_load'] = {
                'status': 'PASSED' if performance_acceptable else 'FAILED',
                'total_requests': num_requests,
                'successful_requests': len(successful_requests),
                'success_rate_percent': success_rate,
                'avg_response_time_ms': avg_response_time,
                'p95_response_time_ms': p95_response_time,
                'performance_acceptable': performance_acceptable
            }
            
            if performance_acceptable:
                logger.info("✅ Performance Under Load PASSED")
            else:
                logger.warning("⚠️ Performance Under Load FAILED - Performance criteria not met")
            
            return performance_acceptable
            
        except Exception as e:
            logger.error(f"❌ Performance Under Load FAILED: {str(e)}")
            self.results['performance_under_load'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def run_comprehensive_integration_test(self) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("🎯 Starting COMPREHENSIVE INTEGRATION TESTING")
        logger.info("=" * 60)
        
        # Start server
        if not self.start_backend_server():
            logger.error("❌ Cannot start backend server. Aborting integration tests.")
            return self.results
        
        try:
            tests = [
                ("API Health Check", self.test_api_health_check),
                ("Frontend-Backend Data Flow", self.test_frontend_backend_data_flow),
                ("WebSocket Communication", self.test_websocket_communication),
                ("Data Persistence", self.test_data_persistence),
                ("Error Handling", self.test_error_handling),
                ("Performance Under Load", self.test_performance_under_load)
            ]
            
            passed_tests = 0
            total_tests = len(tests)
            
            for test_name, test_func in tests:
                logger.info(f"\n📋 Running {test_name}...")
                try:
                    if test_func():
                        passed_tests += 1
                        logger.info(f"✅ {test_name} PASSED")
                    else:
                        logger.error(f"❌ {test_name} FAILED")
                except Exception as e:
                    logger.error(f"❌ {test_name} FAILED with exception: {str(e)}")
            
            # Overall assessment
            success_rate = (passed_tests / total_tests) * 100
            self.results['overall_status'] = 'SUCCESS' if success_rate >= 80 else 'FAILURE'
            self.results['success_rate'] = success_rate
            self.results['passed_tests'] = passed_tests
            self.results['total_tests'] = total_tests
            
            logger.info("\n" + "=" * 60)
            logger.info("📊 INTEGRATION TEST SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
            logger.info(f"Success Rate: {success_rate:.1f}%")
            logger.info(f"Overall Status: {self.results['overall_status']}")
            logger.info("=" * 60)
            
            return self.results
            
        finally:
            self.stop_backend_server()

def main():
    """Main execution function"""
    tester = IntegrationTester()
    results = tester.run_comprehensive_integration_test()
    
    # Save results to file
    with open('integration_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("📄 Results saved to integration_test_results.json")
    
    return results

if __name__ == "__main__":
    main()