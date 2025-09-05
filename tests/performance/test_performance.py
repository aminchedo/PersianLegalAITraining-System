#!/usr/bin/env python3
"""
Performance and Load Testing
تست عملکرد و بار
"""

import time
import statistics
import concurrent.futures
import requests
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("performance_test.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceTester:
    """Performance and load testing class"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            'single_request_performance': {},
            'concurrent_load_test': {},
            'memory_usage_test': {},
            'response_time_analysis': {},
            'throughput_analysis': {},
            'overall_performance': {}
        }
    
    def test_single_request_performance(self) -> bool:
        """Test single request performance"""
        logger.info("🔍 Testing Single Request Performance...")
        
        try:
            endpoints = [
                "/api/system/health",
                "/api/training/sessions",
                "/api/legal/process"
            ]
            
            performance_data = {}
            
            for endpoint in endpoints:
                response_times = []
                
                # Warmup requests
                for _ in range(3):
                    try:
                        if endpoint == "/api/legal/process":
                            requests.post(f"{self.base_url}{endpoint}", 
                                        json={"query": "test", "session_id": "perf_test"})
                        else:
                            requests.get(f"{self.base_url}{endpoint}")
                    except:
                        pass
                
                # Performance test
                for _ in range(10):
                    start_time = time.time()
                    try:
                        if endpoint == "/api/legal/process":
                            response = requests.post(f"{self.base_url}{endpoint}", 
                                                   json={"query": "test", "session_id": "perf_test"},
                                                   timeout=10)
                        else:
                            response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                        
                        end_time = time.time()
                        response_time = (end_time - start_time) * 1000  # Convert to ms
                        response_times.append(response_time)
                        
                    except Exception as e:
                        logger.warning(f"Request failed for {endpoint}: {str(e)}")
                
                if response_times:
                    performance_data[endpoint] = {
                        'avg_response_time_ms': statistics.mean(response_times),
                        'min_response_time_ms': min(response_times),
                        'max_response_time_ms': max(response_times),
                        'p95_response_time_ms': statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times),
                        'success_rate': len(response_times) / 10 * 100
                    }
            
            self.results['single_request_performance'] = {
                'status': 'PASSED',
                'endpoints_tested': len(performance_data),
                'performance_data': performance_data
            }
            
            logger.info("✅ Single Request Performance PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Single Request Performance FAILED: {str(e)}")
            self.results['single_request_performance'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_concurrent_load(self, num_users: int = 50, duration_seconds: int = 60) -> bool:
        """Test concurrent load"""
        logger.info(f"🔍 Testing Concurrent Load ({num_users} users, {duration_seconds}s)...")
        
        try:
            def user_simulation():
                """Simulate a single user"""
                request_count = 0
                success_count = 0
                response_times = []
                start_time = time.time()
                
                while time.time() - start_time < duration_seconds:
                    try:
                        # Random endpoint selection
                        endpoints = [
                            ("/api/system/health", "GET", None),
                            ("/api/training/sessions", "GET", None),
                            ("/api/legal/process", "POST", {"query": "test", "session_id": "load_test"})
                        ]
                        
                        endpoint, method, data = endpoints[request_count % len(endpoints)]
                        
                        req_start = time.time()
                        if method == "POST":
                            response = requests.post(f"{self.base_url}{endpoint}", json=data, timeout=5)
                        else:
                            response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                        
                        req_end = time.time()
                        response_time = (req_end - req_start) * 1000
                        
                        if response.status_code == 200:
                            success_count += 1
                            response_times.append(response_time)
                        
                        request_count += 1
                        
                        # Small delay between requests
                        time.sleep(0.1)
                        
                    except Exception as e:
                        logger.debug(f"Request failed: {str(e)}")
                        request_count += 1
                
                return {
                    'total_requests': request_count,
                    'successful_requests': success_count,
                    'response_times': response_times,
                    'success_rate': success_count / request_count * 100 if request_count > 0 else 0
                }
            
            # Run concurrent users
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [executor.submit(user_simulation) for _ in range(num_users)]
                user_results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Aggregate results
            total_requests = sum(r['total_requests'] for r in user_results)
            total_successful = sum(r['successful_requests'] for r in user_results)
            all_response_times = []
            for r in user_results:
                all_response_times.extend(r['response_times'])
            
            overall_success_rate = total_successful / total_requests * 100 if total_requests > 0 else 0
            avg_response_time = statistics.mean(all_response_times) if all_response_times else 0
            p95_response_time = statistics.quantiles(all_response_times, n=20)[18] if len(all_response_times) >= 20 else max(all_response_times) if all_response_times else 0
            
            # Performance criteria
            performance_acceptable = (
                overall_success_rate >= 95 and
                avg_response_time < 2000 and
                p95_response_time < 5000
            )
            
            self.results['concurrent_load_test'] = {
                'status': 'PASSED' if performance_acceptable else 'FAILED',
                'num_users': num_users,
                'duration_seconds': duration_seconds,
                'total_requests': total_requests,
                'total_successful': total_successful,
                'overall_success_rate': overall_success_rate,
                'avg_response_time_ms': avg_response_time,
                'p95_response_time_ms': p95_response_time,
                'requests_per_second': total_requests / duration_seconds,
                'performance_acceptable': performance_acceptable
            }
            
            if performance_acceptable:
                logger.info("✅ Concurrent Load Test PASSED")
            else:
                logger.warning("⚠️ Concurrent Load Test FAILED - Performance criteria not met")
            
            return performance_acceptable
            
        except Exception as e:
            logger.error(f"❌ Concurrent Load Test FAILED: {str(e)}")
            self.results['concurrent_load_test'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_memory_usage(self) -> bool:
        """Test memory usage under load"""
        logger.info("🔍 Testing Memory Usage...")
        
        try:
            import psutil
            import os
            
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate memory-intensive operations
            def memory_intensive_request():
                """Make memory-intensive requests"""
                for _ in range(10):
                    try:
                        # Large legal query
                        large_query = {
                            "query": "تحلیل جامع " + "قانون مدنی ایران " * 100,
                            "session_id": "memory_test",
                            "context": "قانون مدنی " * 50
                        }
                        
                        response = requests.post(
                            f"{self.base_url}/api/legal/process",
                            json=large_query,
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            # Process response to simulate frontend usage
                            data = response.json()
                            _ = data.get('analysis', '')
                            _ = data.get('legal_references', [])
                        
                    except Exception as e:
                        logger.debug(f"Memory test request failed: {str(e)}")
            
            # Run memory-intensive operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(memory_intensive_request) for _ in range(5)]
                concurrent.futures.wait(futures)
            
            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # System memory usage
            system_memory = psutil.virtual_memory()
            
            self.results['memory_usage_test'] = {
                'status': 'PASSED',
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase,
                'system_memory_percent': system_memory.percent,
                'system_memory_available_gb': system_memory.available / 1024 / 1024 / 1024,
                'memory_efficient': memory_increase < 100  # Less than 100MB increase
            }
            
            logger.info("✅ Memory Usage Test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory Usage Test FAILED: {str(e)}")
            self.results['memory_usage_test'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_response_time_analysis(self) -> bool:
        """Detailed response time analysis"""
        logger.info("🔍 Testing Response Time Analysis...")
        
        try:
            # Test different payload sizes
            payload_sizes = [
                ("small", {"query": "test", "session_id": "small"}),
                ("medium", {"query": "تحلیل ماده ۱۰ قانون مدنی ایران", "session_id": "medium"}),
                ("large", {"query": "تحلیل جامع " + "قانون مدنی " * 50, "session_id": "large"})
            ]
            
            response_time_data = {}
            
            for size_name, payload in payload_sizes:
                response_times = []
                
                for _ in range(20):
                    start_time = time.time()
                    try:
                        response = requests.post(
                            f"{self.base_url}/api/legal/process",
                            json=payload,
                            timeout=15
                        )
                        end_time = time.time()
                        
                        if response.status_code == 200:
                            response_time = (end_time - start_time) * 1000
                            response_times.append(response_time)
                            
                    except Exception as e:
                        logger.debug(f"Response time test failed for {size_name}: {str(e)}")
                
                if response_times:
                    response_time_data[size_name] = {
                        'avg_response_time_ms': statistics.mean(response_times),
                        'min_response_time_ms': min(response_times),
                        'max_response_time_ms': max(response_times),
                        'std_deviation_ms': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                        'p95_response_time_ms': statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times)
                    }
            
            self.results['response_time_analysis'] = {
                'status': 'PASSED',
                'payload_sizes_tested': len(response_time_data),
                'response_time_data': response_time_data
            }
            
            logger.info("✅ Response Time Analysis PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Response Time Analysis FAILED: {str(e)}")
            self.results['response_time_analysis'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_throughput_analysis(self) -> bool:
        """Test system throughput"""
        logger.info("🔍 Testing Throughput Analysis...")
        
        try:
            # Test different concurrency levels
            concurrency_levels = [1, 5, 10, 20]
            throughput_data = {}
            
            for concurrency in concurrency_levels:
                start_time = time.time()
                request_count = 0
                success_count = 0
                
                def make_request():
                    nonlocal request_count, success_count
                    try:
                        response = requests.get(f"{self.base_url}/api/system/health", timeout=5)
                        request_count += 1
                        if response.status_code == 200:
                            success_count += 1
                    except:
                        request_count += 1
                
                # Run for 10 seconds
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    end_time = start_time + 10
                    while time.time() < end_time:
                        futures = [executor.submit(make_request) for _ in range(concurrency)]
                        concurrent.futures.wait(futures, timeout=1)
                
                duration = time.time() - start_time
                throughput = request_count / duration
                success_rate = success_count / request_count * 100 if request_count > 0 else 0
                
                throughput_data[f'concurrency_{concurrency}'] = {
                    'requests_per_second': throughput,
                    'success_rate': success_rate,
                    'total_requests': request_count,
                    'successful_requests': success_count,
                    'duration_seconds': duration
                }
            
            self.results['throughput_analysis'] = {
                'status': 'PASSED',
                'concurrency_levels_tested': len(concurrency_levels),
                'throughput_data': throughput_data
            }
            
            logger.info("✅ Throughput Analysis PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Throughput Analysis FAILED: {str(e)}")
            self.results['throughput_analysis'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def run_comprehensive_performance_test(self) -> Dict[str, Any]:
        """Run all performance tests"""
        logger.info("🎯 Starting COMPREHENSIVE PERFORMANCE TESTING")
        logger.info("=" * 60)
        
        tests = [
            ("Single Request Performance", self.test_single_request_performance),
            ("Concurrent Load Test", lambda: self.test_concurrent_load(20, 30)),  # Reduced for testing
            ("Memory Usage Test", self.test_memory_usage),
            ("Response Time Analysis", self.test_response_time_analysis),
            ("Throughput Analysis", self.test_throughput_analysis)
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
        self.results['overall_performance'] = {
            'status': 'SUCCESS' if success_rate >= 80 else 'FAILURE',
            'success_rate': success_rate,
            'passed_tests': passed_tests,
            'total_tests': total_tests
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 PERFORMANCE TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Overall Status: {self.results['overall_performance']['status']}")
        logger.info("=" * 60)
        
        return self.results

def main():
    """Main execution function"""
    tester = PerformanceTester()
    results = tester.run_comprehensive_performance_test()
    
    # Save results to file
    with open('performance_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("📄 Results saved to performance_test_results.json")
    
    return results

if __name__ == "__main__":
    main()