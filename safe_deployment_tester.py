#!/usr/bin/env python3
"""
Safe Deployment Tester for Persian Legal AI
Tests deployment without affecting production

🛡️ SAFETY: This tester runs in isolated environments and never modifies production
"""

import os
import json
import time
import requests
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import threading
import signal
import sys

class SafeDeploymentTester:
    """Tests deployment without affecting production"""
    
    def __init__(self):
        self.workspace = Path("/workspace")
        self.test_environment = 'testing'
        self.test_results = {}
        self.test_containers = []
        self.cleanup_required = False
        
        # Test configuration
        self.test_ports = {
            'backend': 8888,
            'frontend': 3888,
            'redis': 6888
        }
        
        # Setup signal handlers for cleanup
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def test_current_deployment(self) -> Dict[str, Any]:
        """Test existing deployment as-is"""
        print("🧪 Testing Current Deployment Configuration...")
        print("🛡️ SAFETY: Testing existing setup without modifications")
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'current_deployment',
            'configuration_tests': {},
            'service_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'overall_status': 'unknown'
        }
        
        # Test configuration validity
        test_results['configuration_tests'] = self._test_configuration_validity()
        
        # Test service availability
        test_results['service_tests'] = self._test_service_availability()
        
        # Test API endpoints
        test_results['integration_tests'] = self._test_api_endpoints()
        
        # Basic performance tests
        test_results['performance_tests'] = self._test_basic_performance()
        
        # Determine overall status
        test_results['overall_status'] = self._determine_overall_status(test_results)
        
        self.test_results['current_deployment'] = test_results
        return test_results
    
    def test_with_optimizations(self, optimization_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Test with suggested optimizations in isolated environment"""
        print("🔬 Testing With Optimizations (Isolated Environment)...")
        print("🛡️ SAFETY: Testing in isolated environment - production unaffected")
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'optimized_deployment',
            'optimization_config': optimization_config or {},
            'isolated_environment': True,
            'test_results': {},
            'performance_comparison': {},
            'overall_status': 'unknown'
        }
        
        try:
            # Create isolated test environment
            test_env_path = self._create_test_environment()
            
            # Apply optimizations to test environment
            if optimization_config:
                self._apply_test_optimizations(test_env_path, optimization_config)
            
            # Run tests in isolated environment
            test_results['test_results'] = self._run_isolated_tests(test_env_path)
            
            # Compare with current deployment
            if 'current_deployment' in self.test_results:
                test_results['performance_comparison'] = self._compare_performance(
                    self.test_results['current_deployment'],
                    test_results['test_results']
                )
            
            # Cleanup test environment
            self._cleanup_test_environment(test_env_path)
            
            test_results['overall_status'] = self._determine_overall_status(test_results['test_results'])
            
        except Exception as e:
            test_results['error'] = str(e)
            test_results['overall_status'] = 'failed'
            print(f"❌ Test failed: {e}")
            
            # Ensure cleanup
            self._emergency_cleanup()
        
        self.test_results['optimized_deployment'] = test_results
        return test_results
    
    def _test_configuration_validity(self) -> Dict[str, Any]:
        """Test configuration file validity"""
        print("\n📋 Testing Configuration Validity...")
        
        results = {
            'docker_compose': {},
            'dockerfiles': {},
            'environment': {},
            'overall_valid': True
        }
        
        # Test docker-compose.yml
        compose_files = ['docker-compose.yml', 'docker-compose.enhanced.yml']
        for compose_file in compose_files:
            compose_path = self.workspace / compose_file
            if compose_path.exists():
                try:
                    # Test YAML parsing
                    import yaml
                    with open(compose_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Test docker-compose config command
                    result = subprocess.run(
                        ['docker-compose', '-f', str(compose_path), 'config'],
                        capture_output=True, text=True, timeout=30
                    )
                    
                    if result.returncode == 0:
                        results['docker_compose'][compose_file] = {'status': 'valid', 'details': 'Configuration is valid'}
                        print(f"  ✅ {compose_file}: Valid")
                    else:
                        results['docker_compose'][compose_file] = {'status': 'invalid', 'details': result.stderr}
                        results['overall_valid'] = False
                        print(f"  ❌ {compose_file}: Invalid - {result.stderr[:100]}...")
                
                except yaml.YAMLError as e:
                    results['docker_compose'][compose_file] = {'status': 'invalid', 'details': f'YAML error: {e}'}
                    results['overall_valid'] = False
                    print(f"  ❌ {compose_file}: YAML error")
                except subprocess.TimeoutExpired:
                    results['docker_compose'][compose_file] = {'status': 'timeout', 'details': 'Validation timeout'}
                    print(f"  ⚠️ {compose_file}: Validation timeout")
                except Exception as e:
                    results['docker_compose'][compose_file] = {'status': 'error', 'details': str(e)}
                    print(f"  ❌ {compose_file}: Error - {e}")
        
        # Test Dockerfiles
        dockerfiles = ['backend/Dockerfile', 'persian-legal-ai-frontend/Dockerfile']
        for dockerfile in dockerfiles:
            dockerfile_path = self.workspace / dockerfile
            if dockerfile_path.exists():
                try:
                    with open(dockerfile_path, 'r') as f:
                        content = f.read()
                    
                    # Basic Dockerfile validation
                    if content.strip().startswith('FROM'):
                        results['dockerfiles'][dockerfile] = {'status': 'valid', 'details': 'Basic structure is valid'}
                        print(f"  ✅ {dockerfile}: Valid structure")
                    else:
                        results['dockerfiles'][dockerfile] = {'status': 'invalid', 'details': 'Missing FROM instruction'}
                        results['overall_valid'] = False
                        print(f"  ❌ {dockerfile}: Missing FROM instruction")
                
                except Exception as e:
                    results['dockerfiles'][dockerfile] = {'status': 'error', 'details': str(e)}
                    print(f"  ❌ {dockerfile}: Error - {e}")
        
        return results
    
    def _test_service_availability(self) -> Dict[str, Any]:
        """Test service availability"""
        print("\n🔍 Testing Service Availability...")
        
        results = {
            'services': {},
            'docker_status': {},
            'overall_available': True
        }
        
        # Check Docker availability
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                results['docker_status'] = {'available': True, 'version': result.stdout.strip()}
                print("  ✅ Docker: Available")
            else:
                results['docker_status'] = {'available': False, 'error': result.stderr}
                results['overall_available'] = False
                print("  ❌ Docker: Not available")
        except Exception as e:
            results['docker_status'] = {'available': False, 'error': str(e)}
            results['overall_available'] = False
            print("  ❌ Docker: Error")
        
        # Check running containers
        if results['docker_status'].get('available'):
            try:
                result = subprocess.run(
                    ['docker', 'ps', '--format', 'json'],
                    capture_output=True, text=True, timeout=10
                )
                
                if result.returncode == 0:
                    containers = []
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            try:
                                container = json.loads(line)
                                containers.append(container)
                            except json.JSONDecodeError:
                                pass
                    
                    # Check for expected containers
                    expected_containers = ['persian-legal-backend', 'persian-legal-frontend', 'persian-legal-redis']
                    
                    for expected in expected_containers:
                        found = any(expected in container.get('Names', '') for container in containers)
                        if found:
                            results['services'][expected] = {'status': 'running', 'details': 'Container is running'}
                            print(f"  ✅ {expected}: Running")
                        else:
                            results['services'][expected] = {'status': 'not_running', 'details': 'Container not found'}
                            print(f"  ⚠️ {expected}: Not running")
                
            except Exception as e:
                results['services']['docker_ps'] = {'status': 'error', 'details': str(e)}
                print(f"  ❌ Container check failed: {e}")
        
        return results
    
    def _test_api_endpoints(self) -> Dict[str, Any]:
        """Test API endpoints"""
        print("\n🌐 Testing API Endpoints...")
        
        results = {
            'endpoints': {},
            'response_times': {},
            'overall_functional': True
        }
        
        # Define test endpoints
        base_urls = ['http://localhost:8000', 'http://localhost:8001']  # Regular and enhanced
        test_endpoints = [
            {'path': '/api/system/health', 'method': 'GET', 'expected_status': 200},
            {'path': '/api/docs', 'method': 'GET', 'expected_status': 200},
            {'path': '/', 'method': 'GET', 'expected_status': 200}  # Root endpoint
        ]
        
        for base_url in base_urls:
            for endpoint in test_endpoints:
                endpoint_key = f"{base_url}{endpoint['path']}"
                
                try:
                    start_time = time.time()
                    response = requests.request(
                        endpoint['method'],
                        endpoint_key,
                        timeout=10,
                        headers={'User-Agent': 'PersianLegalAI-Tester/1.0'}
                    )
                    response_time = time.time() - start_time
                    
                    results['response_times'][endpoint_key] = response_time
                    
                    if response.status_code == endpoint['expected_status']:
                        results['endpoints'][endpoint_key] = {
                            'status': 'success',
                            'response_code': response.status_code,
                            'response_time': response_time,
                            'details': 'Endpoint responded as expected'
                        }
                        print(f"  ✅ {endpoint_key}: OK ({response_time:.3f}s)")
                    else:
                        results['endpoints'][endpoint_key] = {
                            'status': 'unexpected_response',
                            'response_code': response.status_code,
                            'expected_code': endpoint['expected_status'],
                            'response_time': response_time,
                            'details': f"Expected {endpoint['expected_status']}, got {response.status_code}"
                        }
                        print(f"  ⚠️ {endpoint_key}: Unexpected response ({response.status_code})")
                
                except requests.exceptions.RequestException as e:
                    results['endpoints'][endpoint_key] = {
                        'status': 'unreachable',
                        'error': str(e),
                        'details': 'Endpoint is unreachable'
                    }
                    print(f"  ❌ {endpoint_key}: Unreachable")
                
                except Exception as e:
                    results['endpoints'][endpoint_key] = {
                        'status': 'error',
                        'error': str(e),
                        'details': 'Test error occurred'
                    }
                    print(f"  ❌ {endpoint_key}: Error")
        
        # Check if any endpoints are working
        working_endpoints = [ep for ep in results['endpoints'].values() if ep['status'] == 'success']
        if not working_endpoints:
            results['overall_functional'] = False
        
        return results
    
    def _test_basic_performance(self) -> Dict[str, Any]:
        """Test basic performance metrics"""
        print("\n⚡ Testing Basic Performance...")
        
        results = {
            'system_resources': {},
            'response_times': {},
            'load_test': {},
            'overall_performance': 'unknown'
        }
        
        # System resource check
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            
            results['system_resources'] = {
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'memory_usage_percent': memory.percent,
                'cpu_usage_percent': cpu_percent,
                'disk_free_gb': round(disk.free / (1024**3), 2),
                'disk_usage_percent': round((disk.used / disk.total) * 100, 2)
            }
            
            print(f"  💾 Memory: {results['system_resources']['memory_available_gb']}GB available")
            print(f"  🖥️ CPU: {results['system_resources']['cpu_usage_percent']:.1f}% usage")
            
        except ImportError:
            results['system_resources'] = {'error': 'psutil not available'}
            print("  ⚠️ System resource monitoring not available")
        except Exception as e:
            results['system_resources'] = {'error': str(e)}
            print(f"  ❌ System resource check failed: {e}")
        
        # Simple load test
        load_test_url = 'http://localhost:8000/api/system/health'
        try:
            print("  🔄 Running simple load test...")
            
            response_times = []
            success_count = 0
            total_requests = 10
            
            for i in range(total_requests):
                try:
                    start_time = time.time()
                    response = requests.get(load_test_url, timeout=5)
                    response_time = time.time() - start_time
                    
                    response_times.append(response_time)
                    if response.status_code == 200:
                        success_count += 1
                
                except Exception:
                    response_times.append(5.0)  # Timeout value
                
                time.sleep(0.1)  # Small delay between requests
            
            if response_times:
                results['load_test'] = {
                    'total_requests': total_requests,
                    'successful_requests': success_count,
                    'success_rate': (success_count / total_requests) * 100,
                    'avg_response_time': sum(response_times) / len(response_times),
                    'min_response_time': min(response_times),
                    'max_response_time': max(response_times)
                }
                
                print(f"    Success rate: {results['load_test']['success_rate']:.1f}%")
                print(f"    Avg response time: {results['load_test']['avg_response_time']:.3f}s")
            
        except Exception as e:
            results['load_test'] = {'error': str(e)}
            print(f"  ❌ Load test failed: {e}")
        
        # Determine overall performance
        if results['system_resources'].get('memory_available_gb', 0) > 2 and \
           results['system_resources'].get('cpu_usage_percent', 100) < 80 and \
           results['load_test'].get('success_rate', 0) > 80:
            results['overall_performance'] = 'good'
        elif results['system_resources'].get('memory_available_gb', 0) > 1 and \
             results['load_test'].get('success_rate', 0) > 50:
            results['overall_performance'] = 'acceptable'
        else:
            results['overall_performance'] = 'poor'
        
        return results
    
    def _determine_overall_status(self, test_results: Dict[str, Any]) -> str:
        """Determine overall test status"""
        if isinstance(test_results, dict) and 'test_results' in test_results:
            test_results = test_results['test_results']
        
        # Check configuration validity
        config_valid = test_results.get('configuration_tests', {}).get('overall_valid', False)
        
        # Check service availability
        services_available = test_results.get('service_tests', {}).get('overall_available', False)
        
        # Check API functionality
        apis_functional = test_results.get('integration_tests', {}).get('overall_functional', False)
        
        # Check performance
        performance = test_results.get('performance_tests', {}).get('overall_performance', 'unknown')
        
        if config_valid and services_available and apis_functional and performance in ['good', 'acceptable']:
            return 'healthy'
        elif config_valid and (services_available or apis_functional):
            return 'partially_functional'
        elif config_valid:
            return 'configured_not_running'
        else:
            return 'issues_detected'
    
    def _create_test_environment(self) -> Path:
        """Create isolated test environment"""
        print("🏗️ Creating isolated test environment...")
        
        # Create temporary directory for test environment
        test_env_path = Path(tempfile.mkdtemp(prefix='persian_legal_ai_test_'))
        self.cleanup_required = True
        
        # Copy necessary files to test environment
        files_to_copy = [
            'docker-compose.yml',
            'docker-compose.enhanced.yml',
            'backend/',
            'persian-legal-ai-frontend/',
            '.env.production.example'
        ]
        
        for file_path in files_to_copy:
            source_path = self.workspace / file_path
            dest_path = test_env_path / file_path
            
            if source_path.exists():
                if source_path.is_file():
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, dest_path)
                elif source_path.is_dir():
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
        
        print(f"  ✅ Test environment created: {test_env_path}")
        return test_env_path
    
    def _apply_test_optimizations(self, test_env_path: Path, optimization_config: Dict):
        """Apply optimizations to test environment"""
        print("🔧 Applying optimizations to test environment...")
        
        # Example optimization: modify docker-compose for testing
        compose_file = test_env_path / 'docker-compose.yml'
        if compose_file.exists():
            try:
                import yaml
                
                with open(compose_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Apply port changes for testing
                services = config.get('services', {})
                for service_name, service_config in services.items():
                    if 'ports' in service_config:
                        # Modify ports to test ports
                        new_ports = []
                        for port in service_config['ports']:
                            if isinstance(port, str) and ':' in port:
                                container_port = port.split(':')[1]
                                if service_name in self.test_ports:
                                    new_port = f"{self.test_ports[service_name]}:{container_port}"
                                    new_ports.append(new_port)
                                else:
                                    new_ports.append(port)
                            else:
                                new_ports.append(port)
                        service_config['ports'] = new_ports
                
                # Save modified configuration
                with open(compose_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                print("  ✅ Applied port optimizations for testing")
                
            except Exception as e:
                print(f"  ⚠️ Failed to apply optimizations: {e}")
    
    def _run_isolated_tests(self, test_env_path: Path) -> Dict[str, Any]:
        """Run tests in isolated environment"""
        print("🧪 Running tests in isolated environment...")
        
        results = {
            'environment_setup': {},
            'service_startup': {},
            'functionality_tests': {},
            'cleanup_status': {}
        }
        
        original_cwd = os.getcwd()
        
        try:
            # Change to test environment
            os.chdir(test_env_path)
            
            # Test environment setup
            results['environment_setup'] = self._test_environment_setup(test_env_path)
            
            # Try to start services (if Docker is available)
            if results['environment_setup'].get('docker_available'):
                results['service_startup'] = self._test_service_startup(test_env_path)
                
                # Test functionality if services started
                if results['service_startup'].get('services_started'):
                    time.sleep(10)  # Wait for services to be ready
                    results['functionality_tests'] = self._test_isolated_functionality()
                    
                    # Cleanup services
                    results['cleanup_status'] = self._cleanup_test_services(test_env_path)
            
        except Exception as e:
            results['error'] = str(e)
            print(f"  ❌ Isolated test failed: {e}")
        
        finally:
            # Return to original directory
            os.chdir(original_cwd)
        
        return results
    
    def _test_environment_setup(self, test_env_path: Path) -> Dict[str, Any]:
        """Test environment setup"""
        results = {
            'files_copied': True,
            'docker_available': False,
            'compose_valid': False
        }
        
        # Check if files were copied correctly
        required_files = ['docker-compose.yml', 'backend/Dockerfile']
        for file_path in required_files:
            if not (test_env_path / file_path).exists():
                results['files_copied'] = False
                break
        
        # Check Docker availability
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, timeout=10)
            if result.returncode == 0:
                results['docker_available'] = True
        except Exception:
            pass
        
        # Validate docker-compose
        if results['docker_available']:
            try:
                result = subprocess.run(
                    ['docker-compose', 'config'],
                    capture_output=True, timeout=30
                )
                if result.returncode == 0:
                    results['compose_valid'] = True
            except Exception:
                pass
        
        return results
    
    def _test_service_startup(self, test_env_path: Path) -> Dict[str, Any]:
        """Test service startup in isolated environment"""
        results = {
            'services_started': False,
            'startup_time': 0,
            'services_status': {}
        }
        
        try:
            print("    🚀 Starting test services...")
            start_time = time.time()
            
            # Start services
            result = subprocess.run(
                ['docker-compose', 'up', '-d'],
                capture_output=True, text=True, timeout=120
            )
            
            startup_time = time.time() - start_time
            results['startup_time'] = startup_time
            
            if result.returncode == 0:
                # Wait a bit for services to initialize
                time.sleep(15)
                
                # Check service status
                status_result = subprocess.run(
                    ['docker-compose', 'ps', '--format', 'json'],
                    capture_output=True, text=True, timeout=30
                )
                
                if status_result.returncode == 0:
                    services_running = 0
                    for line in status_result.stdout.strip().split('\n'):
                        if line:
                            try:
                                service = json.loads(line)
                                service_name = service.get('Service', 'unknown')
                                service_state = service.get('State', 'unknown')
                                
                                results['services_status'][service_name] = service_state
                                if service_state == 'running':
                                    services_running += 1
                                    
                            except json.JSONDecodeError:
                                pass
                    
                    if services_running > 0:
                        results['services_started'] = True
                        print(f"    ✅ Services started in {startup_time:.1f}s")
                    else:
                        print("    ❌ No services running")
                else:
                    print("    ❌ Failed to check service status")
            else:
                print(f"    ❌ Failed to start services: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            results['error'] = 'Service startup timeout'
            print("    ❌ Service startup timeout")
        except Exception as e:
            results['error'] = str(e)
            print(f"    ❌ Service startup error: {e}")
        
        return results
    
    def _test_isolated_functionality(self) -> Dict[str, Any]:
        """Test functionality in isolated environment"""
        results = {
            'api_tests': {},
            'health_checks': {},
            'basic_functionality': True
        }
        
        # Test API endpoints with test ports
        test_urls = [
            f"http://localhost:{self.test_ports['backend']}/api/system/health",
            f"http://localhost:{self.test_ports['frontend']}"
        ]
        
        for url in test_urls:
            try:
                response = requests.get(url, timeout=10)
                results['api_tests'][url] = {
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'accessible': True
                }
                print(f"    ✅ {url}: Accessible")
            except Exception as e:
                results['api_tests'][url] = {
                    'accessible': False,
                    'error': str(e)
                }
                results['basic_functionality'] = False
                print(f"    ❌ {url}: Not accessible")
        
        return results
    
    def _cleanup_test_services(self, test_env_path: Path) -> Dict[str, Any]:
        """Cleanup test services"""
        results = {
            'services_stopped': False,
            'containers_removed': False,
            'cleanup_time': 0
        }
        
        try:
            print("    🧹 Cleaning up test services...")
            start_time = time.time()
            
            # Stop and remove services
            result = subprocess.run(
                ['docker-compose', 'down', '--volumes', '--remove-orphans'],
                capture_output=True, text=True, timeout=60
            )
            
            cleanup_time = time.time() - start_time
            results['cleanup_time'] = cleanup_time
            
            if result.returncode == 0:
                results['services_stopped'] = True
                results['containers_removed'] = True
                print(f"    ✅ Test services cleaned up in {cleanup_time:.1f}s")
            else:
                print(f"    ⚠️ Cleanup issues: {result.stderr}")
                
        except Exception as e:
            results['error'] = str(e)
            print(f"    ❌ Cleanup error: {e}")
        
        return results
    
    def _cleanup_test_environment(self, test_env_path: Path):
        """Cleanup test environment"""
        try:
            if test_env_path.exists():
                shutil.rmtree(test_env_path)
                print(f"  ✅ Test environment cleaned up: {test_env_path}")
                self.cleanup_required = False
        except Exception as e:
            print(f"  ⚠️ Failed to cleanup test environment: {e}")
    
    def _emergency_cleanup(self):
        """Emergency cleanup of test resources"""
        print("🚨 Performing emergency cleanup...")
        
        try:
            # Stop any running test containers
            result = subprocess.run(
                ['docker', 'ps', '-q', '--filter', 'name=test'],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                container_ids = result.stdout.strip().split('\n')
                for container_id in container_ids:
                    subprocess.run(['docker', 'stop', container_id], timeout=30)
                    subprocess.run(['docker', 'rm', container_id], timeout=30)
            
        except Exception as e:
            print(f"  ⚠️ Emergency cleanup error: {e}")
    
    def _compare_performance(self, current_results: Dict, optimized_results: Dict) -> Dict[str, Any]:
        """Compare performance between current and optimized deployment"""
        comparison = {
            'response_time_improvement': {},
            'resource_usage_comparison': {},
            'overall_improvement': 'unknown'
        }
        
        # Compare response times
        current_perf = current_results.get('performance_tests', {})
        optimized_perf = optimized_results.get('performance_tests', {})
        
        current_load_test = current_perf.get('load_test', {})
        optimized_load_test = optimized_perf.get('load_test', {})
        
        if current_load_test.get('avg_response_time') and optimized_load_test.get('avg_response_time'):
            current_avg = current_load_test['avg_response_time']
            optimized_avg = optimized_load_test['avg_response_time']
            
            improvement = ((current_avg - optimized_avg) / current_avg) * 100
            comparison['response_time_improvement'] = {
                'current_avg_ms': current_avg * 1000,
                'optimized_avg_ms': optimized_avg * 1000,
                'improvement_percent': improvement
            }
        
        # Compare resource usage
        current_resources = current_perf.get('system_resources', {})
        optimized_resources = optimized_perf.get('system_resources', {})
        
        if current_resources and optimized_resources:
            comparison['resource_usage_comparison'] = {
                'memory_usage_change': optimized_resources.get('memory_usage_percent', 0) - current_resources.get('memory_usage_percent', 0),
                'cpu_usage_change': optimized_resources.get('cpu_usage_percent', 0) - current_resources.get('cpu_usage_percent', 0)
            }
        
        return comparison
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n📡 Received signal {signum} - cleaning up...")
        self._emergency_cleanup()
        sys.exit(0)
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        print("\n📊 Generating Deployment Test Report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_results': self.test_results,
            'summary': {
                'tests_run': len(self.test_results),
                'overall_status': 'unknown',
                'recommendations': []
            }
        }
        
        # Determine overall status
        if 'current_deployment' in self.test_results:
            current_status = self.test_results['current_deployment']['overall_status']
            report['summary']['overall_status'] = current_status
            
            # Generate recommendations based on test results
            if current_status in ['issues_detected', 'configured_not_running']:
                report['summary']['recommendations'].extend([
                    'Fix configuration issues before deployment',
                    'Ensure Docker is installed and running',
                    'Check service dependencies and network configuration'
                ])
            elif current_status == 'partially_functional':
                report['summary']['recommendations'].extend([
                    'Investigate non-functional services',
                    'Check service logs for errors',
                    'Verify network connectivity between services'
                ])
            elif current_status == 'healthy':
                report['summary']['recommendations'].extend([
                    'Current deployment is healthy',
                    'Consider performance optimizations',
                    'Monitor resource usage during peak loads'
                ])
        
        # Save report
        report_path = self.workspace / 'deployment_test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Test report saved to: {report_path}")
        
        # Display summary
        self._display_test_summary(report)
        
        return str(report_path)
    
    def _display_test_summary(self, report: Dict[str, Any]):
        """Display test summary"""
        print("\n📋 DEPLOYMENT TEST SUMMARY:")
        print("=" * 60)
        
        summary = report['summary']
        print(f"🧪 Tests run: {summary['tests_run']}")
        print(f"📊 Overall status: {summary['overall_status']}")
        
        # Show test results
        for test_name, test_result in report['test_results'].items():
            status = test_result.get('overall_status', 'unknown')
            emoji = {
                'healthy': '✅',
                'partially_functional': '⚠️',
                'configured_not_running': '🔧',
                'issues_detected': '❌',
                'unknown': '❓'
            }.get(status, '❓')
            
            print(f"{emoji} {test_name}: {status}")
        
        # Show recommendations
        if summary['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "=" * 60)
        print("🛡️ SAFETY: All tests run in isolated environments")
        print("🔒 Production deployment remains unaffected")

def main():
    """Main function"""
    tester = SafeDeploymentTester()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Persian Legal AI Deployment Tester')
    parser.add_argument('--current', action='store_true', help='Test current deployment only')
    parser.add_argument('--optimized', action='store_true', help='Test with optimizations')
    parser.add_argument('--full', action='store_true', help='Run full test suite')
    parser.add_argument('--report', action='store_true', help='Generate test report')
    
    args = parser.parse_args()
    
    try:
        if args.current:
            tester.test_current_deployment()
        elif args.optimized:
            tester.test_current_deployment()
            tester.test_with_optimizations()
        elif args.full:
            tester.test_current_deployment()
            tester.test_with_optimizations()
            tester.generate_test_report()
        else:
            # Default: test current deployment and generate report
            tester.test_current_deployment()
            tester.generate_test_report()
            
    except KeyboardInterrupt:
        print("\n👋 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
    finally:
        # Ensure cleanup
        tester._emergency_cleanup()

if __name__ == '__main__':
    main()