#!/usr/bin/env python3
"""
Deployment Health Monitor for Persian Legal AI
Monitors deployment health without changing existing code

🛡️ SAFETY: This monitor only observes and reports - no modifications
"""

import time
import json
import sqlite3
import requests
import psutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import threading
import signal
import sys

@dataclass
class HealthStatus:
    """Health status data structure"""
    service: str
    status: str  # healthy, unhealthy, unreachable, unknown
    response_time: float
    details: str
    timestamp: datetime
    error_count: int = 0

class DeploymentHealthMonitor:
    """Monitors deployment health without interfering with operations"""
    
    def __init__(self):
        self.workspace = Path("/workspace")
        self.original_files = self.scan_original_files()
        self.health_history = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Service endpoints
        self.services = {
            'backend': {
                'url': 'http://localhost:8000/api/system/health',
                'timeout': 10,
                'expected_status': 200
            },
            'frontend': {
                'url': 'http://localhost:3000',
                'timeout': 5,
                'expected_status': 200
            },
            'redis': {
                'container': 'persian-legal-redis',
                'command': ['redis-cli', 'ping'],
                'expected_output': 'PONG'
            }
        }
        
        # Enhanced services (if using enhanced docker-compose)
        self.enhanced_services = {
            'backend-enhanced': {
                'url': 'http://localhost:8001/api/system/health',
                'timeout': 10,
                'expected_status': 200
            },
            'frontend-enhanced': {
                'url': 'http://localhost:3001',
                'timeout': 5,
                'expected_status': 200
            },
            'redis-enhanced': {
                'container': 'persian-legal-redis-enhanced',
                'command': ['redis-cli', 'ping'],
                'expected_output': 'PONG'
            }
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def scan_original_files(self) -> Dict[str, List[Path]]:
        """Catalog all existing files to ensure nothing is lost"""
        print("📁 Scanning original files to ensure integrity...")
        
        original_files = {
            'backend_files': [],
            'frontend_files': [],
            'config_files': [],
            'model_files': [],
            'data_files': []
        }
        
        try:
            # Backend files
            backend_dir = self.workspace / "backend"
            if backend_dir.exists():
                for py_file in backend_dir.glob("**/*.py"):
                    if py_file.is_file():
                        original_files['backend_files'].append(py_file)
            
            # Frontend files
            frontend_dirs = ["frontend", "persian-legal-ai-frontend"]
            for frontend_dir in frontend_dirs:
                frontend_path = self.workspace / frontend_dir
                if frontend_path.exists():
                    for file_pattern in ["**/*.js", "**/*.ts", "**/*.tsx", "**/*.jsx", "**/*.json"]:
                        for file_path in frontend_path.glob(file_pattern):
                            if file_path.is_file() and "node_modules" not in str(file_path):
                                original_files['frontend_files'].append(file_path)
            
            # Configuration files
            config_patterns = ["*.yml", "*.yaml", "*.json", "*.env*", "Dockerfile*"]
            for pattern in config_patterns:
                for config_file in self.workspace.glob(pattern):
                    if config_file.is_file():
                        original_files['config_files'].append(config_file)
            
            # Model files
            model_dirs = ["models", "ai_models"]
            for model_dir in model_dirs:
                model_path = self.workspace / model_dir
                if model_path.exists():
                    for model_file in model_path.glob("**/*"):
                        if model_file.is_file():
                            original_files['model_files'].append(model_file)
            
            # Data files
            data_patterns = ["*.db", "data/**/*"]
            for pattern in data_patterns:
                for data_file in self.workspace.glob(pattern):
                    if data_file.is_file():
                        original_files['data_files'].append(data_file)
            
            total_files = sum(len(files) for files in original_files.values())
            print(f"✅ Cataloged {total_files} original files")
            
        except Exception as e:
            print(f"⚠️ Error scanning files: {e}")
        
        return original_files
    
    def verify_integrity(self) -> Dict[str, Any]:
        """Ensure no original files were modified or deleted"""
        print("🔍 Verifying file integrity...")
        
        integrity_report = {
            'timestamp': datetime.now().isoformat(),
            'files_intact': True,
            'missing_files': [],
            'modified_files': [],
            'total_original_files': 0,
            'total_current_files': 0
        }
        
        try:
            current_files = self.scan_original_files()
            
            # Compare file counts
            for category, original_list in self.original_files.items():
                current_list = current_files.get(category, [])
                
                integrity_report['total_original_files'] += len(original_list)
                integrity_report['total_current_files'] += len(current_list)
                
                # Check for missing files
                original_paths = set(str(f) for f in original_list)
                current_paths = set(str(f) for f in current_list)
                
                missing = original_paths - current_paths
                if missing:
                    integrity_report['missing_files'].extend(list(missing))
                    integrity_report['files_intact'] = False
            
            if integrity_report['files_intact']:
                print("✅ All original files intact")
            else:
                print(f"⚠️ {len(integrity_report['missing_files'])} files missing")
                
        except Exception as e:
            print(f"❌ Integrity check failed: {e}")
            integrity_report['files_intact'] = False
            integrity_report['error'] = str(e)
        
        return integrity_report
    
    def check_service_health(self, service_name: str, config: Dict) -> HealthStatus:
        """Check health of a single service"""
        start_time = time.time()
        
        try:
            if 'url' in config:
                # HTTP-based health check
                response = requests.get(
                    config['url'],
                    timeout=config.get('timeout', 10),
                    headers={'User-Agent': 'PersianLegalAI-HealthMonitor/1.0'}
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == config.get('expected_status', 200):
                    return HealthStatus(
                        service=service_name,
                        status='healthy',
                        response_time=response_time,
                        details=f"HTTP {response.status_code}",
                        timestamp=datetime.now()
                    )
                else:
                    return HealthStatus(
                        service=service_name,
                        status='unhealthy',
                        response_time=response_time,
                        details=f"HTTP {response.status_code}",
                        timestamp=datetime.now(),
                        error_count=1
                    )
            
            elif 'container' in config:
                # Container-based health check
                cmd = ['docker', 'exec', config['container']] + config['command']
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=config.get('timeout', 10)
                )
                
                response_time = time.time() - start_time
                
                if result.returncode == 0 and config.get('expected_output', '') in result.stdout:
                    return HealthStatus(
                        service=service_name,
                        status='healthy',
                        response_time=response_time,
                        details=result.stdout.strip(),
                        timestamp=datetime.now()
                    )
                else:
                    return HealthStatus(
                        service=service_name,
                        status='unhealthy',
                        response_time=response_time,
                        details=result.stderr.strip() or result.stdout.strip(),
                        timestamp=datetime.now(),
                        error_count=1
                    )
            
        except requests.exceptions.RequestException as e:
            return HealthStatus(
                service=service_name,
                status='unreachable',
                response_time=time.time() - start_time,
                details=str(e),
                timestamp=datetime.now(),
                error_count=1
            )
        except subprocess.TimeoutExpired:
            return HealthStatus(
                service=service_name,
                status='unreachable',
                response_time=time.time() - start_time,
                details='Timeout',
                timestamp=datetime.now(),
                error_count=1
            )
        except Exception as e:
            return HealthStatus(
                service=service_name,
                status='unknown',
                response_time=time.time() - start_time,
                details=str(e),
                timestamp=datetime.now(),
                error_count=1
            )
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Load average (Linux/Unix)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                # Windows doesn't have load average
                pass
            
            return {
                'memory': {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2),
                    'used_percent': memory.percent
                },
                'cpu': {
                    'usage_percent': cpu_percent,
                    'core_count': psutil.cpu_count()
                },
                'disk': {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'used_percent': round((disk.used / disk.total) * 100, 2)
                },
                'load_average': load_avg,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def check_docker_status(self) -> Dict[str, Any]:
        """Check Docker container status"""
        docker_status = {
            'docker_available': False,
            'containers': {},
            'networks': {},
            'volumes': {}
        }
        
        try:
            # Check if Docker is available
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                docker_status['docker_available'] = True
                docker_status['docker_version'] = result.stdout.strip()
            
            # List containers
            result = subprocess.run(
                ['docker', 'ps', '--format', 'json'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            container = json.loads(line)
                            docker_status['containers'][container['Names']] = {
                                'status': container['State'],
                                'image': container['Image'],
                                'ports': container.get('Ports', ''),
                                'created': container.get('CreatedAt', '')
                            }
                        except json.JSONDecodeError:
                            pass
            
            # List networks
            result = subprocess.run(
                ['docker', 'network', 'ls', '--format', 'json'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            network = json.loads(line)
                            docker_status['networks'][network['Name']] = {
                                'driver': network['Driver'],
                                'scope': network['Scope']
                            }
                        except json.JSONDecodeError:
                            pass
            
        except FileNotFoundError:
            docker_status['error'] = 'Docker not installed'
        except Exception as e:
            docker_status['error'] = str(e)
        
        return docker_status
    
    def monitor_services(self, interval: int = 30) -> List[HealthStatus]:
        """Monitor all services and return health status"""
        print(f"🔍 Health Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        health_statuses = []
        
        # Determine which services to check
        services_to_check = self.services.copy()
        
        # Check if enhanced services are running
        try:
            result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], capture_output=True, text=True)
            if result.returncode == 0:
                running_containers = result.stdout.strip().split('\n')
                if any('enhanced' in container for container in running_containers):
                    services_to_check.update(self.enhanced_services)
        except:
            pass
        
        # Check each service
        for service_name, config in services_to_check.items():
            status = self.check_service_health(service_name, config)
            health_statuses.append(status)
            
            # Display status
            emoji = {
                'healthy': '✅',
                'unhealthy': '❌',
                'unreachable': '⚠️',
                'unknown': '❓'
            }.get(status.status, '❓')
            
            print(f"{emoji} {service_name.ljust(20)} {status.status.ljust(12)} ({status.response_time:.3f}s)")
            if status.details and status.status != 'healthy':
                print(f"   Details: {status.details}")
        
        # System resources
        print("\n💾 System Resources:")
        resources = self.check_system_resources()
        if 'error' not in resources:
            print(f"   Memory: {resources['memory']['available_gb']:.1f}GB available ({resources['memory']['used_percent']:.1f}% used)")
            print(f"   CPU: {resources['cpu']['usage_percent']:.1f}% usage")
            print(f"   Disk: {resources['disk']['free_gb']:.1f}GB free ({resources['disk']['used_percent']:.1f}% used)")
        
        # Docker status
        print("\n🐳 Docker Status:")
        docker_status = self.check_docker_status()
        if docker_status['docker_available']:
            container_count = len(docker_status['containers'])
            running_count = sum(1 for c in docker_status['containers'].values() if c['status'] == 'running')
            print(f"   Containers: {running_count}/{container_count} running")
        else:
            print(f"   ❌ Docker not available: {docker_status.get('error', 'Unknown error')}")
        
        # Overall health summary
        healthy_count = sum(1 for s in health_statuses if s.status == 'healthy')
        total_count = len(health_statuses)
        
        print("\n" + "=" * 70)
        if healthy_count == total_count:
            print("🎉 All services healthy!")
        else:
            print(f"⚠️ {healthy_count}/{total_count} services healthy")
        
        # Store in history
        self.health_history.append({
            'timestamp': datetime.now().isoformat(),
            'services': [
                {
                    'service': s.service,
                    'status': s.status,
                    'response_time': s.response_time,
                    'details': s.details
                } for s in health_statuses
            ],
            'system_resources': resources,
            'docker_status': docker_status
        })
        
        # Keep only last 100 entries
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        return health_statuses
    
    def generate_health_report(self) -> str:
        """Generate comprehensive health report"""
        report_path = self.workspace / "health_report.json"
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'integrity_check': self.verify_integrity(),
            'latest_health_check': self.health_history[-1] if self.health_history else None,
            'health_history_summary': {
                'total_checks': len(self.health_history),
                'time_range': {
                    'start': self.health_history[0]['timestamp'] if self.health_history else None,
                    'end': self.health_history[-1]['timestamp'] if self.health_history else None
                }
            },
            'service_uptime_stats': self._calculate_uptime_stats()
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Health report saved to: {report_path}")
        return str(report_path)
    
    def _calculate_uptime_stats(self) -> Dict[str, Any]:
        """Calculate service uptime statistics"""
        if not self.health_history:
            return {}
        
        stats = {}
        
        # Group by service
        service_data = {}
        for entry in self.health_history:
            for service in entry['services']:
                service_name = service['service']
                if service_name not in service_data:
                    service_data[service_name] = []
                service_data[service_name].append({
                    'timestamp': entry['timestamp'],
                    'status': service['status'],
                    'response_time': service['response_time']
                })
        
        # Calculate stats for each service
        for service_name, data in service_data.items():
            healthy_count = sum(1 for d in data if d['status'] == 'healthy')
            total_count = len(data)
            
            avg_response_time = sum(d['response_time'] for d in data) / total_count if data else 0
            
            stats[service_name] = {
                'uptime_percentage': (healthy_count / total_count * 100) if total_count > 0 else 0,
                'total_checks': total_count,
                'healthy_checks': healthy_count,
                'average_response_time': round(avg_response_time, 3),
                'last_status': data[-1]['status'] if data else 'unknown'
            }
        
        return stats
    
    def start_continuous_monitoring(self, interval: int = 30):
        """Start continuous monitoring in a separate thread"""
        print(f"🔄 Starting continuous health monitoring (every {interval}s)")
        print("Press Ctrl+C to stop")
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self.monitor_services(interval)
                    
                    if self.monitoring_active:
                        print(f"\n⏰ Next check in {interval} seconds...\n")
                        
                        # Sleep in small intervals to allow for interruption
                        for _ in range(interval):
                            if not self.monitoring_active:
                                break
                            time.sleep(1)
                            
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
                    time.sleep(5)  # Brief pause before retrying
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        try:
            # Keep main thread alive
            while self.monitoring_active:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        print("\n🛑 Stopping health monitoring...")
        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        # Generate final report
        if self.health_history:
            self.generate_health_report()
        
        print("👋 Health monitoring stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n📡 Received signal {signum}")
        self.stop_monitoring()
        sys.exit(0)

def main():
    """Main function"""
    monitor = DeploymentHealthMonitor()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Persian Legal AI Health Monitor')
    parser.add_argument('--monitor', action='store_true', help='Start continuous monitoring')
    parser.add_argument('--interval', type=int, default=30, help='Monitoring interval in seconds')
    parser.add_argument('--report', action='store_true', help='Generate health report')
    parser.add_argument('--integrity', action='store_true', help='Check file integrity only')
    
    args = parser.parse_args()
    
    if args.integrity:
        monitor.verify_integrity()
    elif args.report:
        monitor.monitor_services()
        monitor.generate_health_report()
    elif args.monitor:
        monitor.start_continuous_monitoring(args.interval)
    else:
        # Single health check
        monitor.monitor_services()

if __name__ == '__main__':
    main()