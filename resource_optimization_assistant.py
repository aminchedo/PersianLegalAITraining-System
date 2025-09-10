#!/usr/bin/env python3
"""
Resource Optimization Assistant for Persian Legal AI
Suggests optimizations without making changes

🛡️ SAFETY: This assistant only suggests improvements - no automatic modifications
"""

import os
import json
import psutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import yaml

class ResourceOptimizationAssistant:
    """Suggests optimizations without making changes"""
    
    def __init__(self):
        self.workspace = Path("/workspace")
        self.suggestions = []
        self.current_config = {}
        self.analysis_results = {}
        
    def analyze_current_setup(self) -> Dict[str, Any]:
        """Analyze without modifying"""
        print("🔍 Analyzing Current Resource Setup...")
        print("🛡️ SAFETY: Read-only analysis - no changes will be made")
        
        # Get system information
        self.current_config = self._get_system_info()
        
        # Analyze Docker configuration
        docker_analysis = self._analyze_docker_config()
        
        # Analyze AI model requirements
        ai_analysis = self._analyze_ai_requirements()
        
        # Analyze database usage
        db_analysis = self._analyze_database_usage()
        
        # Analyze network configuration
        network_analysis = self._analyze_network_config()
        
        self.analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.current_config,
            'docker_analysis': docker_analysis,
            'ai_analysis': ai_analysis,
            'database_analysis': db_analysis,
            'network_analysis': network_analysis
        }
        
        return self.analysis_results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            disk = psutil.disk_usage('/')
            
            # Get CPU info
            cpu_freq = psutil.cpu_freq()
            
            # Get load average (Linux/Unix)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                pass
            
            system_info = {
                'memory': {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2),
                    'used_gb': round(memory.used / (1024**3), 2),
                    'percent_used': memory.percent
                },
                'cpu': {
                    'logical_cores': cpu_count,
                    'physical_cores': psutil.cpu_count(logical=False),
                    'current_freq_mhz': cpu_freq.current if cpu_freq else None,
                    'max_freq_mhz': cpu_freq.max if cpu_freq else None,
                    'load_average': load_avg
                },
                'disk': {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'used_gb': round(disk.used / (1024**3), 2),
                    'percent_used': round((disk.used / disk.total) * 100, 2)
                }
            }
            
            print(f"💾 System Memory: {system_info['memory']['total_gb']}GB total, {system_info['memory']['available_gb']}GB available")
            print(f"🖥️ CPU: {system_info['cpu']['logical_cores']} cores ({system_info['cpu']['physical_cores']} physical)")
            print(f"💽 Disk Space: {system_info['disk']['free_gb']}GB free ({system_info['disk']['percent_used']:.1f}% used)")
            
            return system_info
            
        except Exception as e:
            print(f"⚠️ Error getting system info: {e}")
            return {'error': str(e)}
    
    def _analyze_docker_config(self) -> Dict[str, Any]:
        """Analyze Docker configuration for optimization opportunities"""
        analysis = {
            'compose_files': [],
            'resource_limits': {},
            'optimization_opportunities': []
        }
        
        # Check docker-compose files
        compose_files = [
            'docker-compose.yml',
            'docker-compose.production.yml',
            'docker-compose.enhanced.yml'
        ]
        
        for compose_file in compose_files:
            compose_path = self.workspace / compose_file
            if compose_path.exists():
                try:
                    with open(compose_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    analysis['compose_files'].append(compose_file)
                    
                    # Analyze resource limits
                    services = config.get('services', {})
                    for service_name, service_config in services.items():
                        deploy_config = service_config.get('deploy', {})
                        resources = deploy_config.get('resources', {})
                        
                        analysis['resource_limits'][service_name] = {
                            'memory_limit': resources.get('limits', {}).get('memory'),
                            'cpu_limit': resources.get('limits', {}).get('cpus'),
                            'memory_reservation': resources.get('reservations', {}).get('memory'),
                            'cpu_reservation': resources.get('reservations', {}).get('cpus')
                        }
                        
                        # Check for optimization opportunities
                        if not resources.get('limits'):
                            analysis['optimization_opportunities'].append(
                                f"Service '{service_name}' has no resource limits - add limits to prevent resource exhaustion"
                            )
                        
                        if not service_config.get('healthcheck'):
                            analysis['optimization_opportunities'].append(
                                f"Service '{service_name}' has no health check - add health checks for better reliability"
                            )
                
                except Exception as e:
                    print(f"⚠️ Error analyzing {compose_file}: {e}")
        
        return analysis
    
    def _analyze_ai_requirements(self) -> Dict[str, Any]:
        """Analyze AI model requirements and suggest optimizations"""
        analysis = {
            'models_found': [],
            'memory_requirements': {},
            'optimization_suggestions': []
        }
        
        # Check for AI models
        model_dirs = ['models', 'ai_models', 'backend/models']
        for model_dir in model_dirs:
            model_path = self.workspace / model_dir
            if model_path.exists():
                model_files = list(model_path.glob('**/*'))
                if model_files:
                    analysis['models_found'].append(str(model_dir))
        
        # Check requirements.txt for AI dependencies
        requirements_file = self.workspace / 'backend' / 'requirements.txt'
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    content = f.read()
                
                # Analyze AI dependencies
                ai_packages = {
                    'torch': 'PyTorch - Large memory footprint, consider quantization',
                    'transformers': 'Transformers - Memory-intensive, enable model caching',
                    'tensorflow': 'TensorFlow - GPU acceleration recommended',
                    'scikit-learn': 'Scikit-learn - CPU-intensive, consider parallel processing'
                }
                
                for package, description in ai_packages.items():
                    if package in content.lower():
                        analysis['memory_requirements'][package] = description
                
                # Memory optimization suggestions based on available RAM
                available_memory = self.current_config.get('memory', {}).get('available_gb', 0)
                
                if available_memory < 4:
                    analysis['optimization_suggestions'].extend([
                        "⚠️ Low memory detected - Enable model quantization (8-bit or 16-bit)",
                        "💡 Use gradient checkpointing to reduce memory usage",
                        "🔄 Implement model pruning to reduce model size",
                        "📁 Use memory-mapped files for large datasets",
                        "🔢 Reduce batch size for training/inference"
                    ])
                elif available_memory < 8:
                    analysis['optimization_suggestions'].extend([
                        "💡 Consider mixed precision training (float16)",
                        "🔄 Enable gradient accumulation for larger effective batch sizes",
                        "📊 Monitor memory usage during peak loads",
                        "💾 Use model caching to avoid reloading"
                    ])
                else:
                    analysis['optimization_suggestions'].extend([
                        "✅ Sufficient memory available for AI workloads",
                        "🚀 Consider increasing batch sizes for better throughput",
                        "💾 Enable aggressive model caching",
                        "⚡ Consider GPU acceleration if available"
                    ])
                
            except Exception as e:
                print(f"⚠️ Error analyzing AI requirements: {e}")
        
        return analysis
    
    def _analyze_database_usage(self) -> Dict[str, Any]:
        """Analyze database configuration and suggest optimizations"""
        analysis = {
            'databases_found': [],
            'sizes': {},
            'optimization_suggestions': []
        }
        
        # Check for database files
        db_files = [
            'persian_legal_ai.db',
            'data/persian_legal_ai.db',
            'backend/data/persian_legal_ai.db'
        ]
        
        for db_file in db_files:
            db_path = self.workspace / db_file
            if db_path.exists():
                try:
                    size_mb = db_path.stat().st_size / (1024 * 1024)
                    analysis['databases_found'].append(db_file)
                    analysis['sizes'][db_file] = f"{size_mb:.2f} MB"
                    
                    # SQLite-specific optimizations
                    if db_file.endswith('.db'):
                        if size_mb > 100:
                            analysis['optimization_suggestions'].extend([
                                f"Large SQLite database ({size_mb:.1f}MB) - consider VACUUM to reclaim space",
                                "Enable WAL mode for better concurrent access",
                                "Add indexes for frequently queried columns"
                            ])
                        
                        # Check if database can be analyzed
                        try:
                            import sqlite3
                            conn = sqlite3.connect(db_path)
                            cursor = conn.cursor()
                            
                            # Get table information
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                            tables = cursor.fetchall()
                            
                            if len(tables) > 10:
                                analysis['optimization_suggestions'].append(
                                    f"Database has {len(tables)} tables - consider table partitioning for large tables"
                                )
                            
                            conn.close()
                            
                        except Exception:
                            pass  # Skip detailed analysis if SQLite is not available
                
                except Exception as e:
                    print(f"⚠️ Error analyzing database {db_file}: {e}")
        
        if not analysis['databases_found']:
            analysis['optimization_suggestions'].append(
                "No database files found - database will be created on first run"
            )
        
        return analysis
    
    def _analyze_network_config(self) -> Dict[str, Any]:
        """Analyze network configuration for optimization"""
        analysis = {
            'ports_used': [],
            'network_configs': [],
            'optimization_suggestions': []
        }
        
        # Analyze docker-compose network configuration
        compose_files = ['docker-compose.yml', 'docker-compose.enhanced.yml']
        
        for compose_file in compose_files:
            compose_path = self.workspace / compose_file
            if compose_path.exists():
                try:
                    with open(compose_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    services = config.get('services', {})
                    networks = config.get('networks', {})
                    
                    # Collect port mappings
                    for service_name, service_config in services.items():
                        ports = service_config.get('ports', [])
                        for port in ports:
                            if isinstance(port, str) and ':' in port:
                                external_port = port.split(':')[0].replace('"', '')
                                analysis['ports_used'].append(f"{service_name}: {external_port}")
                    
                    # Analyze network configuration
                    for network_name, network_config in networks.items():
                        analysis['network_configs'].append({
                            'name': network_name,
                            'driver': network_config.get('driver', 'bridge'),
                            'subnet': network_config.get('ipam', {}).get('config', [{}])[0].get('subnet')
                        })
                    
                except Exception as e:
                    print(f"⚠️ Error analyzing network config in {compose_file}: {e}")
        
        # Network optimization suggestions
        if len(analysis['ports_used']) > 5:
            analysis['optimization_suggestions'].append(
                "Multiple services exposed - consider using a reverse proxy (nginx) for better security"
            )
        
        analysis['optimization_suggestions'].extend([
            "Consider using internal networks for service-to-service communication",
            "Enable connection pooling for database connections",
            "Add rate limiting to prevent abuse",
            "Configure proper CORS settings for production"
        ])
        
        return analysis
    
    def suggest_memory_optimizations(self) -> List[str]:
        """Suggest memory improvements based on current setup"""
        suggestions = []
        
        available_memory = self.current_config.get('memory', {}).get('available_gb', 0)
        used_percent = self.current_config.get('memory', {}).get('percent_used', 0)
        
        if available_memory < 2:
            suggestions.extend([
                "🚨 Critical: Very low memory available - immediate action required",
                "⚡ Reduce Docker container memory limits",
                "🔄 Enable swap if not already enabled",
                "📱 Close unnecessary applications",
                "🗂️ Use smaller base images in Dockerfiles"
            ])
        elif available_memory < 4:
            suggestions.extend([
                "⚠️ Low memory detected - optimize for memory efficiency",
                "🔢 Reduce AI model batch sizes",
                "💡 Enable model quantization (8-bit or 16-bit)",
                "🔄 Implement lazy loading for heavy dependencies",
                "📁 Use memory-mapped files for large datasets",
                "🗜️ Enable compression for data storage"
            ])
        elif available_memory < 8:
            suggestions.extend([
                "💡 Moderate memory available - consider optimizations",
                "🚀 Enable model caching for better performance",
                "🔄 Optimize batch processing for AI models",
                "📊 Monitor memory usage during peak loads",
                "💾 Consider increasing cache sizes"
            ])
        else:
            suggestions.extend([
                "✅ Sufficient memory available",
                "🚀 Consider enabling larger batch sizes for better throughput",
                "💾 Memory caching can be increased for better performance",
                "⚡ Enable aggressive prefetching and caching"
            ])
        
        # Memory usage suggestions
        if used_percent > 90:
            suggestions.append("🚨 Memory usage is very high - consider adding more RAM or optimizing applications")
        elif used_percent > 80:
            suggestions.append("⚠️ Memory usage is high - monitor for potential issues")
        
        return suggestions
    
    def suggest_cpu_optimizations(self) -> List[str]:
        """Suggest CPU optimizations"""
        suggestions = []
        
        cpu_info = self.current_config.get('cpu', {})
        logical_cores = cpu_info.get('logical_cores', 1)
        physical_cores = cpu_info.get('physical_cores', 1)
        
        suggestions.extend([
            f"🖥️ System has {logical_cores} logical cores ({physical_cores} physical)",
            f"💡 Configure worker processes to match CPU cores",
            f"🔄 Set TORCH_THREADS={min(physical_cores, 4)} for AI workloads"
        ])
        
        if logical_cores > physical_cores:
            suggestions.append("⚡ Hyperthreading detected - be cautious with CPU-intensive tasks")
        
        if logical_cores >= 4:
            suggestions.extend([
                "🚀 Multi-core system detected - enable parallel processing",
                "⚡ Consider using multiprocessing for CPU-intensive tasks",
                "🔄 Enable parallel data loading for AI models"
            ])
        else:
            suggestions.extend([
                "💡 Limited CPU cores - optimize for single-threaded performance",
                "🔄 Avoid CPU-intensive operations during peak usage"
            ])
        
        return suggestions
    
    def suggest_docker_optimizations(self) -> List[str]:
        """Suggest Docker-specific optimizations"""
        suggestions = []
        
        docker_analysis = self.analysis_results.get('docker_analysis', {})
        
        # General Docker optimizations
        suggestions.extend([
            "🐳 Use multi-stage builds to reduce image size",
            "📦 Optimize layer caching in Dockerfiles",
            "🔒 Run containers as non-root users for security",
            "💾 Use named volumes for persistent data",
            "🌐 Configure proper network isolation",
            "🔄 Set up container health checks",
            "📊 Add resource limits (memory, CPU) to prevent resource exhaustion",
            "🚀 Use init system in containers for proper signal handling",
            "📝 Add labels for better container management",
            "🔧 Optimize container startup time"
        ])
        
        # Specific suggestions based on analysis
        optimization_opportunities = docker_analysis.get('optimization_opportunities', [])
        if optimization_opportunities:
            suggestions.extend([f"🔧 {opp}" for opp in optimization_opportunities])
        
        # Resource limit suggestions
        resource_limits = docker_analysis.get('resource_limits', {})
        services_without_limits = [
            service for service, limits in resource_limits.items()
            if not limits.get('memory_limit') or not limits.get('cpu_limit')
        ]
        
        if services_without_limits:
            suggestions.append(
                f"⚠️ Services without resource limits: {', '.join(services_without_limits)}"
            )
        
        return suggestions
    
    def suggest_deployment_improvements(self) -> List[str]:
        """Suggest deployment enhancements"""
        return [
            "🔍 Add comprehensive health checks to all services",
            "🔄 Implement graceful shutdown handlers",
            "📊 Add resource limits to containers to prevent OOM kills",
            "🔁 Enable auto-restart on failures with backoff strategy",
            "📝 Add structured logging for better debugging",
            "⏱️ Configure proper timeout values for all operations",
            "🔐 Add security headers and rate limiting",
            "📈 Implement metrics collection and monitoring",
            "🚨 Set up alerting for critical failures",
            "🔄 Add rolling deployment strategy for zero-downtime updates",
            "💾 Implement automated backups with retention policies",
            "🔒 Enable TLS/SSL for all external communications",
            "🌍 Configure CDN for static assets",
            "📊 Add performance monitoring and profiling",
            "🧪 Implement blue-green deployment for safer releases"
        ]
    
    def suggest_security_improvements(self) -> List[str]:
        """Suggest security enhancements"""
        return [
            "🔐 Use secrets management instead of environment variables for sensitive data",
            "🛡️ Enable container scanning for vulnerabilities",
            "🔒 Implement proper authentication and authorization",
            "🌐 Use HTTPS/TLS for all communications",
            "🚫 Implement rate limiting to prevent abuse",
            "📝 Enable security headers (HSTS, CSP, etc.)",
            "🔍 Add security monitoring and logging",
            "🔄 Regular security updates for base images",
            "🚪 Implement proper session management",
            "📊 Add intrusion detection and prevention",
            "🔐 Use strong encryption for data at rest",
            "🛡️ Implement input validation and sanitization",
            "🔒 Enable firewall rules for network security",
            "📋 Regular security audits and penetration testing",
            "🔑 Implement proper key rotation policies"
        ]
    
    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        print("\n📊 Generating Resource Optimization Report...")
        
        # Perform analysis
        self.analyze_current_setup()
        
        # Generate suggestions
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_analysis': self.analysis_results,
            'optimization_suggestions': {
                'memory_optimizations': self.suggest_memory_optimizations(),
                'cpu_optimizations': self.suggest_cpu_optimizations(),
                'docker_optimizations': self.suggest_docker_optimizations(),
                'deployment_improvements': self.suggest_deployment_improvements(),
                'security_improvements': self.suggest_security_improvements()
            },
            'priority_actions': self._get_priority_actions(),
            'implementation_guide': self._get_implementation_guide()
        }
        
        # Save report
        report_path = self.workspace / 'resource_optimization_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Report saved to: {report_path}")
        
        # Display summary
        self._display_optimization_summary(report)
        
        return str(report_path)
    
    def _get_priority_actions(self) -> List[str]:
        """Get priority actions based on current system state"""
        priority_actions = []
        
        available_memory = self.current_config.get('memory', {}).get('available_gb', 0)
        used_percent = self.current_config.get('memory', {}).get('percent_used', 0)
        
        # Critical memory issues
        if available_memory < 2:
            priority_actions.extend([
                "🚨 CRITICAL: Add more RAM or reduce memory usage immediately",
                "⚡ Reduce Docker container memory limits",
                "🔄 Enable swap if not already enabled"
            ])
        
        # Docker not available
        try:
            subprocess.run(['docker', '--version'], capture_output=True, check=True)
        except:
            priority_actions.append("🐳 PRIORITY: Install Docker for containerized deployment")
        
        # Missing resource limits
        docker_analysis = self.analysis_results.get('docker_analysis', {})
        if docker_analysis.get('optimization_opportunities'):
            priority_actions.append("📊 HIGH: Add resource limits to Docker services")
        
        # No health checks
        compose_files = docker_analysis.get('compose_files', [])
        if compose_files and not any('healthcheck' in str(opp) for opp in docker_analysis.get('optimization_opportunities', [])):
            priority_actions.append("🔍 MEDIUM: Add health checks to all services")
        
        return priority_actions
    
    def _get_implementation_guide(self) -> Dict[str, Any]:
        """Get implementation guide for optimizations"""
        return {
            'immediate_actions': [
                "1. Review and implement priority actions first",
                "2. Test changes in development environment",
                "3. Monitor system resources after changes",
                "4. Document all modifications made"
            ],
            'docker_optimizations': [
                "1. Add resource limits to docker-compose.yml services",
                "2. Implement health checks for all services",
                "3. Use multi-stage builds in Dockerfiles",
                "4. Add proper logging configuration"
            ],
            'ai_optimizations': [
                "1. Enable model quantization if memory is limited",
                "2. Implement model caching for better performance",
                "3. Optimize batch sizes based on available memory",
                "4. Consider GPU acceleration if available"
            ],
            'monitoring_setup': [
                "1. Set up resource monitoring (CPU, memory, disk)",
                "2. Implement application-level metrics",
                "3. Configure alerting for critical issues",
                "4. Regular health checks and reporting"
            ]
        }
    
    def _display_optimization_summary(self, report: Dict[str, Any]):
        """Display optimization summary"""
        print("\n📋 OPTIMIZATION SUMMARY:")
        print("=" * 60)
        
        # System info
        system_info = report['system_analysis']['system_info']
        print(f"💾 Memory: {system_info['memory']['available_gb']}GB available")
        print(f"🖥️ CPU: {system_info['cpu']['logical_cores']} cores")
        print(f"💽 Disk: {system_info['disk']['free_gb']}GB free")
        
        # Priority actions
        priority_actions = report['priority_actions']
        if priority_actions:
            print("\n🚨 PRIORITY ACTIONS:")
            for action in priority_actions[:3]:
                print(f"  {action}")
        
        # Top suggestions by category
        suggestions = report['optimization_suggestions']
        
        print("\n💾 Top Memory Optimizations:")
        for suggestion in suggestions['memory_optimizations'][:3]:
            print(f"  {suggestion}")
        
        print("\n🐳 Top Docker Optimizations:")
        for suggestion in suggestions['docker_optimizations'][:3]:
            print(f"  {suggestion}")
        
        print("\n🚀 Top Deployment Improvements:")
        for suggestion in suggestions['deployment_improvements'][:3]:
            print(f"  {suggestion}")
        
        print("\n" + "=" * 60)
        print("📊 Full report saved with detailed recommendations")
        print("🛡️ SAFETY: All suggestions are non-destructive and preserve existing functionality")

def main():
    """Main function"""
    assistant = ResourceOptimizationAssistant()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Persian Legal AI Resource Optimization Assistant')
    parser.add_argument('--analyze', action='store_true', help='Analyze current setup only')
    parser.add_argument('--report', action='store_true', help='Generate full optimization report')
    parser.add_argument('--memory', action='store_true', help='Show memory optimizations only')
    parser.add_argument('--docker', action='store_true', help='Show Docker optimizations only')
    
    args = parser.parse_args()
    
    if args.analyze:
        assistant.analyze_current_setup()
    elif args.memory:
        assistant.analyze_current_setup()
        suggestions = assistant.suggest_memory_optimizations()
        print("\n💾 Memory Optimization Suggestions:")
        for suggestion in suggestions:
            print(f"  {suggestion}")
    elif args.docker:
        assistant.analyze_current_setup()
        suggestions = assistant.suggest_docker_optimizations()
        print("\n🐳 Docker Optimization Suggestions:")
        for suggestion in suggestions:
            print(f"  {suggestion}")
    else:
        # Generate full report
        assistant.generate_optimization_report()

if __name__ == '__main__':
    main()