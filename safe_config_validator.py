#!/usr/bin/env python3
"""
Safe Configuration Validator for Persian Legal AI
Validates configurations without changing them

🛡️ SAFETY: This validator only reads and reports - no modifications
"""

import os
import json
import yaml
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import sqlite3

class SafeConfigValidator:
    """Validates configurations without changing them"""
    
    def __init__(self):
        self.workspace = Path("/workspace")
        self.validation_results = {}
        self.critical_issues = []
        self.warnings = []
        self.recommendations = []
        
    def validate_all_configs(self) -> Dict[str, Any]:
        """Check all configurations for issues"""
        print("🔍 Validating All Deployment Configurations...")
        print("🛡️ SAFETY: Read-only validation - no changes will be made")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'docker_compose': self._validate_docker_compose(),
            'dockerfiles': self._validate_dockerfiles(),
            'environment': self._validate_environment(),
            'dependencies': self._validate_dependencies(),
            'database': self._validate_database(),
            'network': self._validate_network_config(),
            'security': self._validate_security_config(),
            'ai_models': self._validate_ai_models(),
            'frontend': self._validate_frontend_config(),
            'nginx': self._validate_nginx_config()
        }
        
        self.validation_results = results
        return results
    
    def _validate_docker_compose(self) -> Dict[str, Any]:
        """Validate docker-compose.yml files"""
        print("\n📦 Validating Docker Compose Configuration...")
        
        compose_files = [
            'docker-compose.yml',
            'docker-compose.production.yml', 
            'docker-compose.enhanced.yml'
        ]
        
        results = {}
        
        for compose_file in compose_files:
            compose_path = self.workspace / compose_file
            
            if not compose_path.exists():
                results[compose_file] = {
                    'exists': False,
                    'required': compose_file == 'docker-compose.yml'
                }
                continue
            
            try:
                # Parse YAML
                with open(compose_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                file_result = {
                    'exists': True,
                    'valid_yaml': True,
                    'services': {},
                    'networks': {},
                    'volumes': {},
                    'issues': [],
                    'warnings': [],
                    'recommendations': []
                }
                
                # Validate structure
                if 'services' not in config:
                    file_result['issues'].append('No services defined')
                    results[compose_file] = file_result
                    continue
                
                # Check services
                services = config.get('services', {})
                required_services = ['backend', 'frontend', 'redis']
                
                for service_name, service_config in services.items():
                    service_validation = self._validate_service_config(service_name, service_config)
                    file_result['services'][service_name] = service_validation
                
                # Check for missing required services (for main compose file)
                if compose_file == 'docker-compose.yml':
                    missing_services = [s for s in required_services if s not in services]
                    if missing_services:
                        file_result['issues'].append(f'Missing required services: {missing_services}')
                
                # Validate networks
                networks = config.get('networks', {})
                for network_name, network_config in networks.items():
                    file_result['networks'][network_name] = self._validate_network(network_name, network_config)
                
                # Validate volumes
                volumes = config.get('volumes', {})
                for volume_name, volume_config in volumes.items():
                    file_result['volumes'][volume_name] = self._validate_volume(volume_name, volume_config)
                
                # Check version
                version = config.get('version')
                if not version:
                    file_result['warnings'].append('No version specified')
                elif version < '3.0':
                    file_result['warnings'].append(f'Old Docker Compose version: {version}')
                
                results[compose_file] = file_result
                print(f"  ✅ {compose_file}: Valid YAML structure")
                
            except yaml.YAMLError as e:
                results[compose_file] = {
                    'exists': True,
                    'valid_yaml': False,
                    'error': f'YAML syntax error: {e}',
                    'issues': [f'YAML syntax error: {e}']
                }
                print(f"  ❌ {compose_file}: YAML syntax error")
                
            except Exception as e:
                results[compose_file] = {
                    'exists': True,
                    'valid_yaml': False,
                    'error': f'Validation error: {e}',
                    'issues': [f'Validation error: {e}']
                }
                print(f"  ❌ {compose_file}: Validation error")
        
        return results
    
    def _validate_service_config(self, service_name: str, service_config: Dict) -> Dict[str, Any]:
        """Validate individual service configuration"""
        validation = {
            'has_build_or_image': False,
            'has_ports': False,
            'has_healthcheck': False,
            'has_resource_limits': False,
            'has_restart_policy': False,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check for build or image
        if 'build' in service_config or 'image' in service_config:
            validation['has_build_or_image'] = True
        else:
            validation['issues'].append('No build or image specified')
        
        # Check for ports
        if 'ports' in service_config:
            validation['has_ports'] = True
            # Validate port format
            for port in service_config['ports']:
                if isinstance(port, str) and ':' not in port:
                    validation['warnings'].append(f'Port {port} may need host:container mapping')
        
        # Check for health checks
        if 'healthcheck' in service_config:
            validation['has_healthcheck'] = True
            healthcheck = service_config['healthcheck']
            if 'test' not in healthcheck:
                validation['warnings'].append('Health check missing test command')
        else:
            validation['recommendations'].append('Add health check for better reliability')
        
        # Check for resource limits
        deploy_config = service_config.get('deploy', {})
        resources = deploy_config.get('resources', {})
        if resources.get('limits'):
            validation['has_resource_limits'] = True
        else:
            validation['recommendations'].append('Add resource limits to prevent resource exhaustion')
        
        # Check for restart policy
        if 'restart' in service_config:
            validation['has_restart_policy'] = True
            restart_policy = service_config['restart']
            if restart_policy not in ['no', 'always', 'on-failure', 'unless-stopped']:
                validation['warnings'].append(f'Unknown restart policy: {restart_policy}')
        else:
            validation['recommendations'].append('Add restart policy for production reliability')
        
        # Check environment variables
        env_vars = service_config.get('environment', [])
        if env_vars:
            for env_var in env_vars:
                if isinstance(env_var, str):
                    if 'password' in env_var.lower() and '=' in env_var:
                        validation['warnings'].append('Potential hardcoded password in environment')
        
        return validation
    
    def _validate_network(self, network_name: str, network_config: Dict) -> Dict[str, Any]:
        """Validate network configuration"""
        validation = {
            'driver': network_config.get('driver', 'bridge'),
            'has_ipam': 'ipam' in network_config,
            'issues': [],
            'warnings': []
        }
        
        # Check IPAM configuration
        if 'ipam' in network_config:
            ipam = network_config['ipam']
            if 'config' in ipam:
                for config in ipam['config']:
                    subnet = config.get('subnet')
                    if subnet:
                        # Basic subnet validation
                        if not re.match(r'^\d+\.\d+\.\d+\.\d+/\d+$', subnet):
                            validation['warnings'].append(f'Invalid subnet format: {subnet}')
        
        return validation
    
    def _validate_volume(self, volume_name: str, volume_config: Dict) -> Dict[str, Any]:
        """Validate volume configuration"""
        validation = {
            'driver': volume_config.get('driver', 'local') if volume_config else 'local',
            'external': volume_config.get('external', False) if volume_config else False,
            'issues': [],
            'warnings': []
        }
        
        return validation
    
    def _validate_dockerfiles(self) -> Dict[str, Any]:
        """Validate Dockerfiles"""
        print("\n🐳 Validating Dockerfiles...")
        
        dockerfiles = [
            'backend/Dockerfile',
            'persian-legal-ai-frontend/Dockerfile',
            'Dockerfile.backend',
            'Dockerfile.frontend',
            'Dockerfile.production'
        ]
        
        results = {}
        
        for dockerfile_path in dockerfiles:
            dockerfile = self.workspace / dockerfile_path
            
            if not dockerfile.exists():
                results[dockerfile_path] = {
                    'exists': False,
                    'required': dockerfile_path in ['backend/Dockerfile', 'persian-legal-ai-frontend/Dockerfile']
                }
                continue
            
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                
                validation = self._analyze_dockerfile_content(content)
                validation['exists'] = True
                results[dockerfile_path] = validation
                
                if validation['issues']:
                    print(f"  ❌ {dockerfile_path}: {len(validation['issues'])} issues found")
                else:
                    print(f"  ✅ {dockerfile_path}: No critical issues")
                
            except Exception as e:
                results[dockerfile_path] = {
                    'exists': True,
                    'error': str(e),
                    'issues': [f'Read error: {e}']
                }
                print(f"  ❌ {dockerfile_path}: Read error")
        
        return results
    
    def _analyze_dockerfile_content(self, content: str) -> Dict[str, Any]:
        """Analyze Dockerfile content for best practices"""
        lines = content.strip().split('\n')
        
        validation = {
            'has_from': False,
            'has_user': False,
            'has_healthcheck': False,
            'has_expose': False,
            'has_workdir': False,
            'layer_count': 0,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            instruction = line.split()[0].upper()
            
            if instruction == 'FROM':
                validation['has_from'] = True
                # Check for specific base image recommendations
                if 'alpine' in line.lower():
                    validation['recommendations'].append('Good choice: Using Alpine for smaller image size')
                elif 'slim' in line.lower():
                    validation['recommendations'].append('Good choice: Using slim image')
            
            elif instruction == 'USER':
                validation['has_user'] = True
                if 'root' in line:
                    validation['warnings'].append('Running as root user - consider security implications')
            
            elif instruction == 'HEALTHCHECK':
                validation['has_healthcheck'] = True
            
            elif instruction == 'EXPOSE':
                validation['has_expose'] = True
            
            elif instruction == 'WORKDIR':
                validation['has_workdir'] = True
            
            elif instruction in ['RUN', 'COPY', 'ADD']:
                validation['layer_count'] += 1
        
        # Validation checks
        if not validation['has_from']:
            validation['issues'].append('Missing FROM instruction')
        
        if not validation['has_user']:
            validation['recommendations'].append('Consider adding USER instruction for security')
        
        if not validation['has_healthcheck']:
            validation['recommendations'].append('Consider adding HEALTHCHECK instruction')
        
        if not validation['has_workdir']:
            validation['recommendations'].append('Consider setting WORKDIR for better organization')
        
        if validation['layer_count'] > 10:
            validation['warnings'].append(f'High layer count ({validation["layer_count"]}) - consider combining RUN commands')
        
        # Check for common issues
        if 'apt-get update' in content and 'apt-get clean' not in content:
            validation['recommendations'].append('Clean apt cache after installation to reduce image size')
        
        if 'pip install' in content and '--no-cache-dir' not in content:
            validation['recommendations'].append('Use --no-cache-dir with pip to reduce image size')
        
        return validation
    
    def _validate_environment(self) -> Dict[str, Any]:
        """Validate environment configuration"""
        print("\n🌍 Validating Environment Configuration...")
        
        env_files = ['.env', '.env.local', '.env.production', '.env.example', '.env.production.example']
        
        results = {
            'files_found': [],
            'template_available': False,
            'security_issues': [],
            'missing_variables': [],
            'recommendations': []
        }
        
        for env_file in env_files:
            env_path = self.workspace / env_file
            if env_path.exists():
                results['files_found'].append(env_file)
                
                if 'example' in env_file:
                    results['template_available'] = True
                
                try:
                    with open(env_path, 'r') as f:
                        content = f.read()
                    
                    # Check for security issues
                    lines = content.split('\n')
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        if '=' in line:
                            key, value = line.split('=', 1)
                            
                            # Check for potential secrets
                            if any(secret in key.lower() for secret in ['password', 'secret', 'key', 'token']):
                                if value and not value.startswith('${') and 'example' not in env_file:
                                    results['security_issues'].append(f'{env_file}:{line_num} - Potential hardcoded secret: {key}')
                
                except Exception as e:
                    results['security_issues'].append(f'Error reading {env_file}: {e}')
        
        # Check for required environment variables in docker-compose
        compose_path = self.workspace / 'docker-compose.yml'
        if compose_path.exists():
            try:
                with open(compose_path, 'r') as f:
                    compose_content = f.read()
                
                # Find environment variable references
                env_refs = re.findall(r'\$\{([^}]+)\}', compose_content)
                required_vars = set()
                for ref in env_refs:
                    var_name = ref.split(':')[0]  # Remove default values
                    required_vars.add(var_name)
                
                # Check if we have env files for these variables
                if required_vars and not results['files_found']:
                    results['missing_variables'] = list(required_vars)
                    results['recommendations'].append('Create .env file for environment variables')
                
            except Exception as e:
                results['security_issues'].append(f'Error analyzing docker-compose.yml: {e}')
        
        if not results['template_available']:
            results['recommendations'].append('Create .env.production.example template file')
        
        if results['files_found']:
            print(f"  ✅ Found environment files: {', '.join(results['files_found'])}")
        else:
            print("  ⚠️ No environment files found")
        
        return results
    
    def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate project dependencies"""
        print("\n📚 Validating Dependencies...")
        
        results = {
            'backend': {},
            'frontend': {},
            'issues': [],
            'recommendations': []
        }
        
        # Backend dependencies
        backend_req = self.workspace / 'backend' / 'requirements.txt'
        if backend_req.exists():
            try:
                with open(backend_req, 'r') as f:
                    content = f.read()
                
                results['backend'] = self._analyze_python_requirements(content)
                print("  ✅ Backend requirements.txt found and analyzed")
                
            except Exception as e:
                results['backend'] = {'error': str(e)}
                results['issues'].append(f'Error reading backend requirements.txt: {e}')
        else:
            results['issues'].append('Backend requirements.txt not found')
        
        # Frontend dependencies
        frontend_dirs = ['frontend', 'persian-legal-ai-frontend']
        frontend_found = False
        
        for frontend_dir in frontend_dirs:
            package_json = self.workspace / frontend_dir / 'package.json'
            if package_json.exists():
                try:
                    with open(package_json, 'r') as f:
                        config = json.load(f)
                    
                    results['frontend'] = self._analyze_package_json(config)
                    frontend_found = True
                    print(f"  ✅ {frontend_dir}/package.json found and analyzed")
                    break
                    
                except Exception as e:
                    results['frontend'] = {'error': str(e)}
                    results['issues'].append(f'Error reading {frontend_dir}/package.json: {e}')
        
        if not frontend_found:
            results['issues'].append('Frontend package.json not found')
        
        return results
    
    def _analyze_python_requirements(self, content: str) -> Dict[str, Any]:
        """Analyze Python requirements.txt"""
        analysis = {
            'packages': [],
            'critical_packages': [],
            'version_pinning': {'pinned': 0, 'unpinned': 0},
            'security_issues': [],
            'recommendations': []
        }
        
        lines = content.strip().split('\n')
        critical_deps = ['fastapi', 'uvicorn', 'redis', 'torch', 'transformers']
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse package name and version
            if '==' in line:
                package = line.split('==')[0].strip()
                analysis['version_pinning']['pinned'] += 1
            elif '>=' in line:
                package = line.split('>=')[0].strip()
                analysis['version_pinning']['unpinned'] += 1
            else:
                package = line.strip()
                analysis['version_pinning']['unpinned'] += 1
            
            analysis['packages'].append(package)
            
            if package.lower() in critical_deps:
                analysis['critical_packages'].append(package)
        
        # Check for missing critical dependencies
        missing_critical = [dep for dep in critical_deps if dep not in [p.lower() for p in analysis['packages']]]
        if missing_critical:
            analysis['security_issues'].extend([f'Missing critical dependency: {dep}' for dep in missing_critical])
        
        # Version pinning recommendations
        total_packages = analysis['version_pinning']['pinned'] + analysis['version_pinning']['unpinned']
        if total_packages > 0:
            pinned_percentage = (analysis['version_pinning']['pinned'] / total_packages) * 100
            if pinned_percentage < 80:
                analysis['recommendations'].append('Consider pinning more package versions for reproducible builds')
        
        return analysis
    
    def _analyze_package_json(self, config: Dict) -> Dict[str, Any]:
        """Analyze package.json"""
        analysis = {
            'name': config.get('name'),
            'version': config.get('version'),
            'scripts': list(config.get('scripts', {}).keys()),
            'dependencies_count': len(config.get('dependencies', {})),
            'dev_dependencies_count': len(config.get('devDependencies', {})),
            'issues': [],
            'recommendations': []
        }
        
        # Check for required scripts
        required_scripts = ['build', 'start']
        missing_scripts = [script for script in required_scripts if script not in analysis['scripts']]
        if missing_scripts:
            analysis['issues'].extend([f'Missing script: {script}' for script in missing_scripts])
        
        # Check for security vulnerabilities (basic check)
        dependencies = config.get('dependencies', {})
        for dep, version in dependencies.items():
            if version == '*' or version == 'latest':
                analysis['recommendations'].append(f'Pin version for {dep} instead of using {version}')
        
        return analysis
    
    def _validate_database(self) -> Dict[str, Any]:
        """Validate database configuration"""
        print("\n🗄️ Validating Database Configuration...")
        
        results = {
            'databases_found': [],
            'integrity_checks': {},
            'size_analysis': {},
            'recommendations': []
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
                results['databases_found'].append(db_file)
                
                # File size analysis
                size_mb = db_path.stat().st_size / (1024 * 1024)
                results['size_analysis'][db_file] = f"{size_mb:.2f} MB"
                
                # SQLite integrity check
                if db_file.endswith('.db'):
                    try:
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        
                        # Basic integrity check
                        cursor.execute("PRAGMA integrity_check;")
                        integrity_result = cursor.fetchone()[0]
                        
                        # Get table count
                        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
                        table_count = cursor.fetchone()[0]
                        
                        results['integrity_checks'][db_file] = {
                            'integrity': integrity_result,
                            'table_count': table_count
                        }
                        
                        conn.close()
                        
                        if integrity_result == 'ok':
                            print(f"  ✅ {db_file}: Integrity OK, {table_count} tables")
                        else:
                            print(f"  ❌ {db_file}: Integrity issues detected")
                        
                        # Size-based recommendations
                        if size_mb > 100:
                            results['recommendations'].append(f'{db_file}: Large database ({size_mb:.1f}MB) - consider VACUUM to reclaim space')
                        
                    except Exception as e:
                        results['integrity_checks'][db_file] = {'error': str(e)}
                        print(f"  ⚠️ {db_file}: Could not check integrity - {e}")
        
        if not results['databases_found']:
            results['recommendations'].append('No database files found - database will be created on first run')
            print("  ℹ️ No database files found")
        
        return results
    
    def _validate_network_config(self) -> Dict[str, Any]:
        """Validate network configuration"""
        print("\n🌐 Validating Network Configuration...")
        
        results = {
            'port_mappings': {},
            'port_conflicts': [],
            'network_isolation': {},
            'recommendations': []
        }
        
        # Analyze docker-compose network configuration
        compose_files = ['docker-compose.yml', 'docker-compose.enhanced.yml']
        
        used_ports = {}
        
        for compose_file in compose_files:
            compose_path = self.workspace / compose_file
            if compose_path.exists():
                try:
                    with open(compose_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    services = config.get('services', {})
                    
                    # Analyze port mappings
                    for service_name, service_config in services.items():
                        ports = service_config.get('ports', [])
                        for port in ports:
                            if isinstance(port, str) and ':' in port:
                                external_port = port.split(':')[0].replace('"', '')
                                
                                if external_port in used_ports:
                                    results['port_conflicts'].append({
                                        'port': external_port,
                                        'services': [used_ports[external_port], f"{compose_file}:{service_name}"]
                                    })
                                else:
                                    used_ports[external_port] = f"{compose_file}:{service_name}"
                                
                                results['port_mappings'][f"{compose_file}:{service_name}"] = external_port
                    
                    # Analyze network configuration
                    networks = config.get('networks', {})
                    for network_name, network_config in networks.items():
                        results['network_isolation'][f"{compose_file}:{network_name}"] = {
                            'driver': network_config.get('driver', 'bridge'),
                            'has_ipam': 'ipam' in network_config
                        }
                
                except Exception as e:
                    print(f"  ⚠️ Error analyzing {compose_file}: {e}")
        
        # Port conflict analysis
        if results['port_conflicts']:
            print(f"  ⚠️ Found {len(results['port_conflicts'])} port conflicts")
            for conflict in results['port_conflicts']:
                print(f"    Port {conflict['port']}: {', '.join(conflict['services'])}")
        else:
            print("  ✅ No port conflicts detected")
        
        # Recommendations
        if len(results['port_mappings']) > 5:
            results['recommendations'].append('Multiple services exposed - consider using a reverse proxy for better security')
        
        results['recommendations'].extend([
            'Use internal networks for service-to-service communication',
            'Enable connection pooling for database connections',
            'Configure proper CORS settings for production'
        ])
        
        return results
    
    def _validate_security_config(self) -> Dict[str, Any]:
        """Validate security configuration"""
        print("\n🔒 Validating Security Configuration...")
        
        results = {
            'secrets_management': {},
            'file_permissions': {},
            'container_security': {},
            'issues': [],
            'recommendations': []
        }
        
        # Check for secrets in configuration files
        config_files = ['docker-compose.yml', '.env', '.env.production']
        
        for config_file in config_files:
            config_path = self.workspace / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        content = f.read()
                    
                    # Check for hardcoded secrets
                    secret_patterns = [
                        (r'password\s*[=:]\s*[\'"][^\'"\s]+[\'"]', 'Potential hardcoded password'),
                        (r'secret\s*[=:]\s*[\'"][^\'"\s]+[\'"]', 'Potential hardcoded secret'),
                        (r'key\s*[=:]\s*[\'"][^\'"\s]+[\'"]', 'Potential hardcoded key'),
                        (r'token\s*[=:]\s*[\'"][^\'"\s]+[\'"]', 'Potential hardcoded token')
                    ]
                    
                    for pattern, description in secret_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            results['issues'].append(f'{config_file}: {description}')
                    
                    # Check for environment variable usage
                    if '${' in content:
                        results['secrets_management'][config_file] = 'Uses environment variables'
                    else:
                        results['secrets_management'][config_file] = 'No environment variables detected'
                
                except Exception as e:
                    results['issues'].append(f'Error reading {config_file}: {e}')
        
        # Check .gitignore for sensitive files
        gitignore_path = self.workspace / '.gitignore'
        if gitignore_path.exists():
            try:
                with open(gitignore_path, 'r') as f:
                    gitignore_content = f.read()
                
                sensitive_patterns = ['.env', '*.key', '*.pem', 'secrets/']
                missing_patterns = []
                
                for pattern in sensitive_patterns:
                    if pattern not in gitignore_content:
                        missing_patterns.append(pattern)
                
                if missing_patterns:
                    results['recommendations'].append(f'Add to .gitignore: {", ".join(missing_patterns)}')
                else:
                    print("  ✅ .gitignore properly configured for sensitive files")
                
            except Exception as e:
                results['issues'].append(f'Error reading .gitignore: {e}')
        else:
            results['recommendations'].append('Create .gitignore file to exclude sensitive files')
        
        # Container security recommendations
        results['recommendations'].extend([
            'Use secrets management instead of environment variables for sensitive data',
            'Run containers as non-root users',
            'Enable container scanning for vulnerabilities',
            'Implement proper authentication and authorization',
            'Use HTTPS/TLS for all communications',
            'Enable security headers (HSTS, CSP, etc.)',
            'Regular security updates for base images'
        ])
        
        return results
    
    def _validate_ai_models(self) -> Dict[str, Any]:
        """Validate AI model configuration"""
        print("\n🤖 Validating AI Models Configuration...")
        
        results = {
            'model_directories': [],
            'model_files': {},
            'cache_configuration': {},
            'recommendations': []
        }
        
        # Check for model directories
        model_dirs = ['models', 'ai_models', 'backend/models']
        
        for model_dir in model_dirs:
            model_path = self.workspace / model_dir
            if model_path.exists():
                results['model_directories'].append(model_dir)
                
                # Count model files
                model_files = list(model_path.glob('**/*'))
                file_count = len([f for f in model_files if f.is_file()])
                total_size = sum(f.stat().st_size for f in model_files if f.is_file())
                
                results['model_files'][model_dir] = {
                    'file_count': file_count,
                    'total_size_mb': round(total_size / (1024 * 1024), 2)
                }
                
                print(f"  ✅ {model_dir}: {file_count} files, {results['model_files'][model_dir]['total_size_mb']:.1f}MB")
        
        # Check for model caching configuration
        env_files = ['.env', '.env.production', '.env.production.example']
        for env_file in env_files:
            env_path = self.workspace / env_file
            if env_path.exists():
                try:
                    with open(env_path, 'r') as f:
                        content = f.read()
                    
                    cache_vars = ['HUGGINGFACE_CACHE_DIR', 'TRANSFORMERS_CACHE', 'HF_HOME']
                    for var in cache_vars:
                        if var in content:
                            results['cache_configuration'][env_file] = f'Configures {var}'
                            break
                
                except Exception:
                    pass
        
        # Recommendations based on findings
        if not results['model_directories']:
            results['recommendations'].append('No model directories found - models will be downloaded on first use')
        else:
            total_size = sum(info['total_size_mb'] for info in results['model_files'].values())
            if total_size > 1000:  # > 1GB
                results['recommendations'].extend([
                    f'Large model storage ({total_size:.1f}MB) - ensure sufficient disk space',
                    'Consider model quantization to reduce storage requirements',
                    'Implement model pruning for production deployment'
                ])
        
        if not results['cache_configuration']:
            results['recommendations'].append('Configure model caching (HUGGINGFACE_CACHE_DIR) for better performance')
        
        results['recommendations'].extend([
            'Use model versioning for reproducible deployments',
            'Implement model health checks',
            'Monitor model memory usage',
            'Consider GPU acceleration if available'
        ])
        
        return results
    
    def _validate_frontend_config(self) -> Dict[str, Any]:
        """Validate frontend configuration"""
        print("\n🎨 Validating Frontend Configuration...")
        
        results = {
            'frontend_found': False,
            'build_config': {},
            'api_configuration': {},
            'recommendations': []
        }
        
        # Check frontend directories
        frontend_dirs = ['frontend', 'persian-legal-ai-frontend']
        
        for frontend_dir in frontend_dirs:
            frontend_path = self.workspace / frontend_dir
            if frontend_path.exists():
                results['frontend_found'] = True
                
                # Check package.json
                package_json = frontend_path / 'package.json'
                if package_json.exists():
                    try:
                        with open(package_json, 'r') as f:
                            config = json.load(f)
                        
                        results['build_config'] = {
                            'has_build_script': 'build' in config.get('scripts', {}),
                            'has_start_script': 'start' in config.get('scripts', {}),
                            'dependencies_count': len(config.get('dependencies', {}))
                        }
                        
                    except Exception as e:
                        results['build_config'] = {'error': str(e)}
                
                # Check for Next.js configuration
                next_config = frontend_path / 'next.config.js'
                if next_config.exists():
                    results['build_config']['has_next_config'] = True
                
                # Check for environment configuration
                env_files = ['.env.local', '.env.production']
                for env_file in env_files:
                    env_path = frontend_path / env_file
                    if env_path.exists():
                        try:
                            with open(env_path, 'r') as f:
                                content = f.read()
                            
                            if 'NEXT_PUBLIC_API_URL' in content:
                                results['api_configuration']['has_api_url'] = True
                        except Exception:
                            pass
                
                print(f"  ✅ Frontend found in {frontend_dir}")
                break
        
        if not results['frontend_found']:
            print("  ❌ No frontend directory found")
            results['recommendations'].append('Frontend directory not found')
            return results
        
        # Recommendations
        if not results['build_config'].get('has_build_script'):
            results['recommendations'].append('Add build script to package.json')
        
        if not results['api_configuration'].get('has_api_url'):
            results['recommendations'].append('Configure NEXT_PUBLIC_API_URL for API communication')
        
        results['recommendations'].extend([
            'Optimize bundle size for production',
            'Enable static file compression',
            'Configure CDN for static assets',
            'Implement proper error boundaries',
            'Add performance monitoring'
        ])
        
        return results
    
    def _validate_nginx_config(self) -> Dict[str, Any]:
        """Validate nginx configuration if present"""
        print("\n🌐 Validating Nginx Configuration...")
        
        results = {
            'config_files': [],
            'configuration_valid': True,
            'recommendations': []
        }
        
        # Check for nginx configuration files
        nginx_configs = [
            'nginx.conf',
            'nginx/nginx.conf',
            'nginx/enhanced.conf'
        ]
        
        for config_file in nginx_configs:
            config_path = self.workspace / config_file
            if config_path.exists():
                results['config_files'].append(config_file)
                print(f"  ✅ Found nginx config: {config_file}")
        
        if not results['config_files']:
            print("  ℹ️ No nginx configuration found")
            results['recommendations'].append('Consider adding nginx for reverse proxy and static file serving')
        else:
            results['recommendations'].extend([
                'Enable gzip compression',
                'Configure proper cache headers',
                'Add security headers',
                'Enable rate limiting',
                'Configure SSL/TLS if using HTTPS'
            ])
        
        return results
    
    def generate_fix_recommendations(self) -> List[Dict[str, Any]]:
        """Generate fix recommendations without implementing them"""
        recommendations = []
        
        # Analyze validation results
        if not self.validation_results:
            self.validate_all_configs()
        
        # Docker Compose issues
        docker_compose = self.validation_results.get('docker_compose', {})
        for file_name, file_result in docker_compose.items():
            if isinstance(file_result, dict) and file_result.get('issues'):
                for issue in file_result['issues']:
                    recommendations.append({
                        'category': 'Docker Compose',
                        'file': file_name,
                        'issue': issue,
                        'fix': self._get_docker_compose_fix(issue),
                        'impact': 'Medium',
                        'safe': True
                    })
        
        # Environment issues
        environment = self.validation_results.get('environment', {})
        if environment.get('missing_variables'):
            recommendations.append({
                'category': 'Environment',
                'issue': 'Missing environment variables',
                'fix': 'Create .env.production file with required variables',
                'impact': 'High',
                'safe': True
            })
        
        # Security issues
        security = self.validation_results.get('security', {})
        for issue in security.get('issues', []):
            recommendations.append({
                'category': 'Security',
                'issue': issue,
                'fix': 'Use environment variables or secrets management',
                'impact': 'High',
                'safe': True
            })
        
        # Dependencies issues
        dependencies = self.validation_results.get('dependencies', {})
        for issue in dependencies.get('issues', []):
            recommendations.append({
                'category': 'Dependencies',
                'issue': issue,
                'fix': self._get_dependency_fix(issue),
                'impact': 'High',
                'safe': True
            })
        
        return recommendations
    
    def _get_docker_compose_fix(self, issue: str) -> str:
        """Get fix recommendation for Docker Compose issue"""
        if 'missing services' in issue.lower():
            return 'Add missing services to docker-compose.yml'
        elif 'health check' in issue.lower():
            return 'Add healthcheck configuration to service'
        elif 'resource limits' in issue.lower():
            return 'Add deploy.resources.limits to service configuration'
        else:
            return 'Review and fix Docker Compose configuration'
    
    def _get_dependency_fix(self, issue: str) -> str:
        """Get fix recommendation for dependency issue"""
        if 'requirements.txt' in issue:
            return 'Create backend/requirements.txt with required Python packages'
        elif 'package.json' in issue:
            return 'Create package.json in frontend directory with required dependencies'
        else:
            return 'Install missing dependencies'
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        print("\n📊 Generating Configuration Validation Report...")
        
        # Perform validation if not already done
        if not self.validation_results:
            self.validate_all_configs()
        
        # Generate recommendations
        recommendations = self.generate_fix_recommendations()
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': self.validation_results,
            'recommendations': recommendations,
            'summary': {
                'total_issues': sum(
                    len(result.get('issues', [])) if isinstance(result, dict) else 0
                    for result in self.validation_results.values()
                ),
                'total_warnings': sum(
                    len(result.get('warnings', [])) if isinstance(result, dict) else 0
                    for result in self.validation_results.values()
                ),
                'categories_checked': len(self.validation_results),
                'recommendations_count': len(recommendations)
            }
        }
        
        # Save report
        report_path = self.workspace / 'config_validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Validation report saved to: {report_path}")
        
        # Display summary
        self._display_validation_summary(report)
        
        return str(report_path)
    
    def _display_validation_summary(self, report: Dict[str, Any]):
        """Display validation summary"""
        print("\n📋 CONFIGURATION VALIDATION SUMMARY:")
        print("=" * 60)
        
        summary = report['summary']
        print(f"📊 Categories checked: {summary['categories_checked']}")
        print(f"🚨 Issues found: {summary['total_issues']}")
        print(f"⚠️ Warnings: {summary['total_warnings']}")
        print(f"💡 Recommendations: {summary['recommendations_count']}")
        
        # Show top issues by category
        validation_results = report['validation_results']
        
        categories_with_issues = []
        for category, results in validation_results.items():
            if isinstance(results, dict):
                issues = results.get('issues', [])
                if issues:
                    categories_with_issues.append((category, len(issues)))
        
        if categories_with_issues:
            print(f"\n🔍 Categories with issues:")
            for category, issue_count in sorted(categories_with_issues, key=lambda x: x[1], reverse=True):
                print(f"  {category}: {issue_count} issues")
        
        # Show top recommendations
        recommendations = report['recommendations']
        if recommendations:
            print(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec['category']}: {rec['issue']}")
                print(f"     Fix: {rec['fix']}")
        
        print("\n" + "=" * 60)
        if summary['total_issues'] == 0:
            print("🎉 All configurations are valid!")
        else:
            print(f"⚠️ Found {summary['total_issues']} issues that need attention")
        
        print("🛡️ SAFETY: All recommendations are non-destructive")

def main():
    """Main function"""
    validator = SafeConfigValidator()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Persian Legal AI Configuration Validator')
    parser.add_argument('--docker', action='store_true', help='Validate Docker configuration only')
    parser.add_argument('--env', action='store_true', help='Validate environment configuration only')
    parser.add_argument('--deps', action='store_true', help='Validate dependencies only')
    parser.add_argument('--security', action='store_true', help='Validate security configuration only')
    parser.add_argument('--report', action='store_true', help='Generate full validation report')
    
    args = parser.parse_args()
    
    if args.docker:
        result = validator._validate_docker_compose()
        print(json.dumps(result, indent=2))
    elif args.env:
        result = validator._validate_environment()
        print(json.dumps(result, indent=2))
    elif args.deps:
        result = validator._validate_dependencies()
        print(json.dumps(result, indent=2))
    elif args.security:
        result = validator._validate_security_config()
        print(json.dumps(result, indent=2))
    else:
        # Generate full validation report
        validator.generate_validation_report()

if __name__ == '__main__':
    main()