#!/usr/bin/env python3
"""
Deployment Validator for Persian Legal AI
Validates deployments without making changes
"""

import yaml
import json
import subprocess
from pathlib import Path
from datetime import datetime

class DeploymentValidator:
    def __init__(self):
        self.workspace = Path.cwd()
        self.validation_results = {}
        
    def validate_all_configs(self):
        """Check all configurations for issues"""
        print("🔍 Validating All Deployment Configurations...")
        
        results = {
            'docker_compose': self._validate_docker_compose(),
            'dockerfiles': self._validate_dockerfiles(),
            'environment': self._validate_environment(),
            'dependencies': self._validate_dependencies(),
            'network': self._validate_network_config(),
            'security': self._validate_security_config()
        }
        
        self.validation_results = results
        return results
    
    def _validate_docker_compose(self):
        """Validate docker-compose.yml"""
        compose_file = self.workspace / 'docker-compose.yml'
        
        if not compose_file.exists():
            return {'valid': False, 'error': 'docker-compose.yml not found'}
        
        try:
            # Parse YAML
            with open(compose_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Basic structure validation
            if 'services' not in config:
                return {'valid': False, 'error': 'No services defined'}
            
            # Check critical services
            required_services = ['backend', 'frontend', 'redis']
            missing_services = [s for s in required_services if s not in config['services']]
            
            if missing_services:
                return {'valid': False, 'error': f'Missing services: {missing_services}'}
            
            # Check for health checks
            services_without_health = []
            for service, conf in config['services'].items():
                if 'healthcheck' not in conf:
                    services_without_health.append(service)
            
            warnings = []
            if services_without_health:
                warnings.append(f'Services without health checks: {services_without_health}')
            
            return {'valid': True, 'warnings': warnings}
            
        except yaml.YAMLError as e:
            return {'valid': False, 'error': f'YAML syntax error: {e}'}
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {e}'}
    
    def _validate_dockerfiles(self):
        """Validate Dockerfiles"""
        dockerfiles = [
            'backend/Dockerfile',
            'persian-legal-ai-frontend/Dockerfile'
        ]
        
        results = {}
        for dockerfile_path in dockerfiles:
            dockerfile = self.workspace / dockerfile_path
            
            if not dockerfile.exists():
                results[dockerfile_path] = {'valid': False, 'error': 'File not found'}
                continue
            
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                
                # Basic validation
                if not content.strip().startswith('FROM'):
                    results[dockerfile_path] = {'valid': False, 'error': 'Must start with FROM instruction'}
                    continue
                
                # Check for security best practices
                warnings = []
                if 'USER root' in content or ('USER' not in content and 'FROM' in content):
                    warnings.append('Consider running as non-root user')
                
                if 'HEALTHCHECK' not in content:
                    warnings.append('Consider adding HEALTHCHECK instruction')
                
                results[dockerfile_path] = {'valid': True, 'warnings': warnings}
                
            except Exception as e:
                results[dockerfile_path] = {'valid': False, 'error': str(e)}
        
        return results
    
    def _validate_environment(self):
        """Validate environment configuration"""
        env_files = ['.env', '.env.production', '.env.local']
        
        env_found = False
        for env_file in env_files:
            if (self.workspace / env_file).exists():
                env_found = True
                break
        
        warnings = []
        if not env_found:
            warnings.append('No environment file found - using defaults')
        
        # Check for .env.production.example
        if (self.workspace / '.env.production.example').exists():
            return {'valid': True, 'warnings': warnings, 'template_available': True}
        else:
            warnings.append('No .env.production.example template found')
            return {'valid': True, 'warnings': warnings, 'template_available': False}
    
    def _validate_dependencies(self):
        """Validate project dependencies"""
        results = {}
        
        # Backend dependencies
        backend_req = self.workspace / 'backend' / 'requirements.txt'
        if backend_req.exists():
            try:
                with open(backend_req, 'r') as f:
                    content = f.read()
                
                # Check for critical dependencies
                critical_deps = ['fastapi', 'uvicorn', 'redis', 'torch']
                missing_deps = []
                for dep in critical_deps:
                    if dep not in content.lower():
                        missing_deps.append(dep)
                
                if missing_deps:
                    results['backend'] = {'valid': False, 'error': f'Missing dependencies: {missing_deps}'}
                else:
                    results['backend'] = {'valid': True}
                    
            except Exception as e:
                results['backend'] = {'valid': False, 'error': str(e)}
        else:
            results['backend'] = {'valid': False, 'error': 'requirements.txt not found'}
        
        # Frontend dependencies
        frontend_dirs = ['frontend', 'persian-legal-ai-frontend']
        frontend_found = False
        
        for frontend_dir in frontend_dirs:
            package_json = self.workspace / frontend_dir / 'package.json'
            if package_json.exists():
                frontend_found = True
                try:
                    with open(package_json, 'r') as f:
                        config = json.load(f)
                    
                    if 'scripts' in config and 'build' in config['scripts']:
                        results['frontend'] = {'valid': True}
                    else:
                        results['frontend'] = {'valid': False, 'error': 'No build script found'}
                        
                except Exception as e:
                    results['frontend'] = {'valid': False, 'error': str(e)}
                break
        
        if not frontend_found:
            results['frontend'] = {'valid': False, 'error': 'package.json not found'}
        
        return results
    
    def _validate_network_config(self):
        """Validate network configuration"""
        compose_file = self.workspace / 'docker-compose.yml'
        
        if not compose_file.exists():
            return {'valid': False, 'error': 'docker-compose.yml not found'}
        
        try:
            with open(compose_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check port mappings
            port_conflicts = []
            used_ports = []
            
            for service, conf in config.get('services', {}).items():
                if 'ports' in conf:
                    for port_mapping in conf['ports']:
                        if isinstance(port_mapping, str) and ':' in port_mapping:
                            external_port = port_mapping.split(':')[0].replace('"', '')
                            if external_port in used_ports:
                                port_conflicts.append(external_port)
                            used_ports.append(external_port)
            
            warnings = []
            if port_conflicts:
                warnings.append(f'Port conflicts detected: {port_conflicts}')
            
            return {'valid': True, 'warnings': warnings, 'used_ports': used_ports}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _validate_security_config(self):
        """Validate security configuration"""
        warnings = []
        recommendations = []
        
        # Check for secrets in docker-compose
        compose_file = self.workspace / 'docker-compose.yml'
        if compose_file.exists():
            try:
                with open(compose_file, 'r') as f:
                    content = f.read()
                
                # Check for hardcoded passwords
                if 'password' in content.lower() or 'secret' in content.lower():
                    warnings.append('Potential hardcoded secrets in docker-compose.yml')
                
                # Check for environment variable usage
                if '${' not in content:
                    recommendations.append('Consider using environment variables for configuration')
                
            except Exception:
                pass
        
        # Check for .env files in .gitignore
        gitignore = self.workspace / '.gitignore'
        if gitignore.exists():
            try:
                with open(gitignore, 'r') as f:
                    content = f.read()
                
                if '.env' not in content:
                    warnings.append('.env files should be in .gitignore')
                    
            except Exception:
                pass
        
        return {'valid': True, 'warnings': warnings, 'recommendations': recommendations}
    
    def generate_validation_report(self):
        """Generate validation report"""
        results = self.validate_all_configs()
        
        print("\n📊 Validation Report")
        print("=" * 50)
        
        total_issues = 0
        for category, result in results.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            
            if isinstance(result, dict) and 'valid' in result:
                if result['valid']:
                    print("  ✅ Valid")
                    if 'warnings' in result and result['warnings']:
                        for warning in result['warnings']:
                            print(f"  ⚠️ {warning}")
                else:
                    print(f"  ❌ Invalid: {result.get('error', 'Unknown error')}")
                    total_issues += 1
            else:
                # Handle nested results (like dockerfiles)
                for item, item_result in result.items():
                    if item_result['valid']:
                        print(f"  ✅ {item}: Valid")
                        if 'warnings' in item_result and item_result['warnings']:
                            for warning in item_result['warnings']:
                                print(f"    ⚠️ {warning}")
                    else:
                        print(f"  ❌ {item}: {item_result.get('error', 'Unknown error')}")
                        total_issues += 1
        
        print("\n" + "=" * 50)
        if total_issues == 0:
            print("🎉 All validations passed!")
        else:
            print(f"⚠️ Found {total_issues} issues that need attention")
        
        return results

def main():
    validator = DeploymentValidator()
    validator.generate_validation_report()

if __name__ == '__main__':
    main()
