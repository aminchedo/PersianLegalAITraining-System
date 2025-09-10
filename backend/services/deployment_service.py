"""
Deployment Service for Persian Legal AI
سرویس استقرار برای هوش مصنوعی حقوقی فارسی

Integrates with existing system architecture and extends functionality
"""

import logging
import asyncio
import subprocess
import json
import os
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel

# Import existing modules following established patterns
from ..config.database import get_database
from ..database.models import TrainingSession
from ..utils.persian_health_checker import HealthChecker

logger = logging.getLogger(__name__)

class DeploymentStatus(BaseModel):
    """Deployment status model following existing patterns"""
    status: str
    timestamp: datetime
    services_healthy: bool
    docker_available: bool
    configuration_valid: bool
    resource_sufficient: bool
    deployment_ready: bool

class DeploymentService:
    """
    Deployment service that integrates with existing Persian Legal AI architecture
    """
    
    def __init__(self):
        self.health_checker = HealthChecker()
        self.deployment_checks = {
            'docker': False,
            'docker_compose': False,
            'configuration': False,
            'resources': False,
            'services': False
        }
    
    async def check_deployment_health(self) -> Dict[str, Any]:
        """
        Check deployment health using existing health check patterns
        Extends the existing health endpoint functionality
        """
        try:
            # Use existing health checker
            base_health = await self.health_checker.get_system_health()
            
            # Add deployment-specific checks
            deployment_checks = await self._run_deployment_checks()
            
            # Calculate overall deployment readiness
            deployment_ready = all(deployment_checks.values())
            
            return {
                "status": "healthy" if deployment_ready else "issues_detected",
                "timestamp": datetime.utcnow().isoformat(),
                "base_system_health": base_health,
                "deployment_checks": deployment_checks,
                "deployment_ready": deployment_ready,
                "recommendations": await self._get_deployment_recommendations(deployment_checks)
            }
            
        except Exception as e:
            logger.error(f"Failed to check deployment health: {e}")
            return {
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "deployment_ready": False
            }
    
    async def _run_deployment_checks(self) -> Dict[str, bool]:
        """Run deployment-specific checks"""
        checks = {}
        
        # Check Docker availability
        checks['docker_available'] = await self._check_docker()
        
        # Check Docker Compose availability  
        checks['docker_compose_available'] = await self._check_docker_compose()
        
        # Check configuration files
        checks['configuration_valid'] = await self._check_configuration()
        
        # Check system resources
        checks['resources_sufficient'] = await self._check_resources()
        
        # Check existing services
        checks['services_healthy'] = await self._check_services()
        
        return checks
    
    async def _check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            result = await asyncio.create_subprocess_exec(
                'docker', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            return result.returncode == 0
        except Exception:
            return False
    
    async def _check_docker_compose(self) -> bool:
        """Check if Docker Compose is available"""
        try:
            # Try new docker compose syntax
            result = await asyncio.create_subprocess_exec(
                'docker', 'compose', 'version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            if result.returncode == 0:
                return True
            
            # Try old docker-compose syntax
            result = await asyncio.create_subprocess_exec(
                'docker-compose', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            return result.returncode == 0
        except Exception:
            return False
    
    async def _check_configuration(self) -> bool:
        """Check if configuration files are valid"""
        try:
            workspace = Path("/workspace")
            
            # Check for docker-compose.yml
            compose_file = workspace / "docker-compose.yml"
            if not compose_file.exists():
                return False
            
            # Validate docker-compose configuration
            result = await asyncio.create_subprocess_exec(
                'docker-compose', '-f', str(compose_file), 'config',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace
            )
            await result.wait()
            return result.returncode == 0
            
        except Exception:
            return False
    
    async def _check_resources(self) -> bool:
        """Check if system resources are sufficient"""
        try:
            # Check available memory (need at least 2GB)
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            # Check available disk space (need at least 5GB)
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            
            return available_gb >= 2.0 and free_gb >= 5.0
            
        except Exception:
            return False
    
    async def _check_services(self) -> bool:
        """Check if existing services are healthy using existing patterns"""
        try:
            # Use existing health checker
            health_result = await self.health_checker.get_system_health()
            return health_result.get('status') == 'healthy'
        except Exception:
            return False
    
    async def _get_deployment_recommendations(self, checks: Dict[str, bool]) -> List[str]:
        """Get deployment recommendations based on check results"""
        recommendations = []
        
        if not checks.get('docker_available', False):
            recommendations.append("Install Docker: Run ./deployment-helpers/install-docker.sh")
        
        if not checks.get('docker_compose_available', False):
            recommendations.append("Install Docker Compose or use 'docker compose' plugin")
        
        if not checks.get('configuration_valid', False):
            recommendations.append("Fix configuration issues: Run python3 safe_config_validator.py")
        
        if not checks.get('resources_sufficient', False):
            recommendations.append("Insufficient system resources: Need at least 2GB RAM and 5GB disk space")
        
        if not checks.get('services_healthy', False):
            recommendations.append("Fix service issues: Check logs and dependencies")
        
        if all(checks.values()):
            recommendations.append("✅ All checks passed! Ready to deploy with docker-compose up -d")
        
        return recommendations
    
    async def get_deployment_status(self) -> DeploymentStatus:
        """Get comprehensive deployment status"""
        try:
            health_data = await self.check_deployment_health()
            
            return DeploymentStatus(
                status=health_data.get('status', 'unknown'),
                timestamp=datetime.utcnow(),
                services_healthy=health_data.get('base_system_health', {}).get('status') == 'healthy',
                docker_available=health_data.get('deployment_checks', {}).get('docker_available', False),
                configuration_valid=health_data.get('deployment_checks', {}).get('configuration_valid', False),
                resource_sufficient=health_data.get('deployment_checks', {}).get('resources_sufficient', False),
                deployment_ready=health_data.get('deployment_ready', False)
            )
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return DeploymentStatus(
                status='error',
                timestamp=datetime.utcnow(),
                services_healthy=False,
                docker_available=False,
                configuration_valid=False,
                resource_sufficient=False,
                deployment_ready=False
            )
    
    async def validate_deployment_configuration(self) -> Dict[str, Any]:
        """Validate deployment configuration files"""
        try:
            workspace = Path("/workspace")
            validation_results = {}
            
            # Validate docker-compose files
            compose_files = [
                "docker-compose.yml",
                "docker-compose.enhanced.yml"
            ]
            
            for compose_file in compose_files:
                file_path = workspace / compose_file
                if file_path.exists():
                    validation_results[compose_file] = await self._validate_compose_file(file_path)
                else:
                    validation_results[compose_file] = {
                        "exists": False,
                        "required": compose_file == "docker-compose.yml"
                    }
            
            # Validate environment files
            env_files = [".env", ".env.production", ".env.production.example"]
            for env_file in env_files:
                file_path = workspace / env_file
                validation_results[env_file] = {
                    "exists": file_path.exists(),
                    "template": "example" in env_file
                }
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "validation_results": validation_results,
                "overall_valid": self._calculate_overall_validity(validation_results)
            }
            
        except Exception as e:
            logger.error(f"Failed to validate deployment configuration: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "overall_valid": False
            }
    
    async def _validate_compose_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate a docker-compose file"""
        try:
            result = await asyncio.create_subprocess_exec(
                'docker-compose', '-f', str(file_path), 'config',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=file_path.parent
            )
            stdout, stderr = await result.communicate()
            
            return {
                "exists": True,
                "valid": result.returncode == 0,
                "error": stderr.decode() if result.returncode != 0 else None
            }
            
        except Exception as e:
            return {
                "exists": True,
                "valid": False,
                "error": str(e)
            }
    
    def _calculate_overall_validity(self, validation_results: Dict[str, Any]) -> bool:
        """Calculate overall configuration validity"""
        try:
            # Check if main docker-compose.yml is valid
            main_compose = validation_results.get("docker-compose.yml", {})
            if not main_compose.get("exists") or not main_compose.get("valid"):
                return False
            
            # Check if we have at least one environment template
            env_template_exists = any(
                result.get("exists", False) and result.get("template", False)
                for result in validation_results.values()
                if isinstance(result, dict)
            )
            
            return env_template_exists
            
        except Exception:
            return False
    
    async def get_deployment_recommendations_detailed(self) -> Dict[str, Any]:
        """Get detailed deployment recommendations"""
        try:
            # Get current status
            status = await self.get_deployment_status()
            health_data = await self.check_deployment_health()
            
            recommendations = {
                "immediate_actions": [],
                "optimization_suggestions": [],
                "monitoring_setup": [],
                "security_improvements": []
            }
            
            # Immediate actions based on current status
            if not status.docker_available:
                recommendations["immediate_actions"].append({
                    "priority": "critical",
                    "action": "Install Docker",
                    "command": "./deployment-helpers/install-docker.sh",
                    "description": "Docker is required for containerized deployment"
                })
            
            if not status.configuration_valid:
                recommendations["immediate_actions"].append({
                    "priority": "high",
                    "action": "Fix configuration issues",
                    "command": "python3 safe_config_validator.py --report",
                    "description": "Configuration validation failed"
                })
            
            if not status.resource_sufficient:
                recommendations["immediate_actions"].append({
                    "priority": "medium",
                    "action": "Check system resources",
                    "command": "python3 resource_optimization_assistant.py --report",
                    "description": "Insufficient system resources detected"
                })
            
            # Optimization suggestions
            recommendations["optimization_suggestions"] = [
                {
                    "category": "Performance",
                    "suggestion": "Enable resource limits in docker-compose.yml",
                    "impact": "Prevents resource exhaustion"
                },
                {
                    "category": "Reliability",
                    "suggestion": "Add health checks to all services",
                    "impact": "Better service monitoring and recovery"
                },
                {
                    "category": "Security",
                    "suggestion": "Use environment variables for secrets",
                    "impact": "Improved security posture"
                }
            ]
            
            # Monitoring setup
            recommendations["monitoring_setup"] = [
                {
                    "tool": "Health Monitor",
                    "command": "python3 deployment_health_monitor.py --monitor",
                    "purpose": "Continuous service monitoring"
                },
                {
                    "tool": "Resource Monitor",
                    "command": "python3 resource_optimization_assistant.py --analyze",
                    "purpose": "System resource analysis"
                }
            ]
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "deployment_status": status.dict(),
                "recommendations": recommendations,
                "next_steps": await self._get_next_steps(status)
            }
            
        except Exception as e:
            logger.error(f"Failed to get detailed recommendations: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def _get_next_steps(self, status: DeploymentStatus) -> List[str]:
        """Get next steps based on current deployment status"""
        if status.deployment_ready:
            return [
                "✅ System is ready for deployment!",
                "Run: docker-compose up -d",
                "Monitor: python3 deployment_health_monitor.py --monitor"
            ]
        else:
            steps = ["🔧 Complete the following steps:"]
            
            if not status.docker_available:
                steps.append("1. Install Docker: ./deployment-helpers/install-docker.sh")
            
            if not status.configuration_valid:
                steps.append("2. Fix configuration: python3 safe_config_validator.py")
            
            if not status.resource_sufficient:
                steps.append("3. Check resources: python3 resource_optimization_assistant.py")
            
            steps.append("4. Re-run deployment check")
            
            return steps

# Global deployment service instance
deployment_service = DeploymentService()