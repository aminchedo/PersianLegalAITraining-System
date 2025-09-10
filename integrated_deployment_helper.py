#!/usr/bin/env python3
"""
Integrated Deployment Helper for Persian Legal AI
راهنمای استقرار یکپارچه برای هوش مصنوعی حقوقی فارسی

This script integrates with the existing Persian Legal AI system architecture
and provides deployment assistance while preserving all existing functionality.
"""

import asyncio
import sys
import os
import subprocess
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add backend to Python path to import existing modules
sys.path.append(str(Path(__file__).parent / "backend"))

try:
    # Import existing modules following established patterns
    from backend.services.deployment_service import deployment_service
    from backend.config.database import get_database, test_connection
    from backend.database.migrations import get_database_health
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Backend modules not available: {e}")
    BACKEND_AVAILABLE = False

class IntegratedDeploymentHelper:
    """
    Integrated deployment helper that works with existing Persian Legal AI system
    """
    
    def __init__(self):
        self.workspace = Path("/workspace")
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3000"
        
    async def run_deployment_check(self):
        """Run comprehensive deployment check"""
        print("🚀 Persian Legal AI - Integrated Deployment Helper")
        print("راهنمای استقرار یکپارچه - سیستم هوش مصنوعی حقوقی فارسی")
        print("=" * 70)
        
        # Check if backend is running and use API
        if await self._check_backend_running():
            print("✅ Backend is running - using integrated API")
            await self._run_api_based_check()
        else:
            print("⚠️ Backend not running - using standalone checks")
            await self._run_standalone_check()
    
    async def _check_backend_running(self) -> bool:
        """Check if the backend is running"""
        try:
            response = requests.get(f"{self.backend_url}/api/system/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    async def _run_api_based_check(self):
        """Run deployment check using existing API endpoints"""
        try:
            print("\n🔍 Running API-based deployment checks...")
            
            # Use existing deployment endpoints
            endpoints = [
                ("/api/system/deployment/status", "Deployment Status"),
                ("/api/system/deployment/health", "Deployment Health"),
                ("/api/system/deployment/recommendations", "Recommendations")
            ]
            
            for endpoint, name in endpoints:
                print(f"\n📊 {name}:")
                try:
                    response = requests.get(f"{self.backend_url}{endpoint}", timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        self._display_api_result(name, data)
                    else:
                        print(f"  ❌ Failed to get {name}: HTTP {response.status_code}")
                except Exception as e:
                    print(f"  ❌ Error getting {name}: {e}")
            
            # Check if deployment is ready
            await self._check_deployment_readiness()
            
        except Exception as e:
            print(f"❌ API-based check failed: {e}")
            await self._run_standalone_check()
    
    def _display_api_result(self, name: str, data: Dict[str, Any]):
        """Display API result in a formatted way"""
        if name == "Deployment Status":
            status = data.get('status', 'unknown')
            ready = data.get('deployment_ready', False)
            
            print(f"  Status: {status}")
            print(f"  Deployment Ready: {'✅ Yes' if ready else '❌ No'}")
            print(f"  Docker Available: {'✅' if data.get('docker_available') else '❌'}")
            print(f"  Configuration Valid: {'✅' if data.get('configuration_valid') else '❌'}")
            print(f"  Resources Sufficient: {'✅' if data.get('resource_sufficient') else '❌'}")
            print(f"  Services Healthy: {'✅' if data.get('services_healthy') else '❌'}")
            
        elif name == "Deployment Health":
            status = data.get('status', 'unknown')
            checks = data.get('deployment_checks', {})
            recommendations = data.get('recommendations', [])
            
            print(f"  Overall Status: {status}")
            print(f"  Checks Passed: {sum(1 for v in checks.values() if v)}/{len(checks)}")
            
            if recommendations:
                print("  Recommendations:")
                for rec in recommendations[:3]:  # Show top 3
                    print(f"    • {rec}")
                    
        elif name == "Recommendations":
            immediate = data.get('recommendations', {}).get('immediate_actions', [])
            next_steps = data.get('next_steps', [])
            
            if immediate:
                print("  Immediate Actions:")
                for action in immediate:
                    print(f"    🚨 {action.get('action', 'Unknown')}")
                    if action.get('command'):
                        print(f"       Command: {action['command']}")
            
            if next_steps:
                print("  Next Steps:")
                for i, step in enumerate(next_steps[:3], 1):
                    print(f"    {i}. {step}")
    
    async def _check_deployment_readiness(self):
        """Check overall deployment readiness"""
        try:
            response = requests.get(f"{self.backend_url}/api/system/deployment/status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                ready = data.get('deployment_ready', False)
                
                print(f"\n🎯 DEPLOYMENT READINESS: {'✅ READY' if ready else '❌ NOT READY'}")
                
                if ready:
                    print("\n🚀 Your system is ready for deployment!")
                    print("   Run: docker-compose up -d")
                    print("   Monitor: Access deployment tab in the web interface")
                else:
                    print("\n🔧 Complete the following steps first:")
                    print("   1. Check the recommendations above")
                    print("   2. Fix any issues found")
                    print("   3. Run this script again")
                    
        except Exception as e:
            print(f"⚠️ Could not check deployment readiness: {e}")
    
    async def _run_standalone_check(self):
        """Run standalone deployment checks"""
        print("\n🔍 Running standalone deployment checks...")
        
        checks = {
            "Docker": await self._check_docker(),
            "Docker Compose": await self._check_docker_compose(),
            "Configuration": await self._check_configuration(),
            "Resources": await self._check_resources(),
            "Database": await self._check_database()
        }
        
        print("\n📋 Check Results:")
        for check_name, (passed, details) in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}: {details}")
        
        # Overall assessment
        passed_checks = sum(1 for passed, _ in checks.values() if passed)
        total_checks = len(checks)
        
        print(f"\n📊 Overall: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks == total_checks:
            print("\n🎉 All checks passed! System is ready for deployment.")
            print("   Run: docker-compose up -d")
        else:
            print("\n🔧 Please address the failed checks above.")
            await self._provide_standalone_recommendations(checks)
    
    async def _check_docker(self) -> tuple[bool, str]:
        """Check Docker availability"""
        try:
            result = await asyncio.create_subprocess_exec(
                'docker', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            if result.returncode == 0:
                return True, "Docker is available"
            else:
                return False, "Docker command failed"
        except Exception:
            return False, "Docker not installed"
    
    async def _check_docker_compose(self) -> tuple[bool, str]:
        """Check Docker Compose availability"""
        try:
            # Try new syntax first
            result = await asyncio.create_subprocess_exec(
                'docker', 'compose', 'version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            if result.returncode == 0:
                return True, "Docker Compose (plugin) available"
            
            # Try old syntax
            result = await asyncio.create_subprocess_exec(
                'docker-compose', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            if result.returncode == 0:
                return True, "Docker Compose available"
            else:
                return False, "Docker Compose not available"
        except Exception:
            return False, "Docker Compose not installed"
    
    async def _check_configuration(self) -> tuple[bool, str]:
        """Check configuration files"""
        compose_file = self.workspace / "docker-compose.yml"
        if not compose_file.exists():
            return False, "docker-compose.yml not found"
        
        try:
            result = await asyncio.create_subprocess_exec(
                'docker-compose', '-f', str(compose_file), 'config',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace
            )
            await result.wait()
            if result.returncode == 0:
                return True, "Configuration is valid"
            else:
                return False, "Configuration has errors"
        except Exception:
            return False, "Could not validate configuration"
    
    async def _check_resources(self) -> tuple[bool, str]:
        """Check system resources"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            available_gb = memory.available / (1024**3)
            free_gb = disk.free / (1024**3)
            
            if available_gb >= 2.0 and free_gb >= 5.0:
                return True, f"Sufficient resources ({available_gb:.1f}GB RAM, {free_gb:.1f}GB disk)"
            else:
                return False, f"Insufficient resources ({available_gb:.1f}GB RAM, {free_gb:.1f}GB disk)"
        except ImportError:
            return True, "Could not check resources (psutil not available)"
        except Exception:
            return False, "Resource check failed"
    
    async def _check_database(self) -> tuple[bool, str]:
        """Check database availability"""
        db_file = self.workspace / "persian_legal_ai.db"
        if db_file.exists():
            return True, f"Database exists ({db_file.stat().st_size / (1024*1024):.1f}MB)"
        else:
            return True, "Database will be created on first run"
    
    async def _provide_standalone_recommendations(self, checks: Dict[str, tuple[bool, str]]):
        """Provide recommendations for failed checks"""
        print("\n💡 Recommendations:")
        
        for check_name, (passed, details) in checks.items():
            if not passed:
                if check_name == "Docker":
                    print("   🐳 Install Docker: ./deployment-helpers/install-docker.sh")
                elif check_name == "Docker Compose":
                    print("   🔧 Install Docker Compose or use 'docker compose' plugin")
                elif check_name == "Configuration":
                    print("   ⚙️ Fix configuration: python3 safe_config_validator.py")
                elif check_name == "Resources":
                    print("   💾 Free up system resources or add more RAM/disk space")
    
    async def start_deployment_monitoring(self):
        """Start deployment monitoring"""
        print("\n🔄 Starting deployment monitoring...")
        
        if await self._check_backend_running():
            print("✅ Backend is running")
            print(f"📊 Access deployment monitoring at: {self.frontend_url}")
            print("   Navigate to the 'استقرار' (Deployment) tab")
            
            # Try to open browser
            try:
                import webbrowser
                webbrowser.open(f"{self.frontend_url}")
                print("🌐 Opening web interface...")
            except Exception:
                pass
        else:
            print("⚠️ Backend not running")
            print("   Start the backend first: docker-compose up -d backend")
    
    def show_help(self):
        """Show help information"""
        print("""
🚀 Persian Legal AI - Integrated Deployment Helper
راهنمای استقرار یکپارچه

Available commands:
  check     - Run deployment readiness check
  monitor   - Start deployment monitoring
  help      - Show this help message

Usage examples:
  python3 integrated_deployment_helper.py check
  python3 integrated_deployment_helper.py monitor
  python3 integrated_deployment_helper.py help

Integration features:
  ✅ Uses existing API endpoints when backend is running
  ✅ Integrates with existing health check system
  ✅ Provides web-based monitoring interface
  ✅ Preserves all existing functionality

For more detailed analysis:
  python3 safe_config_validator.py --report
  python3 resource_optimization_assistant.py --report
  python3 deployment_health_monitor.py --monitor
        """)

async def main():
    """Main function"""
    helper = IntegratedDeploymentHelper()
    
    if len(sys.argv) < 2:
        command = "check"
    else:
        command = sys.argv[1]
    
    if command == "check":
        await helper.run_deployment_check()
    elif command == "monitor":
        await helper.start_deployment_monitoring()
    elif command == "help":
        helper.show_help()
    else:
        print(f"Unknown command: {command}")
        helper.show_help()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())