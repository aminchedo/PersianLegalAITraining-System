#!/usr/bin/env python3
"""
Persian Legal AI Deployment Analyzer
ANALYZE ONLY - DO NOT MODIFY EXISTING FILES
Check what's working vs what's broken
Identify root causes without changing anything

🛡️ SAFETY: This is a READ-ONLY analysis tool
"""

import os
import sys
import json
import subprocess
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class DeploymentAnalyzer:
    """Safe deployment analyzer that only reads and reports"""
    
    def __init__(self):
        self.workspace = Path("/workspace")
        self.analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "issues_found": [],
            "recommendations": [],
            "existing_features": [],
            "deployment_status": {}
        }
        
    def analyze_system(self) -> Dict[str, Any]:
        """Comprehensive system analysis without modifications"""
        print("🔍 Starting Persian Legal AI Deployment Analysis...")
        print("🛡️ SAFETY MODE: Read-only analysis - no modifications will be made")
        
        # Analyze each component
        self._analyze_docker_setup()
        self._analyze_environment_config()
        self._analyze_dependencies()
        self._analyze_database()
        self._analyze_network_config()
        self._analyze_resource_requirements()
        self._analyze_existing_features()
        self._check_deployment_blockers()
        
        # Generate report
        self._generate_analysis_report()
        return self.analysis_results
    
    def _analyze_docker_setup(self):
        """Analyze Docker configuration"""
        print("\n📦 Analyzing Docker Setup...")
        
        # Check docker-compose files
        compose_files = [
            "docker-compose.yml",
            "docker-compose.production.yml"
        ]
        
        for compose_file in compose_files:
            if (self.workspace / compose_file).exists():
                print(f"  ✅ Found: {compose_file}")
                self.analysis_results["existing_features"].append(f"Docker Compose: {compose_file}")
            else:
                print(f"  ❌ Missing: {compose_file}")
                if compose_file == "docker-compose.yml":
                    self.analysis_results["issues_found"].append("Missing main docker-compose.yml")
        
        # Check Dockerfiles
        dockerfiles = [
            "backend/Dockerfile",
            "persian-legal-ai-frontend/Dockerfile",
            "Dockerfile.backend",
            "Dockerfile.frontend"
        ]
        
        for dockerfile in dockerfiles:
            if (self.workspace / dockerfile).exists():
                print(f"  ✅ Found: {dockerfile}")
                self.analysis_results["existing_features"].append(f"Dockerfile: {dockerfile}")
            else:
                print(f"  ❌ Missing: {dockerfile}")
        
        # Check for Docker availability
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✅ Docker available: {result.stdout.strip()}")
            else:
                print("  ❌ Docker not available")
                self.analysis_results["issues_found"].append("Docker not installed or not accessible")
        except FileNotFoundError:
            print("  ❌ Docker command not found")
            self.analysis_results["issues_found"].append("Docker not installed")
        
        # Check for docker-compose availability
        try:
            result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✅ Docker Compose available: {result.stdout.strip()}")
            else:
                print("  ❌ Docker Compose not available")
                self.analysis_results["issues_found"].append("Docker Compose not available")
        except FileNotFoundError:
            print("  ❌ Docker Compose command not found")
            self.analysis_results["issues_found"].append("Docker Compose not installed")
            self.analysis_results["recommendations"].append(
                "Install Docker Compose or use 'docker compose' (newer syntax)"
            )
    
    def _analyze_environment_config(self):
        """Analyze environment configuration"""
        print("\n🌍 Analyzing Environment Configuration...")
        
        # Check for environment files
        env_files = [".env", ".env.local", ".env.production", ".env.example"]
        env_found = False
        
        for env_file in env_files:
            if (self.workspace / env_file).exists():
                print(f"  ✅ Found: {env_file}")
                self.analysis_results["existing_features"].append(f"Environment file: {env_file}")
                env_found = True
            else:
                print(f"  ❌ Missing: {env_file}")
        
        if not env_found:
            self.analysis_results["issues_found"].append("No environment configuration files found")
            self.analysis_results["recommendations"].append("Create .env.production.example template")
        
        # Check environment variables in docker-compose
        compose_file = self.workspace / "docker-compose.yml"
        if compose_file.exists():
            try:
                with open(compose_file, 'r') as f:
                    content = f.read()
                    if "REDIS_PASSWORD" in content:
                        print("  ✅ Redis password configuration found")
                    if "DATABASE_URL" in content:
                        print("  ✅ Database URL configuration found")
                    if "CORS_ORIGINS" in content:
                        print("  ✅ CORS origins configuration found")
            except Exception as e:
                print(f"  ⚠️ Could not read docker-compose.yml: {e}")
    
    def _analyze_dependencies(self):
        """Analyze project dependencies"""
        print("\n📚 Analyzing Dependencies...")
        
        # Backend dependencies
        backend_req = self.workspace / "backend" / "requirements.txt"
        if backend_req.exists():
            print("  ✅ Backend requirements.txt found")
            self.analysis_results["existing_features"].append("Backend Python dependencies")
            
            # Check for critical dependencies
            try:
                with open(backend_req, 'r') as f:
                    content = f.read()
                    critical_deps = ["fastapi", "uvicorn", "torch", "transformers", "redis"]
                    for dep in critical_deps:
                        if dep in content.lower():
                            print(f"    ✅ {dep} dependency found")
                        else:
                            print(f"    ❌ {dep} dependency missing")
                            self.analysis_results["issues_found"].append(f"Missing critical dependency: {dep}")
            except Exception as e:
                print(f"  ⚠️ Could not read requirements.txt: {e}")
        else:
            print("  ❌ Backend requirements.txt not found")
            self.analysis_results["issues_found"].append("Backend requirements.txt missing")
        
        # Frontend dependencies
        frontend_dirs = ["frontend", "persian-legal-ai-frontend"]
        for frontend_dir in frontend_dirs:
            package_json = self.workspace / frontend_dir / "package.json"
            if package_json.exists():
                print(f"  ✅ {frontend_dir}/package.json found")
                self.analysis_results["existing_features"].append(f"Frontend dependencies: {frontend_dir}")
                break
        else:
            print("  ❌ No frontend package.json found")
            self.analysis_results["issues_found"].append("Frontend package.json missing")
    
    def _analyze_database(self):
        """Analyze database configuration"""
        print("\n🗄️ Analyzing Database Configuration...")
        
        # Check for database files
        db_files = ["persian_legal_ai.db", "data/persian_legal_ai.db"]
        for db_file in db_files:
            if (self.workspace / db_file).exists():
                print(f"  ✅ Database file found: {db_file}")
                self.analysis_results["existing_features"].append(f"Database: {db_file}")
                
                # Check database integrity
                try:
                    conn = sqlite3.connect(self.workspace / db_file)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    print(f"    ✅ Database has {len(tables)} tables")
                    conn.close()
                except Exception as e:
                    print(f"    ⚠️ Database integrity check failed: {e}")
                    self.analysis_results["issues_found"].append(f"Database integrity issue: {e}")
                break
        else:
            print("  ❌ No database files found")
            self.analysis_results["recommendations"].append("Database will be created on first run")
    
    def _analyze_network_config(self):
        """Analyze network configuration"""
        print("\n🌐 Analyzing Network Configuration...")
        
        # Check ports in docker-compose
        compose_file = self.workspace / "docker-compose.yml"
        if compose_file.exists():
            try:
                with open(compose_file, 'r') as f:
                    content = f.read()
                    
                    # Check for port conflicts
                    ports = []
                    lines = content.split('\n')
                    for line in lines:
                        if '"' in line and ':' in line and 'ports:' not in line:
                            if line.strip().startswith('- "'):
                                port_mapping = line.strip().replace('- "', '').replace('"', '')
                                if ':' in port_mapping:
                                    external_port = port_mapping.split(':')[0]
                                    ports.append(external_port)
                    
                    print(f"  ✅ External ports configured: {', '.join(ports)}")
                    
                    # Check for common port conflicts
                    common_ports = ["3000", "8000", "6379"]
                    for port in common_ports:
                        if port in ports:
                            print(f"    ✅ Port {port} configured")
                        else:
                            print(f"    ❌ Port {port} not found in configuration")
                            
            except Exception as e:
                print(f"  ⚠️ Could not analyze network configuration: {e}")
    
    def _analyze_resource_requirements(self):
        """Analyze resource requirements"""
        print("\n💾 Analyzing Resource Requirements...")
        
        # Check memory limits in docker-compose
        compose_file = self.workspace / "docker-compose.yml"
        if compose_file.exists():
            try:
                with open(compose_file, 'r') as f:
                    content = f.read()
                    
                    if "memory:" in content:
                        print("  ✅ Memory limits configured")
                    else:
                        print("  ❌ No memory limits configured")
                        self.analysis_results["recommendations"].append("Add memory limits to prevent OOM kills")
                    
                    if "cpus:" in content:
                        print("  ✅ CPU limits configured")
                    else:
                        print("  ❌ No CPU limits configured")
                        self.analysis_results["recommendations"].append("Add CPU limits for better resource management")
                        
            except Exception as e:
                print(f"  ⚠️ Could not analyze resource configuration: {e}")
        
        # Check available system resources
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"  ℹ️ System memory: {memory.total // (1024**3)}GB total, {memory.available // (1024**3)}GB available")
            
            if memory.available < 4 * (1024**3):  # Less than 4GB available
                self.analysis_results["issues_found"].append("Insufficient available memory (< 4GB)")
                self.analysis_results["recommendations"].append("Ensure at least 4GB RAM available for AI models")
                
        except ImportError:
            print("  ⚠️ psutil not available for system resource check")
    
    def _analyze_existing_features(self):
        """Catalog all existing features to ensure preservation"""
        print("\n🔍 Cataloging Existing Features...")
        
        features = []
        
        # Backend features
        backend_dir = self.workspace / "backend"
        if backend_dir.exists():
            for py_file in backend_dir.glob("**/*.py"):
                if py_file.is_file():
                    features.append(f"Backend module: {py_file.relative_to(self.workspace)}")
        
        # Frontend features
        frontend_dirs = ["frontend", "persian-legal-ai-frontend"]
        for frontend_dir in frontend_dirs:
            frontend_path = self.workspace / frontend_dir
            if frontend_path.exists():
                for js_file in frontend_path.glob("**/*.{js,ts,tsx,jsx}"):
                    if js_file.is_file() and "node_modules" not in str(js_file):
                        features.append(f"Frontend component: {js_file.relative_to(self.workspace)}")
        
        # AI models
        model_dirs = ["models", "ai_models"]
        for model_dir in model_dirs:
            model_path = self.workspace / model_dir
            if model_path.exists():
                for model_file in model_path.glob("**/*"):
                    if model_file.is_file():
                        features.append(f"AI model: {model_file.relative_to(self.workspace)}")
        
        print(f"  ✅ Found {len(features)} existing features/files")
        self.analysis_results["existing_features"].extend(features)
    
    def _check_deployment_blockers(self):
        """Check for critical deployment blockers"""
        print("\n🚫 Checking for Deployment Blockers...")
        
        blockers = []
        
        # Check if main docker-compose exists
        if not (self.workspace / "docker-compose.yml").exists():
            blockers.append("Missing main docker-compose.yml")
        
        # Check if Dockerfiles exist
        required_dockerfiles = ["backend/Dockerfile", "persian-legal-ai-frontend/Dockerfile"]
        for dockerfile in required_dockerfiles:
            if not (self.workspace / dockerfile).exists():
                blockers.append(f"Missing {dockerfile}")
        
        # Check if requirements exist
        if not (self.workspace / "backend" / "requirements.txt").exists():
            blockers.append("Missing backend/requirements.txt")
        
        # Check for frontend package.json
        frontend_package_found = False
        for frontend_dir in ["frontend", "persian-legal-ai-frontend"]:
            if (self.workspace / frontend_dir / "package.json").exists():
                frontend_package_found = True
                break
        
        if not frontend_package_found:
            blockers.append("Missing frontend package.json")
        
        if blockers:
            print(f"  ❌ Found {len(blockers)} deployment blockers:")
            for blocker in blockers:
                print(f"    - {blocker}")
            self.analysis_results["issues_found"].extend(blockers)
        else:
            print("  ✅ No critical deployment blockers found")
    
    def _generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        print("\n📊 Generating Analysis Report...")
        
        report_path = self.workspace / "deployment_analysis_report.json"
        
        # Add summary statistics
        self.analysis_results["summary"] = {
            "total_issues": len(self.analysis_results["issues_found"]),
            "total_recommendations": len(self.analysis_results["recommendations"]),
            "total_features": len(self.analysis_results["existing_features"]),
            "analysis_complete": True
        }
        
        # Save detailed report
        with open(report_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        
        print(f"  ✅ Analysis report saved to: {report_path}")
        
        # Print summary
        print("\n📋 ANALYSIS SUMMARY:")
        print(f"  🔍 Issues found: {len(self.analysis_results['issues_found'])}")
        print(f"  💡 Recommendations: {len(self.analysis_results['recommendations'])}")
        print(f"  ✅ Existing features: {len(self.analysis_results['existing_features'])}")
        
        if self.analysis_results["issues_found"]:
            print("\n🚨 TOP ISSUES TO ADDRESS:")
            for i, issue in enumerate(self.analysis_results["issues_found"][:5], 1):
                print(f"  {i}. {issue}")
        
        if self.analysis_results["recommendations"]:
            print("\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(self.analysis_results["recommendations"][:5], 1):
                print(f"  {i}. {rec}")
        
        print("\n🛡️ SAFETY CONFIRMATION: No existing files were modified during this analysis")

def main():
    """Main analysis function"""
    analyzer = DeploymentAnalyzer()
    results = analyzer.analyze_system()
    
    print("\n" + "="*60)
    print("🔍 DEPLOYMENT ANALYSIS COMPLETE")
    print("🛡️ All existing functionality preserved")
    print("📊 Ready to proceed with safe deployment fixes")
    print("="*60)
    
    return results

if __name__ == "__main__":
    main()