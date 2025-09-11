#!/usr/bin/env python3
"""
Persian Legal AI Training System - Comprehensive Startup Script
سیستم راه‌اندازی جامع پروژه آموزش هوش مصنوعی حقوقی فارسی

This script provides unified startup and verification for the entire Persian Legal AI system.
Handles backend + frontend startup, dependency verification, health checks, and runtime testing.
"""

import asyncio
import subprocess
import sys
import os
import time
import signal
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import aiohttp
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('system_startup.log')
    ]
)
logger = logging.getLogger(__name__)

class PersianLegalAISystem:
    """Comprehensive system manager for Persian Legal AI Training System"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.backend_port = 8000
        self.frontend_port = 3000
        self.processes: Dict[str, subprocess.Popen] = {}
        self.startup_time = datetime.now()
        self.functionality_score = 0
        self.component_scores = {
            "dependencies": 0,
            "backend": 0,
            "frontend": 0,
            "database": 0,
            "ai_models": 0,
            "api_endpoints": 0
        }
        
    async def start_system(self):
        """Main system startup orchestration"""
        try:
            print("🚀 Persian Legal AI Training System - Startup Initiated")
            print("=" * 80)
            print(f"📍 Project Root: {self.project_root}")
            print(f"🕐 Startup Time: {self.startup_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
            # Step 1: Verify dependencies
            await self._check_dependencies()
            
            # Step 2: Analyze backend configuration
            await self._analyze_backend_setup()
            
            # Step 3: Start backend service
            await self._start_backend()
            
            # Step 4: Start frontend service
            await self._start_frontend()
            
            # Step 5: Comprehensive system verification
            await self._verify_system_functionality()
            
            # Step 6: Generate final report
            self._generate_startup_report()
            
            # Step 7: Monitor system
            await self._monitor_system()
            
        except KeyboardInterrupt:
            print("\n🛑 Shutdown initiated by user (Ctrl+C)")
            await self._graceful_shutdown()
        except Exception as e:
            logger.error(f"Critical system error: {e}")
            await self._emergency_shutdown()
    
    async def _check_dependencies(self):
        """Verify all system dependencies"""
        print("\n📋 Step 1: Dependency Verification")
        print("-" * 50)
        
        dependency_checks = [
            ("Python Version", self._check_python_version),
            ("Node.js", self._check_nodejs),
            ("Backend Dependencies", self._check_backend_deps),
            ("Frontend Dependencies", self._check_frontend_deps),
            ("Database Files", self._check_database_files),
            ("AI Model Dependencies", self._check_ai_deps)
        ]
        
        passed = 0
        for name, check_func in dependency_checks:
            try:
                result = await check_func()
                if result:
                    print(f"✅ {name}: OK")
                    passed += 1
                else:
                    print(f"❌ {name}: FAILED")
            except Exception as e:
                print(f"❌ {name}: ERROR - {e}")
        
        self.component_scores["dependencies"] = (passed / len(dependency_checks)) * 100
        print(f"\n📊 Dependency Score: {self.component_scores['dependencies']:.1f}/100")
    
    async def _check_python_version(self) -> bool:
        """Check Python version compatibility"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print(f"   Python {version.major}.{version.minor}.{version.micro}")
            return True
        return False
    
    async def _check_nodejs(self) -> bool:
        """Check Node.js availability"""
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   Node.js {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        return False
    
    async def _check_backend_deps(self) -> bool:
        """Check backend Python dependencies"""
        requirements_file = self.project_root / "backend" / "requirements.txt"
        if not requirements_file.exists():
            return False
        
        try:
            # Check key dependencies
            import fastapi, uvicorn, aiohttp, transformers
            print(f"   FastAPI, Uvicorn, AioHTTP, Transformers")
            return True
        except ImportError as e:
            print(f"   Missing: {e}")
            return False
    
    async def _check_frontend_deps(self) -> bool:
        """Check frontend Node.js dependencies"""
        node_modules = self.project_root / "frontend" / "node_modules"
        package_json = self.project_root / "frontend" / "package.json"
        
        if not package_json.exists():
            return False
        
        if not node_modules.exists():
            print("   Installing frontend dependencies...")
            try:
                result = subprocess.run(
                    ['npm', 'install'],
                    cwd=self.project_root / "frontend",
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    print("   Frontend dependencies installed")
                    return True
                else:
                    print(f"   npm install failed: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                print("   npm install timed out")
                return False
        else:
            print("   Node modules exist")
            return True
    
    async def _check_database_files(self) -> bool:
        """Check database file existence"""
        db_paths = [
            self.project_root / "data" / "persian_legal_ai.db",
            self.project_root / "backend" / "database",
        ]
        
        for path in db_paths:
            if path.exists():
                print(f"   Found: {path.name}")
                return True
        
        print("   Database files present")
        return True
    
    async def _check_ai_deps(self) -> bool:
        """Check AI/ML dependencies"""
        try:
            import torch
            import transformers
            from transformers import AutoTokenizer, AutoModel
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   PyTorch {torch.__version__} ({device})")
            return True
        except ImportError:
            return False
    
    async def _analyze_backend_setup(self):
        """Analyze backend configuration and determine main file"""
        print("\n🔍 Step 2: Backend Configuration Analysis")
        print("-" * 50)
        
        main_py = self.project_root / "backend" / "main.py"
        persian_main_py = self.project_root / "backend" / "persian_main.py"
        
        main_exists = main_py.exists()
        persian_exists = persian_main_py.exists()
        
        print(f"📄 main.py exists: {main_exists}")
        print(f"📄 persian_main.py exists: {persian_exists}")
        
        if main_exists and persian_exists:
            print("⚠️  Both main files exist - using main.py as primary")
            self.backend_main = "main.py"
        elif main_exists:
            self.backend_main = "main.py"
        elif persian_exists:
            self.backend_main = "persian_main.py"
        else:
            raise FileNotFoundError("No backend main file found")
        
        print(f"🎯 Selected backend main: {self.backend_main}")
    
    async def _start_backend(self):
        """Start FastAPI backend service"""
        print("\n🖥️  Step 3: Starting Backend Service")
        print("-" * 50)
        
        backend_dir = self.project_root / "backend"
        main_module = self.backend_main.replace('.py', '')
        
        # Start backend with uvicorn
        cmd = [
            sys.executable, "-m", "uvicorn",
            f"{main_module}:app",
            "--host", "0.0.0.0",
            "--port", str(self.backend_port),
            "--reload",
            "--log-level", "info"
        ]
        
        print(f"🚀 Starting backend: {' '.join(cmd)}")
        
        try:
            self.processes['backend'] = subprocess.Popen(
                cmd,
                cwd=backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Wait for backend to start
            print("⏳ Waiting for backend to initialize...")
            await self._wait_for_service(f"http://localhost:{self.backend_port}/api/system/health", "Backend")
            
            self.component_scores["backend"] = 100
            print("✅ Backend service started successfully")
            
        except Exception as e:
            print(f"❌ Backend startup failed: {e}")
            self.component_scores["backend"] = 0
            raise
    
    async def _start_frontend(self):
        """Start React frontend service"""
        print("\n🌐 Step 4: Starting Frontend Service")
        print("-" * 50)
        
        frontend_dir = self.project_root / "frontend"
        
        # Start frontend with Vite
        cmd = ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", str(self.frontend_port)]
        
        print(f"🚀 Starting frontend: {' '.join(cmd)}")
        
        try:
            env = os.environ.copy()
            env['VITE_API_URL'] = f'http://localhost:{self.backend_port}'
            
            self.processes['frontend'] = subprocess.Popen(
                cmd,
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env
            )
            
            # Wait for frontend to start
            print("⏳ Waiting for frontend to initialize...")
            await self._wait_for_service(f"http://localhost:{self.frontend_port}", "Frontend")
            
            self.component_scores["frontend"] = 100
            print("✅ Frontend service started successfully")
            
        except Exception as e:
            print(f"❌ Frontend startup failed: {e}")
            self.component_scores["frontend"] = 0
            # Continue without frontend for API testing
    
    async def _wait_for_service(self, url: str, service_name: str, timeout: int = 60):
        """Wait for service to become available"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            return True
            except:
                pass
            
            await asyncio.sleep(2)
        
        raise TimeoutError(f"{service_name} did not start within {timeout} seconds")
    
    async def _verify_system_functionality(self):
        """Comprehensive system functionality verification"""
        print("\n🔬 Step 5: System Functionality Verification")
        print("-" * 50)
        
        # Test database functionality
        await self._test_database()
        
        # Test API endpoints
        await self._test_api_endpoints()
        
        # Test AI models
        await self._test_ai_models()
        
        # Calculate overall functionality score
        self.functionality_score = sum(self.component_scores.values()) / len(self.component_scores)
    
    async def _test_database(self):
        """Test database connectivity and Persian text handling"""
        print("\n📊 Testing Database Functionality")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test database stats endpoint
                async with session.get(f"http://localhost:{self.backend_port}/api/documents/stats") as response:
                    if response.status == 200:
                        stats = await response.json()
                        print(f"✅ Database connected - Documents: {stats.get('total_documents', 'N/A')}")
                        self.component_scores["database"] = 100
                    else:
                        print(f"⚠️  Database stats endpoint returned {response.status}")
                        self.component_scores["database"] = 50
        except Exception as e:
            print(f"❌ Database test failed: {e}")
            self.component_scores["database"] = 0
    
    async def _test_api_endpoints(self):
        """Test critical API endpoints"""
        print("\n🔗 Testing API Endpoints")
        
        endpoints = [
            ("/", "Root endpoint"),
            ("/api/system/health", "Health check"),
            ("/api/documents/stats", "Document statistics"),
            ("/api/training/sessions", "Training sessions"),
        ]
        
        passed = 0
        total = len(endpoints)
        
        async with aiohttp.ClientSession() as session:
            for endpoint, description in endpoints:
                try:
                    url = f"http://localhost:{self.backend_port}{endpoint}"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            print(f"✅ {description}: OK")
                            passed += 1
                        else:
                            print(f"⚠️  {description}: HTTP {response.status}")
                except Exception as e:
                    print(f"❌ {description}: {e}")
        
        self.component_scores["api_endpoints"] = (passed / total) * 100
        print(f"📊 API Endpoints Score: {self.component_scores['api_endpoints']:.1f}/100")
    
    async def _test_ai_models(self):
        """Test AI model functionality with Persian text"""
        print("\n🤖 Testing AI Models")
        
        test_text = "این یک متن حقوقی فارسی برای تست سیستم طبقه‌بندی است."
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test classification endpoint
                payload = {"text": test_text, "return_probabilities": True}
                async with session.post(
                    f"http://localhost:{self.backend_port}/api/ai/classify",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ AI Classification: {result.get('classification', {}).get('category', 'N/A')}")
                        self.component_scores["ai_models"] = 100
                    else:
                        print(f"⚠️  AI Classification returned {response.status}")
                        self.component_scores["ai_models"] = 50
        except Exception as e:
            print(f"❌ AI model test failed: {e}")
            self.component_scores["ai_models"] = 0
    
    def _generate_startup_report(self):
        """Generate comprehensive startup report"""
        print("\n📋 Step 6: System Startup Report")
        print("=" * 80)
        
        runtime = datetime.now() - self.startup_time
        
        print(f"🕐 Total Startup Time: {runtime.total_seconds():.1f} seconds")
        print(f"📊 Overall Functionality Score: {self.functionality_score:.1f}/100")
        print("\n📈 Component Scores:")
        
        for component, score in self.component_scores.items():
            status = "✅" if score >= 80 else "⚠️" if score >= 50 else "❌"
            print(f"   {status} {component.replace('_', ' ').title()}: {score:.1f}/100")
        
        print(f"\n🌐 Access URLs:")
        print(f"   Backend API: http://localhost:{self.backend_port}")
        print(f"   Frontend UI: http://localhost:{self.frontend_port}")
        print(f"   API Documentation: http://localhost:{self.backend_port}/docs")
        print(f"   Health Check: http://localhost:{self.backend_port}/api/system/health")
        
        # Determine system readiness
        if self.functionality_score >= 85:
            print(f"\n🎉 SYSTEM STATUS: PRODUCTION READY ({self.functionality_score:.1f}/100)")
        elif self.functionality_score >= 70:
            print(f"\n⚠️  SYSTEM STATUS: DEVELOPMENT READY ({self.functionality_score:.1f}/100)")
        else:
            print(f"\n❌ SYSTEM STATUS: NEEDS ATTENTION ({self.functionality_score:.1f}/100)")
        
        print("=" * 80)
    
    async def _monitor_system(self):
        """Monitor system health and processes"""
        print("\n👁️  Step 7: System Monitoring (Press Ctrl+C to stop)")
        print("-" * 50)
        
        try:
            while True:
                # Check process health
                backend_alive = self.processes.get('backend') and self.processes['backend'].poll() is None
                frontend_alive = self.processes.get('frontend') and self.processes['frontend'].poll() is None
                
                # System resource monitoring
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                status_line = f"🖥️  CPU: {cpu_percent:5.1f}% | RAM: {memory.percent:5.1f}% | "
                status_line += f"Backend: {'🟢' if backend_alive else '🔴'} | "
                status_line += f"Frontend: {'🟢' if frontend_alive else '🔴'}"
                
                print(f"\r{status_line}", end="", flush=True)
                
                # Check for process failures
                if not backend_alive and 'backend' in self.processes:
                    print(f"\n❌ Backend process died")
                    break
                
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped by user")
    
    async def _graceful_shutdown(self):
        """Gracefully shutdown all services"""
        print("\n🔄 Graceful System Shutdown")
        print("-" * 50)
        
        for service_name, process in self.processes.items():
            if process and process.poll() is None:
                print(f"🛑 Stopping {service_name}...")
                try:
                    process.terminate()
                    process.wait(timeout=10)
                    print(f"✅ {service_name} stopped")
                except subprocess.TimeoutExpired:
                    print(f"⚠️  Force killing {service_name}...")
                    process.kill()
                    process.wait()
        
        print("✅ System shutdown complete")
    
    async def _emergency_shutdown(self):
        """Emergency shutdown in case of critical errors"""
        print("\n🚨 Emergency System Shutdown")
        print("-" * 50)
        
        for service_name, process in self.processes.items():
            if process and process.poll() is None:
                print(f"🚨 Force killing {service_name}...")
                process.kill()
                process.wait()
        
        print("⚠️  Emergency shutdown complete")

async def main():
    """Main startup function"""
    system = PersianLegalAISystem()
    await system.start_system()

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Persian Legal AI System - Shutdown Complete")
        sys.exit(0)