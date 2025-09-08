#!/usr/bin/env python3
"""
Persian Legal AI Frontend Functionality Testing Script
تست جامع عملکرد فرانت‌اند سیستم هوش مصنوعی حقوقی فارسی

This script provides comprehensive testing of the React frontend build process,
deployment verification, and API connectivity testing.
"""

import subprocess
import sys
import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import aiohttp
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('frontend_functionality_test.log')
    ]
)
logger = logging.getLogger(__name__)

class PersianLegalFrontendTester:
    """Comprehensive frontend functionality tester for Persian Legal AI system"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.frontend_dir = self.project_root / "frontend"
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "frontend_dir": str(self.frontend_dir),
            "tests_performed": [],
            "dependencies_check": {},
            "build_results": {},
            "dev_server_test": {},
            "api_connectivity": {},
            "persian_ui_tests": {},
            "responsive_tests": {},
            "performance_metrics": {}
        }
        
    async def run_comprehensive_test(self):
        """Run comprehensive frontend functionality tests"""
        print("🌐 Persian Legal AI - Frontend Functionality Testing")
        print("=" * 80)
        print(f"📁 Frontend Directory: {self.frontend_dir}")
        print(f"🕐 Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        try:
            # Phase 1: Dependencies and Environment Check
            await self._test_dependencies()
            
            # Phase 2: Build Process Testing
            await self._test_build_process()
            
            # Phase 3: Development Server Testing
            await self._test_dev_server()
            
            # Phase 4: API Connectivity Testing
            await self._test_api_connectivity()
            
            # Phase 5: Persian UI and RTL Testing
            await self._test_persian_ui()
            
            # Phase 6: Production Build Testing
            await self._test_production_build()
            
            # Generate final report
            self._generate_test_report()
            
        except Exception as e:
            logger.error(f"Critical frontend test error: {e}")
            self.test_results["critical_error"] = str(e)
    
    async def _test_dependencies(self):
        """Test Node.js dependencies and environment"""
        print("\n📦 Phase 1: Dependencies and Environment Check")
        print("-" * 50)
        
        dependencies_result = {
            "node_version": None,
            "npm_version": None,
            "package_json_exists": False,
            "node_modules_exists": False,
            "dependencies_installed": False,
            "typescript_available": False,
            "vite_available": False
        }
        
        # Check Node.js version
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                dependencies_result["node_version"] = result.stdout.strip()
                print(f"✅ Node.js: {dependencies_result['node_version']}")
            else:
                print("❌ Node.js: Not found")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ Node.js: Not available")
        
        # Check npm version
        try:
            result = subprocess.run(['npm', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                dependencies_result["npm_version"] = result.stdout.strip()
                print(f"✅ npm: {dependencies_result['npm_version']}")
            else:
                print("❌ npm: Not found")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ npm: Not available")
        
        # Check package.json
        package_json = self.frontend_dir / "package.json"
        if package_json.exists():
            dependencies_result["package_json_exists"] = True
            print("✅ package.json: Found")
            
            # Parse package.json
            try:
                with open(package_json, 'r', encoding='utf-8') as f:
                    package_data = json.load(f)
                    dependencies_result["package_info"] = {
                        "name": package_data.get("name"),
                        "version": package_data.get("version"),
                        "scripts": list(package_data.get("scripts", {}).keys()),
                        "dependencies_count": len(package_data.get("dependencies", {})),
                        "dev_dependencies_count": len(package_data.get("devDependencies", {}))
                    }
            except Exception as e:
                print(f"⚠️  package.json parsing error: {e}")
        else:
            print("❌ package.json: Not found")
        
        # Check node_modules
        node_modules = self.frontend_dir / "node_modules"
        if node_modules.exists() and node_modules.is_dir():
            dependencies_result["node_modules_exists"] = True
            print("✅ node_modules: Found")
        else:
            print("❌ node_modules: Not found - attempting npm install...")
            await self._install_dependencies()
        
        # Check TypeScript
        try:
            result = subprocess.run(['npx', 'tsc', '--version'], 
                                  cwd=self.frontend_dir, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                dependencies_result["typescript_available"] = True
                dependencies_result["typescript_version"] = result.stdout.strip()
                print(f"✅ TypeScript: {dependencies_result['typescript_version']}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ TypeScript: Not available")
        
        # Check Vite
        try:
            result = subprocess.run(['npx', 'vite', '--version'], 
                                  cwd=self.frontend_dir, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                dependencies_result["vite_available"] = True
                dependencies_result["vite_version"] = result.stdout.strip()
                print(f"✅ Vite: {dependencies_result['vite_version']}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ Vite: Not available")
        
        self.test_results["dependencies_check"] = dependencies_result
    
    async def _install_dependencies(self):
        """Install npm dependencies if needed"""
        print("📥 Installing npm dependencies...")
        
        try:
            result = subprocess.run(['npm', 'install'], 
                                  cwd=self.frontend_dir, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=300)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                self.test_results["dependencies_check"]["dependencies_installed"] = True
            else:
                print(f"❌ npm install failed: {result.stderr}")
                self.test_results["dependencies_check"]["install_error"] = result.stderr
        except subprocess.TimeoutExpired:
            print("❌ npm install timed out (5 minutes)")
            self.test_results["dependencies_check"]["install_error"] = "Timeout"
    
    async def _test_build_process(self):
        """Test the build process"""
        print("\n🔨 Phase 2: Build Process Testing")
        print("-" * 50)
        
        build_result = {
            "type_check_passed": False,
            "build_succeeded": False,
            "build_time": None,
            "build_output_size": None,
            "errors": [],
            "warnings": []
        }
        
        # Test TypeScript type checking
        print("🔍 Running TypeScript type check...")
        try:
            start_time = time.time()
            result = subprocess.run(['npm', 'run', 'type-check'], 
                                  cwd=self.frontend_dir, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=120)
            
            if result.returncode == 0:
                print("✅ TypeScript type check: PASSED")
                build_result["type_check_passed"] = True
            else:
                print(f"❌ TypeScript type check: FAILED")
                build_result["type_check_errors"] = result.stderr
        except subprocess.TimeoutExpired:
            print("⏱️  TypeScript type check: TIMEOUT")
            build_result["type_check_errors"] = "Timeout"
        except Exception as e:
            print(f"❌ TypeScript type check error: {e}")
            build_result["type_check_errors"] = str(e)
        
        # Test build process
        print("🏗️  Running production build...")
        try:
            start_time = time.time()
            result = subprocess.run(['npm', 'run', 'build'], 
                                  cwd=self.frontend_dir, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=300)
            
            build_time = time.time() - start_time
            build_result["build_time"] = build_time
            
            if result.returncode == 0:
                print(f"✅ Build: SUCCESS ({build_time:.1f}s)")
                build_result["build_succeeded"] = True
                
                # Check build output
                dist_dir = self.frontend_dir / "dist"
                if dist_dir.exists():
                    build_result["dist_exists"] = True
                    # Calculate build size
                    total_size = sum(f.stat().st_size for f in dist_dir.rglob('*') if f.is_file())
                    build_result["build_output_size"] = total_size
                    print(f"📦 Build output size: {total_size / 1024 / 1024:.1f} MB")
                else:
                    print("⚠️  dist directory not found after build")
                    build_result["dist_exists"] = False
            else:
                print(f"❌ Build: FAILED ({build_time:.1f}s)")
                build_result["build_errors"] = result.stderr
                print(f"   Error: {result.stderr[:200]}...")
                
        except subprocess.TimeoutExpired:
            print("⏱️  Build: TIMEOUT (5 minutes)")
            build_result["build_errors"] = "Timeout"
        except Exception as e:
            print(f"❌ Build error: {e}")
            build_result["build_errors"] = str(e)
        
        self.test_results["build_results"] = build_result
    
    async def _test_dev_server(self):
        """Test development server functionality"""
        print("\n🖥️  Phase 3: Development Server Testing")
        print("-" * 50)
        
        dev_server_result = {
            "server_started": False,
            "startup_time": None,
            "accessible": False,
            "response_time": None,
            "content_loaded": False
        }
        
        print("🚀 Starting development server...")
        
        try:
            # Start dev server in background
            env = os.environ.copy()
            env['VITE_API_URL'] = 'http://localhost:8000'
            
            process = subprocess.Popen(
                ['npm', 'run', 'dev', '--', '--host', '0.0.0.0', '--port', '3000'],
                cwd=self.frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env
            )
            
            # Wait for server to start
            print("⏳ Waiting for dev server to initialize...")
            start_time = time.time()
            
            for _ in range(30):  # Wait up to 30 seconds
                try:
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                        async with session.get('http://localhost:3000') as response:
                            if response.status == 200:
                                startup_time = time.time() - start_time
                                dev_server_result["server_started"] = True
                                dev_server_result["startup_time"] = startup_time
                                dev_server_result["accessible"] = True
                                print(f"✅ Dev server accessible ({startup_time:.1f}s)")
                                
                                # Test response time
                                response_start = time.time()
                                content = await response.text()
                                response_time = (time.time() - response_start) * 1000
                                dev_server_result["response_time"] = response_time
                                
                                # Check if content contains expected elements
                                if 'Persian Legal AI' in content or 'persian' in content.lower():
                                    dev_server_result["content_loaded"] = True
                                    print(f"✅ Content loaded correctly ({response_time:.1f}ms)")
                                else:
                                    print("⚠️  Content may not be loading correctly")
                                
                                break
                except:
                    pass
                
                await asyncio.sleep(1)
            
            # Cleanup
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            
            if not dev_server_result["server_started"]:
                print("❌ Dev server failed to start or become accessible")
                
        except Exception as e:
            print(f"❌ Dev server test error: {e}")
            dev_server_result["error"] = str(e)
        
        self.test_results["dev_server_test"] = dev_server_result
    
    async def _test_api_connectivity(self):
        """Test API connectivity from frontend perspective"""
        print("\n🔗 Phase 4: API Connectivity Testing")
        print("-" * 50)
        
        api_result = {
            "backend_accessible": False,
            "cors_configured": False,
            "api_endpoints_working": [],
            "response_times": {}
        }
        
        # Test backend accessibility
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8000/api/system/health') as response:
                    if response.status == 200:
                        api_result["backend_accessible"] = True
                        print("✅ Backend API: Accessible")
                        
                        # Check CORS headers
                        cors_headers = response.headers.get('Access-Control-Allow-Origin')
                        if cors_headers:
                            api_result["cors_configured"] = True
                            print(f"✅ CORS: Configured ({cors_headers})")
                        else:
                            print("⚠️  CORS: Headers not found")
                    else:
                        print(f"❌ Backend API: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Backend API: Not accessible - {e}")
        
        # Test key endpoints that frontend would use
        test_endpoints = [
            '/api/system/health',
            '/api/documents/stats',
            '/api/training/sessions'
        ]
        
        for endpoint in test_endpoints:
            try:
                start_time = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.get(f'http://localhost:8000{endpoint}') as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            api_result["api_endpoints_working"].append(endpoint)
                            api_result["response_times"][endpoint] = response_time
                            print(f"✅ {endpoint}: OK ({response_time:.1f}ms)")
                        else:
                            print(f"❌ {endpoint}: HTTP {response.status}")
            except Exception as e:
                print(f"❌ {endpoint}: Error - {e}")
        
        self.test_results["api_connectivity"] = api_result
    
    async def _test_persian_ui(self):
        """Test Persian UI and RTL functionality"""
        print("\n📝 Phase 5: Persian UI and RTL Testing")
        print("-" * 50)
        
        persian_ui_result = {
            "rtl_css_found": False,
            "persian_fonts_configured": False,
            "persian_text_components": [],
            "tailwind_rtl_support": False
        }
        
        # Check for RTL CSS configurations
        css_files = list(self.frontend_dir.glob("**/*.css"))
        for css_file in css_files:
            try:
                with open(css_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'rtl' in content.lower() or 'direction' in content.lower():
                        persian_ui_result["rtl_css_found"] = True
                        print(f"✅ RTL CSS found in: {css_file.name}")
                        break
            except Exception:
                pass
        
        # Check Tailwind configuration for RTL
        tailwind_config = self.frontend_dir / "tailwind.config.js"
        if tailwind_config.exists():
            try:
                with open(tailwind_config, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'rtl' in content.lower():
                        persian_ui_result["tailwind_rtl_support"] = True
                        print("✅ Tailwind RTL support: Configured")
                    else:
                        print("⚠️  Tailwind RTL support: Not explicitly configured")
            except Exception as e:
                print(f"❌ Tailwind config error: {e}")
        
        # Check for Persian text in components
        tsx_files = list(self.frontend_dir.glob("**/*.tsx"))
        persian_files = []
        
        for tsx_file in tsx_files[:10]:  # Check first 10 files
            try:
                with open(tsx_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for Persian characters
                    if any(ord(char) >= 0x0600 and ord(char) <= 0x06FF for char in content):
                        persian_files.append(tsx_file.name)
            except Exception:
                pass
        
        if persian_files:
            persian_ui_result["persian_text_components"] = persian_files
            print(f"✅ Persian text found in {len(persian_files)} components")
        else:
            print("⚠️  No Persian text found in components")
        
        self.test_results["persian_ui_tests"] = persian_ui_result
    
    async def _test_production_build(self):
        """Test production build and preview"""
        print("\n🚀 Phase 6: Production Build Testing")
        print("-" * 50)
        
        production_result = {
            "preview_server_works": False,
            "build_optimized": False,
            "assets_generated": False,
            "performance_score": None
        }
        
        # Check if build was successful (from previous phase)
        if not self.test_results["build_results"].get("build_succeeded"):
            print("⚠️  Skipping production tests - build failed")
            self.test_results["production_build"] = production_result
            return
        
        # Test preview server
        print("🔍 Testing production preview...")
        try:
            process = subprocess.Popen(
                ['npm', 'run', 'preview', '--', '--host', '0.0.0.0', '--port', '4173'],
                cwd=self.frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Wait for preview server
            await asyncio.sleep(5)
            
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get('http://localhost:4173') as response:
                        if response.status == 200:
                            production_result["preview_server_works"] = True
                            print("✅ Production preview: Working")
                        else:
                            print(f"❌ Production preview: HTTP {response.status}")
            except Exception as e:
                print(f"❌ Production preview: Error - {e}")
            
            # Cleanup
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                
        except Exception as e:
            print(f"❌ Preview server error: {e}")
        
        # Check build optimization
        dist_dir = self.frontend_dir / "dist"
        if dist_dir.exists():
            # Check for minified files
            js_files = list(dist_dir.glob("**/*.js"))
            css_files = list(dist_dir.glob("**/*.css"))
            
            if js_files and css_files:
                production_result["assets_generated"] = True
                print(f"✅ Assets generated: {len(js_files)} JS, {len(css_files)} CSS files")
                
                # Check if files are minified (simple heuristic)
                sample_js = js_files[0] if js_files else None
                if sample_js:
                    with open(sample_js, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content.split('\n')) < 10:  # Minified files have few lines
                            production_result["build_optimized"] = True
                            print("✅ Build optimization: Files appear minified")
                        else:
                            print("⚠️  Build optimization: Files may not be minified")
        
        self.test_results["production_build"] = production_result
    
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📋 Frontend Functionality Test Report")
        print("=" * 80)
        
        # Calculate overall score
        scores = []
        
        # Dependencies score
        deps = self.test_results["dependencies_check"]
        deps_score = sum([
            deps.get("node_version") is not None,
            deps.get("npm_version") is not None,
            deps.get("package_json_exists", False),
            deps.get("node_modules_exists", False) or deps.get("dependencies_installed", False),
            deps.get("typescript_available", False),
            deps.get("vite_available", False)
        ]) / 6 * 100
        scores.append(deps_score)
        
        # Build score
        build = self.test_results["build_results"]
        build_score = sum([
            build.get("type_check_passed", False),
            build.get("build_succeeded", False),
            build.get("dist_exists", False)
        ]) / 3 * 100
        scores.append(build_score)
        
        # Dev server score
        dev = self.test_results["dev_server_test"]
        dev_score = sum([
            dev.get("server_started", False),
            dev.get("accessible", False),
            dev.get("content_loaded", False)
        ]) / 3 * 100
        scores.append(dev_score)
        
        # API connectivity score
        api = self.test_results["api_connectivity"]
        api_score = sum([
            api.get("backend_accessible", False),
            api.get("cors_configured", False),
            len(api.get("api_endpoints_working", [])) > 0
        ]) / 3 * 100
        scores.append(api_score)
        
        overall_score = sum(scores) / len(scores)
        
        print(f"📊 Test Summary:")
        print(f"   Dependencies: {deps_score:.1f}/100")
        print(f"   Build Process: {build_score:.1f}/100")
        print(f"   Dev Server: {dev_score:.1f}/100")
        print(f"   API Connectivity: {api_score:.1f}/100")
        print(f"   Overall Score: {overall_score:.1f}/100")
        
        # Performance metrics
        if build.get("build_time"):
            print(f"\n⚡ Performance Metrics:")
            print(f"   Build Time: {build['build_time']:.1f}s")
            if build.get("build_output_size"):
                print(f"   Build Size: {build['build_output_size'] / 1024 / 1024:.1f} MB")
        
        if dev.get("response_time"):
            print(f"   Dev Server Response: {dev['response_time']:.1f}ms")
        
        # Persian UI status
        persian = self.test_results["persian_ui_tests"]
        if persian.get("persian_text_components"):
            print(f"\n📝 Persian UI Status:")
            print(f"   RTL CSS: {'✅' if persian.get('rtl_css_found') else '❌'}")
            print(f"   Persian Components: {len(persian.get('persian_text_components', []))}")
            print(f"   Tailwind RTL: {'✅' if persian.get('tailwind_rtl_support') else '⚠️'}")
        
        # Overall assessment
        print(f"\n🎯 Frontend Assessment:")
        if overall_score >= 90:
            print(f"   🎉 EXCELLENT: Frontend is production ready ({overall_score:.1f}/100)")
        elif overall_score >= 80:
            print(f"   ✅ GOOD: Frontend is mostly functional ({overall_score:.1f}/100)")
        elif overall_score >= 60:
            print(f"   ⚠️  FAIR: Frontend has some issues ({overall_score:.1f}/100)")
        else:
            print(f"   ❌ POOR: Frontend needs significant work ({overall_score:.1f}/100)")
        
        # Save detailed report
        report_file = Path("frontend_test_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        print("=" * 80)

async def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Persian Legal AI Frontend Functionality Tester")
    parser.add_argument("--project-root", type=Path, 
                       help="Project root directory (default: parent of script directory)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    tester = PersianLegalFrontendTester(args.project_root)
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())