#!/usr/bin/env python3
"""
Persian Legal AI Training System Launcher - Improved Version
توسط dreammaker طراحی شده

این اسکریپت هم backend و هم frontend را همزمان اجرا می‌کند
"""

import subprocess
import sys
import os
import time
import threading
import signal
import platform
import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

class Colors:
    """رنگ‌ها برای خروجی ترمینال"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(message, color):
    """چاپ پیام با رنگ"""
    print(f"{color}{message}{Colors.END}")

def print_banner():
    """نمایش بنر سیستم"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║          Persian Legal AI Training System Launcher          ║
    ║                    سیستم آموزش هوش مصنوعی حقوقی فارسی         ║
    ║                                                              ║
    ║                    Crafted by dreammaker                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print_colored(banner, Colors.CYAN + Colors.BOLD)

def check_command_exists(command: str) -> bool:
    """بررسی وجود یک command در سیستم"""
    return shutil.which(command) is not None

def get_command_version(command: str) -> Optional[str]:
    """دریافت نسخه یک command"""
    try:
        result = subprocess.run([command, "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None

def check_requirements():
    """بررسی وجود requirements"""
    print_colored("🔍 Checking system requirements...", Colors.YELLOW)
    
    requirements_met = True
    
    # بررسی Python
    try:
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            print_colored("❌ Python 3.8+ required!", Colors.RED)
            requirements_met = False
        else:
            print_colored(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}", Colors.GREEN)
    except Exception as e:
        print_colored(f"❌ Python check failed: {e}", Colors.RED)
        requirements_met = False
    
    # بررسی Node.js
    if check_command_exists("node"):
        node_version = get_command_version("node")
        if node_version:
            print_colored(f"✅ Node.js {node_version}", Colors.GREEN)
        else:
            print_colored("❌ Node.js version check failed", Colors.RED)
            requirements_met = False
    else:
        print_colored("❌ Node.js not found! Please install Node.js 16+", Colors.RED)
        requirements_met = False
    
    # بررسی npm
    if check_command_exists("npm"):
        npm_version = get_command_version("npm")
        if npm_version:
            print_colored(f"✅ npm {npm_version}", Colors.GREEN)
        else:
            print_colored("❌ npm version check failed", Colors.RED)
            requirements_met = False
    else:
        print_colored("❌ npm not found!", Colors.RED)
        requirements_met = False
    
    return requirements_met

def create_frontend_package_json():
    """ایجاد package.json برای frontend اگر وجود نداشته باشد"""
    frontend_path = Path("frontend")
    package_json_path = frontend_path / "package.json"
    
    if not package_json_path.exists():
        print_colored("📦 Creating package.json for frontend...", Colors.YELLOW)
        
        package_json = {
            "name": "persian-legal-ai-frontend",
            "private": True,
            "version": "0.0.0",
            "type": "module",
            "scripts": {
                "dev": "vite",
                "build": "tsc && vite build",
                "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
                "preview": "vite preview"
            },
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "react-router-dom": "^6.8.0",
                "recharts": "^2.5.0",
                "lucide-react": "^0.263.1",
                "axios": "^1.4.0"
            },
            "devDependencies": {
                "@types/react": "^18.2.15",
                "@types/react-dom": "^18.2.7",
                "@typescript-eslint/eslint-plugin": "^6.0.0",
                "@typescript-eslint/parser": "^6.0.0",
                "@vitejs/plugin-react": "^4.0.3",
                "eslint": "^8.45.0",
                "eslint-plugin-react-hooks": "^4.6.0",
                "eslint-plugin-react-refresh": "^0.4.3",
                "typescript": "^5.0.2",
                "vite": "^4.4.5"
            }
        }
        
        try:
            with open(package_json_path, 'w', encoding='utf-8') as f:
                json.dump(package_json, f, indent=2, ensure_ascii=False)
            print_colored("✅ package.json created", Colors.GREEN)
            return True
        except Exception as e:
            print_colored(f"❌ Failed to create package.json: {e}", Colors.RED)
            return False
    
    return True

def install_dependencies():
    """نصب dependencies"""
    print_colored("\n📦 Installing dependencies...", Colors.YELLOW)
    
    # نصب Python dependencies
    print_colored("Installing Python dependencies...", Colors.BLUE)
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, timeout=600)  # Increased timeout
        print_colored("✅ Python dependencies installed", Colors.GREEN)
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Failed to install Python dependencies: {e}", Colors.RED)
        return False
    except subprocess.TimeoutExpired:
        print_colored("❌ Python dependency installation timed out", Colors.RED)
        return False
    
    # بررسی و ایجاد package.json برای frontend
    if not create_frontend_package_json():
        return False
    
    # نصب Frontend dependencies
    frontend_path = Path("frontend")
    if frontend_path.exists():
        print_colored("Installing Frontend dependencies...", Colors.BLUE)
        try:
            subprocess.run(["npm", "install"], cwd=frontend_path, check=True, timeout=600)
            print_colored("✅ Frontend dependencies installed", Colors.GREEN)
        except subprocess.CalledProcessError as e:
            print_colored(f"❌ Failed to install frontend dependencies: {e}", Colors.RED)
            return False
        except subprocess.TimeoutExpired:
            print_colored("❌ Frontend dependency installation timed out", Colors.RED)
            return False
    
    return True

def find_backend_entry_point() -> Optional[str]:
    """پیدا کردن نقطه ورود backend"""
    possible_files = [
        "main.py",
        "backend/main.py", 
        "app.py",
        "backend/app.py",
        "run_dashboard.py"
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            # بررسی اینکه آیا فایل FastAPI app دارد
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'FastAPI' in content or 'app =' in content:
                        return file_path
            except Exception:
                continue
    
    # اگر هیچ فایل FastAPI پیدا نشد، اولین فایل موجود را برگردان
    for file_path in possible_files:
        if os.path.exists(file_path):
            return file_path
    
    return None

def run_backend():
    """اجرای backend"""
    print_colored("🚀 Starting Backend...", Colors.MAGENTA)
    try:
        backend_file = find_backend_entry_point()
        
        if not backend_file:
            print_colored("❌ Backend entry point not found!", Colors.RED)
            print_colored("Available files:", Colors.YELLOW)
            for file in ["main.py", "backend/main.py", "app.py", "run_dashboard.py"]:
                if os.path.exists(file):
                    print_colored(f"  - {file}", Colors.CYAN)
            return
        
        print_colored(f"📁 Using backend file: {backend_file}", Colors.BLUE)
        
        # تشخیص نوع backend
        if 'run_dashboard.py' in backend_file:
            # اجرای dashboard
            cmd = [sys.executable, backend_file]
        else:
            # اجرای با uvicorn
            module_name = backend_file.replace('.py', '').replace('/', '.')
            cmd = [sys.executable, "-m", "uvicorn", f"{module_name}:app", 
                   "--reload", "--host", "0.0.0.0", "--port", "8000"]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print_colored("\n🛑 Backend stopped by user", Colors.YELLOW)
    except Exception as e:
        print_colored(f"❌ Backend error: {e}", Colors.RED)

def run_frontend():
    """اجرای frontend"""
    print_colored("⚛️ Starting Frontend (React/Vite)...", Colors.CYAN)
    try:
        frontend_path = Path("frontend")
        if not frontend_path.exists():
            print_colored("❌ Frontend directory not found!", Colors.RED)
            return
        
        # بررسی وجود package.json
        package_json_path = frontend_path / "package.json"
        if not package_json_path.exists():
            print_colored("❌ package.json not found in frontend directory!", Colors.RED)
            return
        
        # اجرای frontend
        if platform.system() == "Windows":
            subprocess.run(["npm", "run", "dev"], cwd=frontend_path, shell=True)
        else:
            subprocess.run(["npm", "run", "dev"], cwd=frontend_path)
            
    except KeyboardInterrupt:
        print_colored("\n🛑 Frontend stopped by user", Colors.YELLOW)
    except Exception as e:
        print_colored(f"❌ Frontend error: {e}", Colors.RED)

def open_browser():
    """باز کردن مرورگر"""
    time.sleep(8)  # انتظار بیشتر برای startup
    try:
        import webbrowser
        print_colored("🌐 Opening browser...", Colors.GREEN)
        
        # باز کردن frontend
        webbrowser.open("http://localhost:3000")
        time.sleep(2)
        
        # باز کردن backend docs اگر FastAPI باشد
        webbrowser.open("http://localhost:8000/docs")
        
    except Exception as e:
        print_colored(f"⚠️ Could not open browser: {e}", Colors.YELLOW)

def cleanup_processes():
    """پاکسازی processes"""
    print_colored("🧹 Cleaning up processes...", Colors.YELLOW)
    
    if platform.system() != "Windows":
        try:
            # Kill uvicorn processes
            subprocess.run(["pkill", "-f", "uvicorn"], check=False)
            # Kill npm processes
            subprocess.run(["pkill", "-f", "npm"], check=False)
            # Kill node processes related to our project
            subprocess.run(["pkill", "-f", "vite"], check=False)
        except Exception as e:
            print_colored(f"⚠️ Cleanup warning: {e}", Colors.YELLOW)
    else:
        try:
            # Windows cleanup
            subprocess.run(["taskkill", "/F", "/IM", "python.exe"], check=False)
            subprocess.run(["taskkill", "/F", "/IM", "node.exe"], check=False)
        except Exception:
            pass

def signal_handler(signum, frame):
    """مدیریت signal ها"""
    print_colored("\n\n🛑 Shutting down Persian Legal AI Training System...", Colors.YELLOW + Colors.BOLD)
    cleanup_processes()
    print_colored("👋 Goodbye! Created with passion by dreammaker", Colors.CYAN)
    sys.exit(0)

def main():
    """تابع اصلی"""
    # تنظیم signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print_banner()
    
    # بررسی requirements
    if not check_requirements():
        print_colored("\n❌ Requirements not met. Please install missing components.", Colors.RED)
        return
    
    # سوال درباره نصب dependencies
    install_deps = input(f"\n{Colors.YELLOW}Install/Update dependencies? (y/n): {Colors.END}").lower().strip()
    if install_deps in ['y', 'yes', 'بله', '']:
        if not install_dependencies():
            print_colored("\n❌ Dependency installation failed.", Colors.RED)
            return
    
    print_colored("\n🎯 Starting Persian Legal AI Training System...", Colors.GREEN + Colors.BOLD)
    print_colored("✨ Backend will be available at: http://localhost:8000", Colors.CYAN)
    print_colored("✨ Frontend will be available at: http://localhost:3000", Colors.CYAN)
    print_colored("✨ API Documentation at: http://localhost:8000/docs", Colors.CYAN)
    print_colored("\n⚠️  Press Ctrl+C to stop all services\n", Colors.YELLOW)
    
    # ایجاد thread ها برای اجرای همزمان
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    frontend_thread = threading.Thread(target=run_frontend, daemon=True)
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    
    try:
        # شروع services
        backend_thread.start()
        time.sleep(3)  # انتظار برای backend startup
        frontend_thread.start()
        browser_thread.start()
        
        # انتظار برای thread ها
        backend_thread.join()
        frontend_thread.join()
        
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()