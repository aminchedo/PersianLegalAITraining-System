#!/usr/bin/env python3
"""
Persian Legal AI Training System Launcher - Fully Functional
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
import webbrowser
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

def check_project_structure():
    """بررسی ساختار پروژه"""
    print_colored("📁 Checking project structure...", Colors.YELLOW)
    
    required_files = [
        "requirements.txt",
        "package.json",
        "backend/main.py",
        "frontend/package.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print_colored(f"✅ {file_path}", Colors.GREEN)
    
    if missing_files:
        print_colored("❌ Missing required files:", Colors.RED)
        for file_path in missing_files:
            print_colored(f"   - {file_path}", Colors.RED)
        return False
    
    return True

def install_dependencies():
    """نصب dependencies"""
    print_colored("\n📦 Installing dependencies...", Colors.YELLOW)
    
    # نصب Python dependencies
    print_colored("Installing Python dependencies...", Colors.BLUE)
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, timeout=600)
        print_colored("✅ Python dependencies installed", Colors.GREEN)
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Failed to install Python dependencies: {e}", Colors.RED)
        return False
    except subprocess.TimeoutExpired:
        print_colored("❌ Python dependency installation timed out", Colors.RED)
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

def run_backend():
    """اجرای backend"""
    print_colored("🚀 Starting Backend (FastAPI)...", Colors.MAGENTA)
    try:
        # اجرای backend با uvicorn
        cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", 
               "--reload", "--host", "0.0.0.0", "--port", "8000"]
        
        print_colored(f"Running: {' '.join(cmd)}", Colors.BLUE)
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
    time.sleep(8)  # انتظار برای startup
    try:
        print_colored("🌐 Opening browser...", Colors.GREEN)
        
        # باز کردن frontend
        webbrowser.open("http://localhost:3000")
        time.sleep(2)
        
        # باز کردن backend docs
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

def show_help():
    """نمایش راهنما"""
    help_text = """
    🚀 Persian Legal AI Training System - Available Commands:
    
    📦 Setup Commands:
       npm run setup     - Install all dependencies
       npm run start     - Start full system (this script)
       npm run dev       - Start full system (alias)
    
    🔧 Individual Services:
       npm run backend   - Start backend only (port 8000)
       npm run frontend  - Start frontend only (port 3000)
    
    🧪 Testing & Validation:
       npm run test      - Run Python tests
       npm run test:full - Run full system tests
       npm run validate  - Validate system
    
    🐳 Docker Commands:
       npm run docker:build - Build Docker containers
       npm run docker:up    - Start with Docker
       npm run docker:down  - Stop Docker containers
    
    🛠️ Development Tools:
       npm run format    - Format code (black + isort)
       npm run lint      - Lint code
       npm run clean     - Clean cache files
       npm run docs      - Open API documentation
       npm run health    - Check system health
    
    🌐 URLs:
       Frontend: http://localhost:3000
       Backend:  http://localhost:8000
       API Docs: http://localhost:8000/docs
    """
    print_colored(help_text, Colors.CYAN)

def main():
    """تابع اصلی"""
    # تنظیم signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print_banner()
    
    # بررسی arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h', 'help']:
            show_help()
            return
        elif sys.argv[1] == '--no-browser':
            global open_browser
            open_browser = lambda: None
    
    # بررسی requirements
    if not check_requirements():
        print_colored("\n❌ Requirements not met. Please install missing components.", Colors.RED)
        return
    
    # بررسی ساختار پروژه
    if not check_project_structure():
        print_colored("\n❌ Project structure incomplete.", Colors.RED)
        return
    
    # سوال درباره نصب dependencies
    install_deps = input(f"\n{Colors.YELLOW}Install/Update dependencies? (y/n): {Colors.END}").lower().strip()
    if install_deps in ['y', 'yes', 'بله', '']:
        if not install_dependencies():
            print_colored("\n❌ Dependency installation failed.", Colors.RED)
            print_colored("💡 Try running: npm run setup", Colors.CYAN)
            return
    
    print_colored("\n🎯 Starting Persian Legal AI Training System...", Colors.GREEN + Colors.BOLD)
    print_colored("✨ Backend will be available at: http://localhost:8000", Colors.CYAN)
    print_colored("✨ Frontend will be available at: http://localhost:3000", Colors.CYAN)
    print_colored("✨ API Documentation at: http://localhost:8000/docs", Colors.CYAN)
    print_colored("\n💡 Available npm scripts:", Colors.BLUE)
    print_colored("   npm run setup    - Install all dependencies", Colors.WHITE)
    print_colored("   npm run backend  - Start backend only", Colors.WHITE)
    print_colored("   npm run frontend - Start frontend only", Colors.WHITE)
    print_colored("   npm run test     - Run tests", Colors.WHITE)
    print_colored("   npm run validate - Validate system", Colors.WHITE)
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