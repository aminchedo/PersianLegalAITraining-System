#!/usr/bin/env python3
"""
Start Persian Legal AI Training System
شروع سیستم آموزش هوش مصنوعی حقوقی فارسی
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import torch
        import transformers
        import fastapi
        import uvicorn
        import psutil
        import hazm
        logger.info("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        return False

def run_backend():
    """Run the backend server"""
    try:
        logger.info("🚀 Starting backend server...")
        
        # Change to backend directory
        backend_dir = Path(__file__).parent / "backend"
        os.chdir(backend_dir)
        
        # Run backend server
        subprocess.run([
            sys.executable, "main.py"
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Backend server failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to start backend: {e}")
        return False

def run_frontend():
    """Run the frontend development server"""
    try:
        logger.info("🎨 Starting frontend server...")
        
        # Change to frontend directory
        frontend_dir = Path(__file__).parent / "frontend"
        os.chdir(frontend_dir)
        
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            logger.info("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], check=True)
        
        # Run frontend development server
        subprocess.run([
            "npm", "run", "dev"
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Frontend server failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to start frontend: {e}")
        return False

def run_system_test():
    """Run the comprehensive system test"""
    try:
        logger.info("🧪 Running comprehensive system test...")
        
        # Change to project root
        project_root = Path(__file__).parent
        os.chdir(project_root)
        
        # Run test script
        result = subprocess.run([
            sys.executable, "run_full_system_test.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ System test passed!")
            return True
        else:
            logger.error("❌ System test failed!")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to run system test: {e}")
        return False

def main():
    """Main function"""
    try:
        logger.info("🌟 Persian Legal AI Training System")
        logger.info("=" * 50)
        
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Get user choice
        print("\nChoose an option:")
        print("1. Run Backend Server")
        print("2. Run Frontend Server")
        print("3. Run System Test")
        print("4. Run Both Backend and Frontend")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            run_backend()
        elif choice == "2":
            run_frontend()
        elif choice == "3":
            run_system_test()
        elif choice == "4":
            logger.info("🚀 Starting both backend and frontend...")
            logger.info("Backend will run on http://localhost:8000")
            logger.info("Frontend will run on http://localhost:3000")
            logger.info("Press Ctrl+C to stop both servers")
            
            # Start backend in background
            backend_process = subprocess.Popen([
                sys.executable, "start_system.py", "--backend-only"
            ])
            
            # Wait a bit for backend to start
            time.sleep(3)
            
            # Start frontend
            try:
                run_frontend()
            except KeyboardInterrupt:
                logger.info("🛑 Stopping servers...")
                backend_process.terminate()
                backend_process.wait()
        elif choice == "5":
            logger.info("👋 Goodbye!")
            sys.exit(0)
        else:
            logger.error("❌ Invalid choice")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopped by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--backend-only":
        run_backend()
    else:
        main()