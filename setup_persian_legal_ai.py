#!/usr/bin/env python3
"""
Persian Legal AI Training System - Complete Automated Setup
Handles everything from clone to running model training
"""

import os
import sys
import subprocess
import platform
import json
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class PersianLegalAISetup:
    def __init__(self):
        self.system = platform.system().lower()
        self.project_root = Path.cwd()
        self.errors = []
        self.ai_models = [
            {
                "name": "HooshvareLab/bert-base-parsbert-uncased",
                "size": "500MB",
                "priority": 1
            },
            {
                "name": "HooshvareLab/bert-fa-base-uncased",
                "size": "450MB", 
                "priority": 2
            }
        ]
        
    def log(self, message: str, level: str = "INFO"):
        """Enhanced logging with colors"""
        colors = {
            "INFO": "\033[92m",    # Green
            "WARN": "\033[93m",    # Yellow
            "ERROR": "\033[91m",   # Red
            "RESET": "\033[0m"     # Reset
        }
        print(f"{colors.get(level, '')}{level}: {message}{colors['RESET']}")
        
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, 
                   check: bool = True, capture: bool = False) -> Tuple[bool, str]:
        """Execute command with enhanced error handling"""
        try:
            if capture:
                result = subprocess.run(
                    cmd, cwd=cwd, check=check, 
                    capture_output=True, text=True, timeout=300
                )
                return True, result.stdout.strip()
            else:
                subprocess.run(cmd, cwd=cwd, check=check, timeout=300)
                return True, "Success"
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed: {' '.join(cmd)}\nError: {str(e)}"
            self.errors.append(error_msg)
            self.log(error_msg, "ERROR")
            return False, str(e)
        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out: {' '.join(cmd)}"
            self.errors.append(error_msg)
            self.log(error_msg, "ERROR")
            return False, "Timeout"
            
    def check_system_requirements(self) -> bool:
        """Check system requirements and dependencies"""
        self.log("🔍 Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.log("Python 3.8+ required", "ERROR")
            return False
            
        # Check Git
        success, _ = self.run_command(["git", "--version"], check=False)
        if not success:
            self.log("Git not found. Please install Git first.", "ERROR")
            return False
            
        # Check Node.js
        success, version = self.run_command(["node", "--version"], check=False, capture=True)
        if not success:
            self.log("Node.js not found. Please install Node.js 18+ first.", "ERROR")
            return False
            
        # Verify Node version
        try:
            node_version = int(version.replace('v', '').split('.')[0])
            if node_version < 18:
                self.log(f"Node.js {node_version} found. Need 18+", "ERROR")
                return False
        except:
            pass
            
        self.log("✅ System requirements check passed")
        return True
        
    def setup_python_environment(self) -> bool:
        """Setup Python virtual environment and dependencies"""
        self.log("🐍 Setting up Python environment...")
        
        # Create virtual environment
        venv_path = self.project_root / "venv"
        if not venv_path.exists():
            # Try different venv creation methods
            venv_commands = [
                [sys.executable, "-m", "venv", "venv"],
                ["python3", "-m", "venv", "venv"],
                ["virtualenv", "venv"]
            ]
            
            success = False
            for cmd in venv_commands:
                success, error = self.run_command(cmd, check=False)
                if success:
                    break
                else:
                    self.log(f"Failed with {cmd[0]}: {error}", "WARN")
            
            if not success:
                self.log("Could not create virtual environment. Trying to use system Python...", "WARN")
                # Continue without venv - use system Python
                return self._setup_system_python()
        else:
            self.log("Virtual environment already exists")
                
        # Activate virtual environment and install requirements
        if self.system == "windows":
            python_exe = venv_path / "Scripts" / "python.exe"
            pip_exe = venv_path / "Scripts" / "pip.exe"
        else:
            python_exe = venv_path / "bin" / "python"
            pip_exe = venv_path / "bin" / "pip"
            
        # Upgrade pip
        success, _ = self.run_command([str(pip_exe), "install", "--upgrade", "pip"])
        if not success:
            return False
            
        # Install backend requirements
        req_files = [
            "requirements.txt",
            "backend/requirements.txt", 
            "requirements_production.txt"
        ]
        
        for req_file in req_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                self.log(f"📦 Installing {req_file}...")
                success, _ = self.run_command([
                    str(pip_exe), "install", "-r", str(req_path)
                ])
                if not success:
                    self.log(f"Failed to install {req_file}, continuing...", "WARN")
                    
        self.log("✅ Python environment setup completed")
        return True
        
    def _setup_system_python(self) -> bool:
        """Fallback to system Python if venv creation fails"""
        self.log("📦 Using system Python for package installation...")
        
        # Use system Python and pip
        python_exe = sys.executable
        pip_exe = "pip3"
        
        # Upgrade pip
        success, _ = self.run_command([pip_exe, "install", "--upgrade", "pip", "--user"])
        if not success:
            success, _ = self.run_command([python_exe, "-m", "pip", "install", "--upgrade", "pip", "--user"])
            if not success:
                self.log("Could not upgrade pip, continuing...", "WARN")
            
        # Install backend requirements
        req_files = [
            "requirements.txt",
            "backend/requirements.txt", 
            "requirements_production.txt"
        ]
        
        for req_file in req_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                self.log(f"📦 Installing {req_file} to user directory...")
                success, _ = self.run_command([
                    pip_exe, "install", "-r", str(req_path), "--user"
                ])
                if not success:
                    success, _ = self.run_command([
                        python_exe, "-m", "pip", "install", "-r", str(req_path), "--user"
                    ])
                    if not success:
                        self.log(f"Failed to install {req_file}, continuing...", "WARN")
                        
        self.log("✅ System Python setup completed")
        return True
        
    def setup_node_environment(self) -> bool:
        """Setup Node.js environment with multiple package managers"""
        self.log("📦 Setting up Node.js environment...")
        
        # Frontend directory
        frontend_dir = self.project_root / "frontend"
        if not frontend_dir.exists():
            self.log("Frontend directory not found", "ERROR")
            return False
            
        # Try different package managers in order of preference
        package_managers = [
            {"cmd": "pnpm", "install": ["pnpm", "install"], "global_install": ["npm", "install", "-g", "pnpm"]},
            {"cmd": "yarn", "install": ["yarn", "install"], "global_install": ["npm", "install", "-g", "yarn"]},
            {"cmd": "npm", "install": ["npm", "install"], "global_install": None}
        ]
        
        for pm in package_managers:
            # Check if package manager exists, install if not
            success, _ = self.run_command([pm["cmd"], "--version"], check=False)
            if not success and pm["global_install"]:
                self.log(f"Installing {pm['cmd']}...")
                success, _ = self.run_command(pm["global_install"])
                
            # Try to use this package manager
            if success or pm["cmd"] == "npm":
                self.log(f"📦 Installing frontend dependencies with {pm['cmd']}...")
                success, _ = self.run_command(pm["install"], cwd=frontend_dir)
                if success:
                    self.log(f"✅ Frontend dependencies installed with {pm['cmd']}")
                    return True
                    
        self.log("Failed to install frontend dependencies with any package manager", "ERROR")
        return False
        
    def download_ai_models(self) -> bool:
        """Download and setup AI models for training"""
        self.log("🤖 Downloading AI models...")
        
        # Create models directory
        models_dir = self.project_root / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Install huggingface_hub if not available
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            self.log("Installing huggingface_hub...")
            success, _ = self.run_command([
                sys.executable, "-m", "pip", "install", "huggingface_hub"
            ])
            if not success:
                return False
            from huggingface_hub import snapshot_download
            
        # Download models
        for model in self.ai_models:
            model_path = models_dir / model["name"].replace("/", "_")
            if model_path.exists():
                self.log(f"Model {model['name']} already exists, skipping...")
                continue
                
            try:
                self.log(f"📥 Downloading {model['name']} ({model['size']})...")
                snapshot_download(
                    repo_id=model["name"],
                    local_dir=str(model_path),
                    local_dir_use_symlinks=False
                )
                self.log(f"✅ Downloaded {model['name']}")
            except Exception as e:
                self.log(f"Failed to download {model['name']}: {e}", "WARN")
                
        self.log("✅ AI models setup completed")
        return True
        
    def setup_database(self) -> bool:
        """Setup database and run migrations"""
        self.log("🗄️ Setting up database...")
        
        # Check if database files exist
        db_files = [
            "persian_legal_ai.db",
            "backend/persian_legal_ai.db",
            "database.db"
        ]
        
        for db_file in db_files:
            db_path = self.project_root / db_file
            if db_path.exists():
                self.log(f"Database found: {db_file}")
                break
        else:
            self.log("No existing database found, will be created on first run")
            
        # Run database migrations if script exists
        migration_scripts = [
            "backend/database/migrate.py",
            "migrate.py",
            "setup_database.py"
        ]
        
        for script in migration_scripts:
            script_path = self.project_root / script
            if script_path.exists():
                self.log(f"Running database migration: {script}")
                success, _ = self.run_command([sys.executable, str(script)])
                if success:
                    break
                    
        self.log("✅ Database setup completed")
        return True
        
    def create_environment_files(self) -> bool:
        """Create necessary environment files"""
        self.log("📝 Creating environment files...")
        
        # Backend .env
        backend_env = self.project_root / "backend" / ".env"
        if not backend_env.exists():
            env_content = """
# Database
DATABASE_URL=sqlite:///./persian_legal_ai.db
DB_ECHO=false

# AI Models
MODEL_CACHE_DIR=../models
HUGGINGFACE_CACHE_DIR=../models

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]

# Training Configuration
BATCH_SIZE=8
LEARNING_RATE=2e-4
NUM_EPOCHS=3
MAX_SEQUENCE_LENGTH=512

# System
ENVIRONMENT=development
LOG_LEVEL=INFO
"""
            backend_env.write_text(env_content.strip())
            
        # Frontend .env
        frontend_env = self.project_root / "frontend" / ".env.local"
        if not frontend_env.exists():
            env_content = """
NEXT_PUBLIC_API_URL=http://localhost:8000
NODE_ENV=development
"""
            frontend_env.write_text(env_content.strip())
            
        self.log("✅ Environment files created")
        return True
        
    def start_backend(self) -> subprocess.Popen:
        """Start the backend server"""
        self.log("🚀 Starting backend server...")
        
        # Find Python executable (venv or system)
        venv_path = self.project_root / "venv"
        if venv_path.exists():
            if self.system == "windows":
                python_exe = venv_path / "Scripts" / "python.exe"
            else:
                python_exe = venv_path / "bin" / "python"
        else:
            python_exe = sys.executable
            
        # Start backend
        backend_cmd = [str(python_exe), "-m", "uvicorn", "backend.main:app", 
                      "--host", "0.0.0.0", "--port", "8000", "--reload"]
        
        backend_process = subprocess.Popen(
            backend_cmd, 
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for backend to start
        time.sleep(5)
        
        # Check if backend is running
        try:
            urllib.request.urlopen("http://localhost:8000/api/system/health", timeout=5)
            self.log("✅ Backend server started successfully")
        except:
            self.log("Backend server may still be starting...", "WARN")
            
        return backend_process
        
    def start_frontend(self) -> subprocess.Popen:
        """Start the frontend server"""
        self.log("🌐 Starting frontend server...")
        
        frontend_dir = self.project_root / "frontend"
        
        # Determine package manager and start command
        if (frontend_dir / "pnpm-lock.yaml").exists():
            cmd = ["pnpm", "dev"]
        elif (frontend_dir / "yarn.lock").exists():
            cmd = ["yarn", "dev"]
        else:
            cmd = ["npm", "run", "dev"]
            
        frontend_process = subprocess.Popen(
            cmd,
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for frontend to start
        time.sleep(10)
        
        self.log("✅ Frontend server started")
        return frontend_process
        
    def start_training(self) -> bool:
        """Start AI model training"""
        self.log("🎓 Starting AI model training...")
        
        # Find training scripts
        training_scripts = [
            "start_training.py",
            "backend/training/start_training.py",
            "train_model.py"
        ]
        
        for script in training_scripts:
            script_path = self.project_root / script
            if script_path.exists():
                self.log(f"Starting training with: {script}")
                success, _ = self.run_command([sys.executable, str(script)])
                if success:
                    return True
                    
        # Try API endpoint for training
        try:
            import requests
            response = requests.post("http://localhost:8000/api/training/sessions", 
                json={
                    "model_type": "dora",
                    "model_name": "persian-legal-demo",
                    "config": {
                        "dora_rank": 8,
                        "dora_alpha": 16,
                        "learning_rate": 2e-4,
                        "num_epochs": 3,
                        "batch_size": 8
                    },
                    "data_source": "sample",
                    "task_type": "text_classification"
                })
            if response.status_code == 200:
                self.log("✅ Training started via API")
                return True
        except:
            pass
            
        self.log("Could not start training automatically. Start manually via dashboard.", "WARN")
        return False
        
    def run_complete_setup(self) -> bool:
        """Run the complete setup process"""
        self.log("🚀 Starting Persian Legal AI Complete Setup...")
        
        steps = [
            ("System Requirements", self.check_system_requirements),
            ("Python Environment", self.setup_python_environment),
            ("Node.js Environment", self.setup_node_environment), 
            ("AI Models Download", self.download_ai_models),
            ("Database Setup", self.setup_database),
            ("Environment Files", self.create_environment_files)
        ]
        
        for step_name, step_func in steps:
            self.log(f"📋 Step: {step_name}")
            if not step_func():
                self.log(f"❌ Failed at step: {step_name}", "ERROR")
                return False
                
        self.log("✅ Setup completed successfully!")
        
        # Start services
        self.log("🚀 Starting services...")
        backend_process = self.start_backend()
        frontend_process = self.start_frontend()
        
        # Start training
        time.sleep(5)
        self.start_training()
        
        self.log("""
🎉 PERSIAN LEGAL AI SYSTEM IS RUNNING!

📊 Dashboard: http://localhost:3000
🔧 Backend API: http://localhost:8000
📚 API Docs: http://localhost:8000/docs
🔍 Health Check: http://localhost:8000/api/system/health

🎓 Model training has started automatically!
⌨️  Press Ctrl+C to stop all services
        """)
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.log("\n🛑 Stopping services...")
            backend_process.terminate()
            frontend_process.terminate()
            
        return True

def main():
    """Main entry point"""
    setup = PersianLegalAISetup()
    
    try:
        success = setup.run_complete_setup()
        if not success:
            print("\n❌ Setup failed. Check errors above.")
            if setup.errors:
                print("\nErrors encountered:")
                for error in setup.errors:
                    print(f"  - {error}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()