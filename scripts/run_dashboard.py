#!/usr/bin/env python3
"""
Persian Legal AI Dashboard Launcher
Simple script to launch the Streamlit dashboard
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit dashboard"""
    
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Set environment variables
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    
    # Dashboard path
    dashboard_path = project_root / "interface" / "streamlit_dashboard.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        return 1
    
    print("🚀 Starting Persian Legal AI Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("⚖️ Persian Legal AI Training System - 2025 Edition")
    print("-" * 50)
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--browser.gatherUsageStats=false",
            "--theme.base=dark"
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)