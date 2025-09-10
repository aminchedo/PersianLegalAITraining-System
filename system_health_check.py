#!/usr/bin/env python3
"""System Health Check for Persian Legal AI"""

import os
import sys
import sqlite3
import subprocess
from pathlib import Path

def check_virtual_environment():
    """Check if virtual environment is set up correctly"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return True, "Virtual environment active"
    elif Path('venv').exists():
        return True, "Virtual environment exists"
    else:
        return False, "Virtual environment not found"

def check_dependencies():
    """Check if critical dependencies are installed"""
    critical_packages = ['fastapi', 'uvicorn', 'sqlalchemy', 'aiosqlite', 'requests']
    missing = []
    
    for package in critical_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        return False, f"Missing packages: {', '.join(missing)}"
    else:
        return True, "All critical packages installed"

def check_database():
    """Check database status"""
    db_path = "persian_legal_ai.db"
    if not Path(db_path).exists():
        return False, "Database file not found"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()
        
        expected_tables = ['legal_documents', 'training_sessions']
        found_tables = [table[0] for table in tables]
        
        if all(table in found_tables for table in expected_tables):
            return True, f"Database OK with {len(tables)} tables"
        else:
            return False, f"Missing expected tables. Found: {found_tables}"
    except Exception as e:
        return False, f"Database error: {e}"

def check_configuration():
    """Check configuration files"""
    config_files = ['.env', 'config/database.py', 'config/logging.py']
    missing = []
    
    for file in config_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        return False, f"Missing config files: {', '.join(missing)}"
    else:
        return True, "All configuration files present"

def check_applications():
    """Check if main applications can be imported"""
    try:
        from main import app
        main_app_ok = True
    except Exception as e:
        main_app_ok = False
        main_error = str(e)
    
    try:
        from persian_main import app as persian_app
        persian_app_ok = True
    except Exception as e:
        persian_app_ok = False
        persian_error = str(e)
    
    if main_app_ok and persian_app_ok:
        return True, "Both applications import successfully"
    elif main_app_ok:
        return False, f"Persian app import failed: {persian_error}"
    elif persian_app_ok:
        return False, f"Main app import failed: {main_error}"
    else:
        return False, f"Both apps failed: main({main_error}), persian({persian_error})"

def check_directories():
    """Check if required directories exist"""
    required_dirs = ['logs', 'uploads', 'models', 'data', 'config', 'database']
    missing = []
    
    for directory in required_dirs:
        if not Path(directory).exists():
            missing.append(directory)
    
    if missing:
        return False, f"Missing directories: {', '.join(missing)}"
    else:
        return True, "All required directories present"

def main():
    """Run complete system health check"""
    print("🔍 PERSIAN LEGAL AI - SYSTEM HEALTH CHECK")
    print("=" * 60)
    
    checks = [
        ("Virtual Environment", check_virtual_environment),
        ("Dependencies", check_dependencies),
        ("Database", check_database),
        ("Configuration", check_configuration),
        ("Applications", check_applications),
        ("Directories", check_directories),
    ]
    
    total_score = 0
    max_score = len(checks)
    
    for check_name, check_func in checks:
        try:
            success, message = check_func()
            if success:
                print(f"✅ {check_name}: {message}")
                total_score += 1
            else:
                print(f"❌ {check_name}: {message}")
        except Exception as e:
            print(f"❌ {check_name}: Error during check - {e}")
    
    print("\n" + "=" * 60)
    health_percentage = (total_score / max_score) * 100
    print(f"📊 SYSTEM HEALTH SCORE: {total_score}/{max_score} ({health_percentage:.1f}%)")
    
    if health_percentage >= 90:
        print("🎉 SYSTEM STATUS: EXCELLENT")
        return 0
    elif health_percentage >= 75:
        print("✅ SYSTEM STATUS: GOOD")
        return 0
    elif health_percentage >= 50:
        print("⚠️  SYSTEM STATUS: FAIR - Some issues need attention")
        return 1
    else:
        print("❌ SYSTEM STATUS: POOR - Critical issues detected")
        return 2

if __name__ == "__main__":
    sys.exit(main())