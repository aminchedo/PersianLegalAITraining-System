#!/usr/bin/env python3
"""
Comprehensive System Test for Persian Legal AI Training System
Tests all major components and functionality
"""

import requests
import json
import sqlite3
import os
import sys
import time
from datetime import datetime

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_test(test_name, status, details=""):
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {test_name}")
    if details:
        print(f"   {details}")

def test_database():
    """Test database connectivity and structure"""
    print_header("DATABASE TESTS")
    
    try:
        # Check if database file exists
        db_path = 'persian_legal_ai.db'
        if not os.path.exists(db_path):
            print_test("Database File", False, "persian_legal_ai.db not found")
            return False
            
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        
        expected_tables = ['training_sessions', 'data_sources', 'legal_documents', 
                          'model_checkpoints', 'training_metrics', 'system_logs']
        
        all_tables_exist = all(table in tables for table in expected_tables)
        print_test("Database Tables", all_tables_exist, f"Found: {tables}")
        
        # Check data
        cursor.execute("SELECT COUNT(*) FROM legal_documents")
        doc_count = cursor.fetchone()[0]
        print_test("Database Data", doc_count > 0, f"Documents: {doc_count}")
        
        conn.close()
        return True
        
    except Exception as e:
        print_test("Database Connection", False, str(e))
        return False

def test_api_endpoints():
    """Test all major API endpoints"""
    print_header("API ENDPOINT TESTS")
    
    base_url = "http://localhost:8000"
    
    # Test endpoints
    endpoints = [
        ("/", "Root endpoint"),
        ("/docs", "API Documentation"),
        ("/api/system/health", "Health Check"),
        ("/api/documents/stats", "Document Stats"),
    ]
    
    results = []
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            success = response.status_code == 200
            print_test(description, success, f"Status: {response.status_code}")
            results.append(success)
        except Exception as e:
            print_test(description, False, str(e))
            results.append(False)
    
    return all(results)

def test_ai_classification():
    """Test AI classification functionality"""
    print_header("AI CLASSIFICATION TESTS")
    
    base_url = "http://localhost:8000"
    
    # Test cases
    test_cases = [
        {
            "text": "این یک متن حقوقی نمونه است",
            "description": "Persian Legal Text"
        },
        {
            "text": "قرارداد خرید و فروش ملک",
            "description": "Contract Text"
        },
        {
            "text": "مقررات جدید قانون مدنی",
            "description": "Regulation Text"
        }
    ]
    
    results = []
    for test_case in test_cases:
        try:
            response = requests.post(
                f"{base_url}/api/ai/classify",
                json={"text": test_case["text"]},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                confidence = data.get("confidence", 0)
                predicted_class = data.get("predicted_class", "unknown")
                
                success = confidence > 0.1  # Minimum confidence threshold
                print_test(
                    test_case["description"], 
                    success, 
                    f"Class: {predicted_class}, Confidence: {confidence:.3f}"
                )
                results.append(success)
            else:
                print_test(test_case["description"], False, f"HTTP {response.status_code}")
                results.append(False)
                
        except Exception as e:
            print_test(test_case["description"], False, str(e))
            results.append(False)
    
    return all(results)

def test_system_performance():
    """Test system performance and resources"""
    print_header("SYSTEM PERFORMANCE TESTS")
    
    try:
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        print_test("CPU Usage", cpu_percent < 80, f"{cpu_percent}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        print_test("Memory Usage", memory_percent < 80, f"{memory_percent}%")
        
        # Disk space
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        print_test("Disk Space", disk_percent < 90, f"{disk_percent:.1f}%")
        
        return True
        
    except ImportError:
        print_test("Performance Monitoring", False, "psutil not available")
        return False
    except Exception as e:
        print_test("Performance Monitoring", False, str(e))
        return False

def generate_report(db_ok, api_ok, ai_ok, perf_ok):
    """Generate comprehensive test report"""
    print_header("COMPREHENSIVE TEST REPORT")
    
    total_tests = 4
    passed_tests = sum([db_ok, api_ok, ai_ok, perf_ok])
    
    print(f"📊 Test Results: {passed_tests}/{total_tests} passed")
    print(f"🔍 Database Tests: {'✅ PASS' if db_ok else '❌ FAIL'}")
    print(f"🌐 API Tests: {'✅ PASS' if api_ok else '❌ FAIL'}")
    print(f"🤖 AI Tests: {'✅ PASS' if ai_ok else '❌ FAIL'}")
    print(f"⚡ Performance Tests: {'✅ PASS' if perf_ok else '❌ FAIL'}")
    
    overall_status = "✅ SYSTEM READY" if passed_tests >= 3 else "❌ SYSTEM NEEDS ATTENTION"
    print(f"\n🎯 Overall Status: {overall_status}")
    
    if passed_tests >= 3:
        print("\n🚀 System is ready for production deployment!")
        print("📋 Recommended next steps:")
        print("   1. Deploy to production service (Railway/Heroku)")
        print("   2. Update frontend API URLs")
        print("   3. Run end-to-end integration tests")
        print("   4. Set up monitoring and alerts")
    else:
        print("\n⚠️  System requires attention before production deployment")
        print("🔧 Please resolve failing tests before proceeding")
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_report_{timestamp}.json"
    
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "database_tests": db_ok,
        "api_tests": api_ok,
        "ai_tests": ai_ok,
        "performance_tests": perf_ok,
        "overall_status": "ready" if passed_tests >= 3 else "needs_attention"
    }
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return passed_tests >= 3

def main():
    """Run comprehensive system tests"""
    print_header("PERSIAN LEGAL AI SYSTEM - COMPREHENSIVE TEST")
    print(f"🕐 Test started at: {datetime.now()}")
    print(f"🐍 Python version: {sys.version}")
    
    # Run all tests
    db_ok = test_database()
    api_ok = test_api_endpoints()
    ai_ok = test_ai_classification()
    perf_ok = test_system_performance()
    
    # Generate report
    system_ready = generate_report(db_ok, api_ok, ai_ok, perf_ok)
    
    # Exit with appropriate code
    sys.exit(0 if system_ready else 1)

if __name__ == "__main__":
    main()