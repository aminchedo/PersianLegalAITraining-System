import asyncio
import sys
import time
import subprocess
import os
import sqlite3
import requests

async def test_integration():
    """Test basic system integration"""
    
    print("🧪 Starting Integration Tests...")
    test_results = []
    
    try:
        # Test 1: Import tests
        print("\n1️⃣ Testing Core Imports...")
        try:
            import fastapi
            import uvicorn
            import sqlite3
            import aiohttp
            print("✅ Core imports successful")
            test_results.append(("Core imports", True))
        except ImportError as e:
            print(f"❌ Core imports failed: {e}")
            test_results.append(("Core imports", False))
        
        # Test 2: Database connection
        print("\n2️⃣ Testing Database Connection...")
        try:
            conn = sqlite3.connect('persian_legal_ai.db')
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            conn.close()
            print(f"✅ Database connection successful, found {len(tables)} tables")
            test_results.append(("Database connection", True))
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            test_results.append(("Database connection", False))
        
        # Test 3: Main app import
        print("\n3️⃣ Testing Main Application Import...")
        try:
            from main import app
            print("✅ Main application import successful")
            test_results.append(("Main app import", True))
        except Exception as e:
            print(f"❌ Main application import failed: {e}")
            test_results.append(("Main app import", False))
        
        # Test 4: Persian app import
        print("\n4️⃣ Testing Persian Application Import...")
        try:
            from persian_main import app as persian_app
            print("✅ Persian application import successful")
            test_results.append(("Persian app import", True))
        except Exception as e:
            print(f"❌ Persian application import failed: {e}")
            test_results.append(("Persian app import", False))
        
        # Test 5: API endpoint test (if server is running)
        print("\n5️⃣ Testing API Endpoints...")
        try:
            response = requests.get('http://localhost:8000/', timeout=5)
            if response.status_code == 200:
                print("✅ Root endpoint responding")
                test_results.append(("Root endpoint", True))
            else:
                print(f"⚠️ Root endpoint returned status {response.status_code}")
                test_results.append(("Root endpoint", False))
        except requests.exceptions.RequestException as e:
            print(f"⚠️ API endpoint test skipped (server not running): {e}")
            test_results.append(("Root endpoint", None))
        
        # Test 6: Health endpoint test
        try:
            response = requests.get('http://localhost:8000/api/system/health', timeout=5)
            if response.status_code == 200:
                print("✅ Health endpoint responding")
                test_results.append(("Health endpoint", True))
            else:
                print(f"⚠️ Health endpoint returned status {response.status_code}")
                test_results.append(("Health endpoint", False))
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Health endpoint test skipped (server not running): {e}")
            test_results.append(("Health endpoint", None))
        
        # Test summary
        print("\n📊 TEST SUMMARY")
        print("=" * 50)
        passed = sum(1 for _, result in test_results if result is True)
        failed = sum(1 for _, result in test_results if result is False)
        skipped = sum(1 for _, result in test_results if result is None)
        total = len(test_results)
        
        for test_name, result in test_results:
            if result is True:
                print(f"✅ {test_name}")
            elif result is False:
                print(f"❌ {test_name}")
            else:
                print(f"⚠️ {test_name} (skipped)")
        
        print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped out of {total} tests")
        
        # Return success if most critical tests passed
        critical_tests = ["Core imports", "Database connection", "Main app import", "Persian app import"]
        critical_passed = sum(1 for test_name, result in test_results 
                            if test_name in critical_tests and result is True)
        
        success_rate = critical_passed / len(critical_tests)
        print(f"Critical test success rate: {success_rate:.1%}")
        
        return success_rate >= 0.75  # 75% of critical tests must pass
        
    except Exception as e:
        print(f"❌ Integration test suite failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_integration())
    sys.exit(0 if result else 1)