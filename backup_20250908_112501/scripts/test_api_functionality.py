#!/usr/bin/env python3
"""
Persian Legal AI API Functionality Testing Script
تست جامع عملکرد API سیستم هوش مصنوعی حقوقی فارسی

This script provides comprehensive testing of all API endpoints to verify
claimed functionality and measure actual system performance.
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys
import os

# Add project paths for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_functionality_test.log')
    ]
)
logger = logging.getLogger(__name__)

class PersianLegalAPITester:
    """Comprehensive API functionality tester for Persian Legal AI system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "base_url": base_url,
            "endpoints_tested": 0,
            "endpoints_passed": 0,
            "endpoints_failed": 0,
            "total_response_time": 0,
            "test_details": [],
            "persian_text_tests": [],
            "ai_model_tests": [],
            "database_tests": [],
            "performance_metrics": {}
        }
        
        # Persian test data
        self.persian_test_texts = [
            "این یک متن حقوقی فارسی است که برای تست سیستم طبقه‌بندی استفاده می‌شود.",
            "قانون اساسی جمهوری اسلامی ایران در مورد حقوق شهروندان مقررات مفصلی دارد.",
            "دادگاه عالی کشور رأی نهایی در مورد این پرونده را صادر کرده است.",
            "وکیل مدافع درخواست تجدیدنظر در این پرونده کیفری را ارائه داد.",
            "قرارداد خرید و فروش املاک باید طبق قوانین مدنی تنظیم شود."
        ]
        
        # Test document for upload
        self.test_document = {
            "title": "سند تستی برای بررسی عملکرد سیستم",
            "content": "این یک سند تستی است که برای بررسی عملکرد سیستم آپلود و پردازش اسناد حقوقی فارسی طراحی شده است. این سند شامل متن فارسی با محتوای حقوقی می‌باشد.",
            "category": "تست",
            "document_type": "سند آزمایشی",
            "source_url": "test_upload",
            "persian_date": "1403/01/01"
        }
    
    async def run_comprehensive_test(self):
        """Run comprehensive API functionality tests"""
        print("🔬 Persian Legal AI - Comprehensive API Testing")
        print("=" * 80)
        print(f"🌐 Base URL: {self.base_url}")
        print(f"🕐 Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        try:
            # Phase 1: System Health Tests
            await self._test_system_health()
            
            # Phase 2: Document Management Tests
            await self._test_document_management()
            
            # Phase 3: AI Classification Tests
            await self._test_ai_classification()
            
            # Phase 4: Training System Tests
            await self._test_training_system()
            
            # Phase 5: Performance Tests
            await self._test_performance()
            
            # Phase 6: Persian Text Handling Tests
            await self._test_persian_text_handling()
            
            # Generate final report
            self._generate_test_report()
            
        except Exception as e:
            logger.error(f"Critical test error: {e}")
            self.test_results["critical_error"] = str(e)
    
    async def _test_system_health(self):
        """Test system health and status endpoints"""
        print("\n🏥 Phase 1: System Health Tests")
        print("-" * 50)
        
        health_endpoints = [
            ("/", "Root endpoint"),
            ("/api/system/health", "Health check"),
            ("/api/system/status", "System status"),
            ("/docs", "API documentation"),
            ("/redoc", "ReDoc documentation")
        ]
        
        for endpoint, description in health_endpoints:
            await self._test_endpoint("GET", endpoint, description, expected_status=200)
    
    async def _test_document_management(self):
        """Test document management endpoints"""
        print("\n📄 Phase 2: Document Management Tests")
        print("-" * 50)
        
        # Test document statistics
        await self._test_endpoint("GET", "/api/documents/stats", "Document statistics")
        
        # Test database statistics
        await self._test_endpoint("GET", "/api/database/statistics", "Database statistics")
        
        # Test document search (empty query)
        search_data = {"query": "", "limit": 5}
        await self._test_endpoint("POST", "/api/documents/search", "Document search (empty)", 
                                 data=search_data)
        
        # Test document search with Persian query
        persian_search = {"query": "حقوقی", "limit": 10}
        await self._test_endpoint("POST", "/api/documents/search", "Persian document search", 
                                 data=persian_search)
        
        # Test document upload
        await self._test_endpoint("POST", "/api/documents/upload", "Document upload", 
                                 data=self.test_document)
        
        # Test document insertion (alternative endpoint)
        await self._test_endpoint("POST", "/api/documents/insert", "Document insertion", 
                                 data=self.test_document)
    
    async def _test_ai_classification(self):
        """Test AI classification endpoints"""
        print("\n🤖 Phase 3: AI Classification Tests")
        print("-" * 50)
        
        # Test model info
        await self._test_endpoint("GET", "/api/ai/model-info", "AI model information")
        
        # Test classification with different Persian texts
        for i, text in enumerate(self.persian_test_texts[:3], 1):
            classification_data = {
                "text": text,
                "return_probabilities": True
            }
            result = await self._test_endpoint("POST", "/api/classification/classify", 
                                             f"Classification test {i}", 
                                             data=classification_data)
            
            # Alternative endpoint
            await self._test_endpoint("POST", "/api/ai/classify", 
                                     f"AI Classification test {i}", 
                                     data=classification_data)
            
            if result and result.get("success"):
                self.test_results["ai_model_tests"].append({
                    "text_sample": text[:50] + "...",
                    "classification": result.get("response", {}).get("category", "N/A"),
                    "confidence": result.get("response", {}).get("confidence", 0),
                    "processing_time": result.get("response_time", 0)
                })
    
    async def _test_training_system(self):
        """Test training system endpoints"""
        print("\n🎓 Phase 4: Training System Tests")
        print("-" * 50)
        
        # Test training sessions list
        await self._test_endpoint("GET", "/api/training/sessions", "Training sessions list")
        
        # Test training configuration
        training_config = {
            "model_type": "dora",
            "epochs": 1,
            "learning_rate": 0.0002,
            "batch_size": 8,
            "use_dora": True,
            "notes": "API functionality test"
        }
        
        # Start a test training session
        result = await self._test_endpoint("POST", "/api/training/start", "Start training session", 
                                          data=training_config)
        
        # If training started, check status
        if result and result.get("success"):
            session_id = result.get("response", {}).get("session_id")
            if session_id:
                await asyncio.sleep(2)  # Wait a moment
                await self._test_endpoint("GET", f"/api/training/sessions/{session_id}/status", 
                                        "Training session status")
    
    async def _test_performance(self):
        """Test system performance endpoints"""
        print("\n⚡ Phase 5: Performance Tests")
        print("-" * 50)
        
        # Test ping endpoint for latency
        ping_times = []
        for i in range(5):
            result = await self._test_endpoint("GET", "/api/test/ping", f"Ping test {i+1}")
            if result and result.get("success"):
                ping_times.append(result.get("response_time", 0))
        
        if ping_times:
            avg_ping = sum(ping_times) / len(ping_times)
            self.test_results["performance_metrics"]["average_ping"] = avg_ping
            print(f"   📊 Average ping: {avg_ping:.2f}ms")
        
        # Test system metrics
        await self._test_endpoint("GET", "/api/system/metrics", "System metrics")
        
        # Test performance summary
        await self._test_endpoint("GET", "/api/system/performance-summary", "Performance summary")
    
    async def _test_persian_text_handling(self):
        """Test Persian text processing capabilities"""
        print("\n📝 Phase 6: Persian Text Handling Tests")
        print("-" * 50)
        
        for i, text in enumerate(self.persian_test_texts, 1):
            # Test classification
            classification_data = {"text": text}
            result = await self._test_endpoint("POST", "/api/classification/classify", 
                                             f"Persian text {i} classification", 
                                             data=classification_data)
            
            if result and result.get("success"):
                response = result.get("response", {})
                self.test_results["persian_text_tests"].append({
                    "test_number": i,
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "classification": response.get("category", "N/A"),
                    "confidence": response.get("confidence", 0),
                    "processing_time": result.get("response_time", 0)
                })
    
    async def _test_endpoint(self, method: str, endpoint: str, description: str, 
                           data: Optional[Dict] = None, expected_status: int = 200) -> Optional[Dict]:
        """Test a specific API endpoint"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if method.upper() == "GET":
                    async with session.get(url) as response:
                        response_data = await response.json() if response.content_type == 'application/json' else await response.text()
                        status = response.status
                elif method.upper() == "POST":
                    async with session.post(url, json=data) as response:
                        response_data = await response.json() if response.content_type == 'application/json' else await response.text()
                        status = response.status
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Determine success
                success = status == expected_status
                status_icon = "✅" if success else "❌"
                
                print(f"   {status_icon} {description}: HTTP {status} ({response_time:.1f}ms)")
                
                # Record test result
                test_detail = {
                    "endpoint": endpoint,
                    "method": method,
                    "description": description,
                    "status_code": status,
                    "expected_status": expected_status,
                    "success": success,
                    "response_time": response_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                if not success:
                    test_detail["error"] = response_data if isinstance(response_data, str) else str(response_data)
                
                self.test_results["test_details"].append(test_detail)
                self.test_results["endpoints_tested"] += 1
                self.test_results["total_response_time"] += response_time
                
                if success:
                    self.test_results["endpoints_passed"] += 1
                    return {
                        "success": True,
                        "response": response_data,
                        "response_time": response_time
                    }
                else:
                    self.test_results["endpoints_failed"] += 1
                    return {
                        "success": False,
                        "error": response_data,
                        "response_time": response_time
                    }
                
        except asyncio.TimeoutError:
            print(f"   ⏱️  {description}: TIMEOUT")
            self.test_results["test_details"].append({
                "endpoint": endpoint,
                "method": method,
                "description": description,
                "success": False,
                "error": "Timeout",
                "response_time": 30000,  # 30 seconds
                "timestamp": datetime.now().isoformat()
            })
            self.test_results["endpoints_tested"] += 1
            self.test_results["endpoints_failed"] += 1
            return None
            
        except Exception as e:
            print(f"   ❌ {description}: ERROR - {str(e)}")
            self.test_results["test_details"].append({
                "endpoint": endpoint,
                "method": method,
                "description": description,
                "success": False,
                "error": str(e),
                "response_time": (time.time() - start_time) * 1000,
                "timestamp": datetime.now().isoformat()
            })
            self.test_results["endpoints_tested"] += 1
            self.test_results["endpoints_failed"] += 1
            return None
    
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📋 API Functionality Test Report")
        print("=" * 80)
        
        # Calculate metrics
        total_endpoints = self.test_results["endpoints_tested"]
        passed_endpoints = self.test_results["endpoints_passed"]
        failed_endpoints = self.test_results["endpoints_failed"]
        success_rate = (passed_endpoints / total_endpoints * 100) if total_endpoints > 0 else 0
        avg_response_time = (self.test_results["total_response_time"] / total_endpoints) if total_endpoints > 0 else 0
        
        print(f"📊 Test Summary:")
        print(f"   Total Endpoints Tested: {total_endpoints}")
        print(f"   ✅ Passed: {passed_endpoints}")
        print(f"   ❌ Failed: {failed_endpoints}")
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        print(f"   ⚡ Average Response Time: {avg_response_time:.1f}ms")
        
        # Persian text processing results
        if self.test_results["persian_text_tests"]:
            print(f"\n📝 Persian Text Processing Results:")
            for test in self.test_results["persian_text_tests"]:
                print(f"   Text {test['test_number']}: {test['classification']} "
                      f"(confidence: {test['confidence']:.2f}, {test['processing_time']:.1f}ms)")
        
        # AI model performance
        if self.test_results["ai_model_tests"]:
            print(f"\n🤖 AI Model Performance:")
            for test in self.test_results["ai_model_tests"]:
                print(f"   Classification: {test['classification']} "
                      f"(confidence: {test['confidence']:.2f}, {test['processing_time']:.1f}ms)")
        
        # Performance metrics
        if self.test_results["performance_metrics"]:
            print(f"\n⚡ Performance Metrics:")
            for metric, value in self.test_results["performance_metrics"].items():
                print(f"   {metric.replace('_', ' ').title()}: {value:.2f}ms")
        
        # Failed endpoints details
        failed_tests = [t for t in self.test_results["test_details"] if not t["success"]]
        if failed_tests:
            print(f"\n❌ Failed Endpoints Details:")
            for test in failed_tests:
                print(f"   {test['method']} {test['endpoint']}: {test.get('error', 'Unknown error')}")
        
        # Overall system assessment
        print(f"\n🎯 System Assessment:")
        if success_rate >= 90:
            print(f"   🎉 EXCELLENT: System is production ready ({success_rate:.1f}% success rate)")
        elif success_rate >= 80:
            print(f"   ✅ GOOD: System is mostly functional ({success_rate:.1f}% success rate)")
        elif success_rate >= 60:
            print(f"   ⚠️  FAIR: System has some issues ({success_rate:.1f}% success rate)")
        else:
            print(f"   ❌ POOR: System needs significant work ({success_rate:.1f}% success rate)")
        
        # Save detailed report to file
        report_file = Path("api_test_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        print("=" * 80)

async def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Persian Legal AI API Functionality Tester")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL for API testing (default: http://localhost:8000)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    tester = PersianLegalAPITester(args.url)
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())