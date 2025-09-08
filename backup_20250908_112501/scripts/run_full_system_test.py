#!/usr/bin/env python3
"""
Persian Legal AI - Comprehensive End-to-End System Integration Test
تست یکپارچه جامع سیستم هوش مصنوعی حقوقی فارسی
"""

import sys
import os
import time
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersianLegalAISystemIntegrationTester:
    """Comprehensive end-to-end system integration tester"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.backend_port = 8000
        self.frontend_port = 3000
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "system_startup": {},
            "component_integration": {},
            "user_workflows": {},
            "data_flow_tests": {},
            "final_functionality_score": 0
        }
        
    async def run_comprehensive_integration_test(self):
        """Run comprehensive end-to-end system integration tests"""
        print("🔬 Persian Legal AI - End-to-End System Integration Test")
        print("=" * 80)
        
        try:
            await self._test_system_startup()
            await self._test_component_integration()
            await self._test_user_workflows()
            await self._test_data_flow()
            self._calculate_final_score()
            self._generate_integration_report()
        except Exception as e:
            logger.error(f"Critical test error: {e}")
            self.test_results["critical_error"] = str(e)
    
    async def _test_system_startup(self):
        """Test system startup and health"""
        print("\n🚀 Phase 1: System Startup and Health Check")
        print("-" * 50)
        
        startup_results = {
            "backend_accessible": False,
            "frontend_accessible": False,
            "system_health": {}
        }
        
        # Test backend
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f'http://localhost:{self.backend_port}/api/system/health') as response:
                    if response.status == 200:
                        health_data = await response.json()
                        startup_results["backend_accessible"] = True
                        startup_results["system_health"] = health_data
                        print("✅ Backend accessible and healthy")
                    else:
                        print(f"⚠️  Backend unhealthy (HTTP {response.status})")
        except Exception as e:
            print(f"❌ Backend not accessible: {e}")
        
        # Test frontend
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f'http://localhost:{self.frontend_port}') as response:
                    if response.status == 200:
                        startup_results["frontend_accessible"] = True
                        print("✅ Frontend accessible")
                    else:
                        print(f"⚠️  Frontend returns HTTP {response.status}")
        except Exception as e:
            print(f"❌ Frontend not accessible: {e}")
        
        self.test_results["system_startup"] = startup_results
    
    async def _test_component_integration(self):
        """Test component integration"""
        print("\n🔗 Phase 2: Component Integration Tests")
        print("-" * 50)
        
        integration_results = {
            "database_backend_integration": False,
            "ai_backend_integration": False
        }
        
        # Test database integration
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://localhost:{self.backend_port}/api/documents/stats') as response:
                    if response.status == 200:
                        stats = await response.json()
                        integration_results["database_backend_integration"] = True
                        print(f"✅ Database integration working")
                    else:
                        print(f"❌ Database integration failed - HTTP {response.status}")
        except Exception as e:
            print(f"❌ Database integration test failed: {e}")
        
        # Test AI integration
        try:
            payload = {"text": "این یک متن تستی است.", "return_probabilities": True}
            async with aiohttp.ClientSession() as session:
                async with session.post(f'http://localhost:{self.backend_port}/api/ai/classify', json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        integration_results["ai_backend_integration"] = True
                        print("✅ AI integration working")
                    else:
                        print(f"❌ AI integration failed - HTTP {response.status}")
        except Exception as e:
            print(f"❌ AI integration test failed: {e}")
        
        self.test_results["component_integration"] = integration_results
    
    async def _test_user_workflows(self):
        """Test user workflows"""
        print("\n👤 Phase 3: User Workflow Tests")
        print("-" * 50)
        
        workflow_results = {"successful_workflows": 0, "total_workflows": 2}
        
        # Test document upload workflow
        test_doc = {
            "title": "تست سند حقوقی",
            "content": "این یک سند تستی برای بررسی عملکرد سیستم است.",
            "category": "تست"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f'http://localhost:{self.backend_port}/api/documents/upload', json=test_doc) as response:
                    if response.status == 200:
                        workflow_results["successful_workflows"] += 1
                        print("✅ Document upload workflow successful")
                    else:
                        print("❌ Document upload workflow failed")
        except Exception as e:
            print(f"❌ Document upload workflow error: {e}")
        
        # Test search workflow
        try:
            search_payload = {"query": "تست", "limit": 5}
            async with aiohttp.ClientSession() as session:
                async with session.post(f'http://localhost:{self.backend_port}/api/documents/search', json=search_payload) as response:
                    if response.status == 200:
                        workflow_results["successful_workflows"] += 1
                        print("✅ Search workflow successful")
                    else:
                        print("❌ Search workflow failed")
        except Exception as e:
            print(f"❌ Search workflow error: {e}")
        
        self.test_results["user_workflows"] = workflow_results
    
    async def _test_data_flow(self):
        """Test data flow"""
        print("\n🌊 Phase 4: Data Flow Tests")
        print("-" * 50)
        
        data_flow_results = {"end_to_end_flow": False}
        
        test_document = {
            "title": "تست جریان داده",
            "content": "این سند برای تست جریان کامل داده طراحی شده است.",
            "category": "تست"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Upload document
                async with session.post(f'http://localhost:{self.backend_port}/api/documents/upload', json=test_document) as response:
                    if response.status == 200:
                        # Classify document
                        classification_payload = {"text": test_document["content"]}
                        async with session.post(f'http://localhost:{self.backend_port}/api/ai/classify', json=classification_payload) as class_response:
                            if class_response.status == 200:
                                data_flow_results["end_to_end_flow"] = True
                                print("✅ End-to-end data flow successful")
                            else:
                                print("❌ Classification failed")
                    else:
                        print("❌ Document upload failed")
        except Exception as e:
            print(f"❌ Data flow test failed: {e}")
        
        self.test_results["data_flow_tests"] = data_flow_results
    
    def _calculate_final_score(self):
        """Calculate final functionality score"""
        scores = []
        
        # System startup (30%)
        startup = self.test_results["system_startup"]
        startup_score = sum([
            startup.get("backend_accessible", False),
            startup.get("frontend_accessible", False)
        ]) / 2 * 100
        scores.append(("System Startup", startup_score, 0.30))
        
        # Component integration (30%)
        integration = self.test_results["component_integration"]
        integration_score = sum([
            integration.get("database_backend_integration", False),
            integration.get("ai_backend_integration", False)
        ]) / 2 * 100
        scores.append(("Component Integration", integration_score, 0.30))
        
        # User workflows (25%)
        workflows = self.test_results["user_workflows"]
        workflow_score = (workflows.get("successful_workflows", 0) / 
                         workflows.get("total_workflows", 1)) * 100
        scores.append(("User Workflows", workflow_score, 0.25))
        
        # Data flow (15%)
        data_flow = self.test_results["data_flow_tests"]
        data_flow_score = 100 if data_flow.get("end_to_end_flow") else 0
        scores.append(("Data Flow", data_flow_score, 0.15))
        
        final_score = sum(score * weight for _, score, weight in scores)
        self.test_results["final_functionality_score"] = final_score
        self.test_results["score_breakdown"] = [
            {"component": name, "score": score, "weight": weight}
            for name, score, weight in scores
        ]
        
        return final_score
    
    def _generate_integration_report(self):
        """Generate integration test report"""
        print("\n�� End-to-End System Integration Test Report")
        print("=" * 80)
        
        final_score = self.test_results["final_functionality_score"]
        print(f"🎯 Final System Functionality Score: {final_score:.1f}/100")
        
        print("\n📊 Score Breakdown:")
        for breakdown in self.test_results["score_breakdown"]:
            print(f"   {breakdown['component']}: {breakdown['score']:.1f}/100 "
                  f"(weight: {breakdown['weight']*100:.0f}%)")
        
        # System assessment
        print(f"\n🎯 Overall System Assessment:")
        if final_score >= 85:
            print(f"   🎉 EXCELLENT: System is production ready ({final_score:.1f}/100)")
        elif final_score >= 70:
            print(f"   ✅ GOOD: System is mostly functional ({final_score:.1f}/100)")
        elif final_score >= 50:
            print(f"   ⚠️  FAIR: System has functional gaps ({final_score:.1f}/100)")
        else:
            print(f"   ❌ POOR: System needs significant work ({final_score:.1f}/100)")
        
        # Save report
        report_file = Path("system_integration_test_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        print("=" * 80)

async def main():
    tester = PersianLegalAISystemIntegrationTester()
    await tester.run_comprehensive_integration_test()

if __name__ == "__main__":
    asyncio.run(main())
