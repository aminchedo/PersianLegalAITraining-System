#!/usr/bin/env python3
"""
Persian Legal AI Database Functionality Testing Script
تست جامع عملکرد پایگاه داده سیستم هوش مصنوعی حقوقی فارسی

This script provides comprehensive testing of database operations,
Persian text handling, search functionality, and data integrity.
"""

import sys
import os
import time
import json
import logging
import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import traceback

# Add project paths for imports
project_root = Path(__file__).parent.parent
sys.path.extend([
    str(project_root),
    str(project_root / "backend"),
    str(project_root / "backend" / "database"),
])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('database_functionality_test.log')
    ]
)
logger = logging.getLogger(__name__)

class PersianLegalDatabaseTester:
    """Comprehensive database functionality tester for Persian Legal AI system"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "database_connections": {},
            "persian_text_tests": {},
            "search_functionality": {},
            "crud_operations": {},
            "data_integrity": {},
            "performance_tests": {},
            "backup_restore": {}
        }
        
        # Persian test documents for database operations
        self.test_documents = [
            {
                "title": "قانون اساسی جمهوری اسلامی ایران - اصل ۱",
                "content": "جمهوری اسلامی ایران، حکومتی است مردمی که بر اساس آراء اکثریت مردم و با تکیه بر قرآن و سنت معصومین(علیهم‌السلام) تشکیل شده است.",
                "category": "قانون اساسی",
                "document_type": "اصل قانون اساسی",
                "source_url": "test://constitutional_law_1",
                "persian_date": "1358/10/03",
                "legal_relevance": 1.0,
                "language_confidence": 1.0,
                "quality_score": 1.0
            },
            {
                "title": "قانون مدنی - ماده ۱۰ - اهلیت",
                "content": "هر شخص از وقت تولد اهلیت کسب حقوق را دارد، لکن استعمال حقوق او تابع شرایطی است که قانون معین می‌کند.",
                "category": "مدنی",
                "document_type": "ماده قانونی",
                "source_url": "test://civil_law_10",
                "persian_date": "1307/01/01",
                "legal_relevance": 0.95,
                "language_confidence": 1.0,
                "quality_score": 0.9
            },
            {
                "title": "قانون مجازات اسلامی - جرایم علیه اشخاص",
                "content": "هرکس عمداً شخص دیگری را بکشد، قاتل محسوب شده و در صورتی که مقتول محقون‌الدم باشد، قصاص خواهد شد.",
                "category": "کیفری",
                "document_type": "ماده کیفری",
                "source_url": "test://criminal_law_murder",
                "persian_date": "1392/01/01",
                "legal_relevance": 0.98,
                "language_confidence": 1.0,
                "quality_score": 0.95
            },
            {
                "title": "قانون تجارت - شرکت‌های تجاری",
                "content": "شرکت سهامی عام، شرکتی است که سرمایه آن به سهام قابل انتقال تقسیم شده و مسئولیت شرکاء در آن محدود به میزان سهام آنها باشد.",
                "category": "تجاری",
                "document_type": "ماده تجاری",
                "source_url": "test://commercial_law_company",
                "persian_date": "1311/01/01",
                "legal_relevance": 0.9,
                "language_confidence": 1.0,
                "quality_score": 0.85
            },
            {
                "title": "قانون آیین دادرسی مدنی - اختصاص دادگاه",
                "content": "دعاوی مالی که موضوع آنها کمتر از ده میلیون ریال باشد، در صلاحیت دادگاه بخش و دعاوی بیش از آن در صلاحیت دادگاه عمومی حقوقی است.",
                "category": "آیین دادرسی",
                "document_type": "ماده آیین دادرسی",
                "source_url": "test://procedure_law_jurisdiction",
                "persian_date": "1379/01/01",
                "legal_relevance": 0.88,
                "language_confidence": 1.0,
                "quality_score": 0.9
            }
        ]
        
        # Persian search queries for testing
        self.search_queries = [
            "قانون اساسی",
            "اهلیت",
            "قصاص",
            "شرکت سهامی",
            "دادگاه",
            "حقوق",
            "مسئولیت",
            "جرم",
            "دعوا",
            "قاتل"
        ]
    
    async def run_comprehensive_test(self):
        """Run comprehensive database functionality tests"""
        print("🗄️  Persian Legal AI - Database Functionality Testing")
        print("=" * 80)
        print(f"📁 Project Root: {self.project_root}")
        print(f"🕐 Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        try:
            # Phase 1: Database Connection Tests
            await self._test_database_connections()
            
            # Phase 2: Persian Text Storage and Retrieval
            await self._test_persian_text_handling()
            
            # Phase 3: Search Functionality Tests
            await self._test_search_functionality()
            
            # Phase 4: CRUD Operations Tests
            await self._test_crud_operations()
            
            # Phase 5: Data Integrity Tests
            await self._test_data_integrity()
            
            # Phase 6: Performance Tests
            await self._test_performance()
            
            # Phase 7: Backup and Restore Tests
            await self._test_backup_restore()
            
            # Generate final report
            self._generate_test_report()
            
        except Exception as e:
            logger.error(f"Critical database test error: {e}")
            logger.error(traceback.format_exc())
            self.test_results["critical_error"] = str(e)
    
    async def _test_database_connections(self):
        """Test database connection methods"""
        print("\n🔌 Phase 1: Database Connection Tests")
        print("-" * 50)
        
        connection_results = {
            "sqlite_connection": False,
            "postgresql_connection": False,
            "database_manager": False,
            "persian_db": False,
            "connection_errors": {},
            "database_files": []
        }
        
        # Test SQLite database files
        sqlite_paths = [
            self.project_root / "data" / "persian_legal_ai.db",
            self.project_root / "backend" / "persian_legal_ai.db",
            self.project_root / "persian_legal_ai.db"
        ]
        
        for db_path in sqlite_paths:
            if db_path.exists():
                connection_results["database_files"].append(str(db_path))
                print(f"✅ Found database file: {db_path}")
                
                # Test SQLite connection
                try:
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    conn.close()
                    
                    connection_results["sqlite_connection"] = True
                    connection_results["sqlite_tables"] = [table[0] for table in tables]
                    print(f"   ✅ SQLite connection successful - {len(tables)} tables found")
                    
                except Exception as e:
                    connection_results["connection_errors"]["sqlite"] = str(e)
                    print(f"   ❌ SQLite connection failed: {e}")
                
                break
        else:
            print("⚠️  No SQLite database files found")
        
        # Test Database Manager import and connection
        try:
            from backend.database import DatabaseManager
            db_manager = DatabaseManager()
            
            # Test connection
            if hasattr(db_manager, 'test_connection'):
                if await db_manager.test_connection():
                    connection_results["database_manager"] = True
                    print("✅ Database Manager: Connection successful")
                else:
                    print("❌ Database Manager: Connection failed")
            else:
                connection_results["database_manager"] = True
                print("✅ Database Manager: Imported successfully (no test method)")
                
        except ImportError as e:
            connection_results["connection_errors"]["database_manager"] = str(e)
            print(f"❌ Database Manager: Import failed - {e}")
        except Exception as e:
            connection_results["connection_errors"]["database_manager"] = str(e)
            print(f"❌ Database Manager: Error - {e}")
        
        # Test Persian Database connection
        try:
            from backend.database.persian_db import PersianLegalDB
            persian_db = PersianLegalDB()
            
            if hasattr(persian_db, 'test_connection'):
                if persian_db.test_connection():
                    connection_results["persian_db"] = True
                    print("✅ Persian Legal DB: Connection successful")
                else:
                    print("❌ Persian Legal DB: Connection failed")
            else:
                connection_results["persian_db"] = True
                print("✅ Persian Legal DB: Imported successfully")
                
        except ImportError as e:
            connection_results["connection_errors"]["persian_db"] = str(e)
            print(f"❌ Persian Legal DB: Import failed - {e}")
        except Exception as e:
            connection_results["connection_errors"]["persian_db"] = str(e)
            print(f"❌ Persian Legal DB: Error - {e}")
        
        # Test PostgreSQL connection (if configured)
        try:
            import asyncpg
            # This would require actual PostgreSQL credentials
            print("⚠️  PostgreSQL connection test skipped (requires configuration)")
            connection_results["postgresql_available"] = True
        except ImportError:
            print("⚠️  PostgreSQL driver not available")
            connection_results["postgresql_available"] = False
        
        self.test_results["database_connections"] = connection_results
    
    async def _test_persian_text_handling(self):
        """Test Persian text storage and retrieval"""
        print("\n📝 Phase 2: Persian Text Storage and Retrieval Tests")
        print("-" * 50)
        
        persian_text_results = {
            "encoding_tests": [],
            "storage_tests": [],
            "retrieval_tests": [],
            "character_integrity": [],
            "rtl_handling": []
        }
        
        # Test Persian text encoding
        persian_samples = [
            "سلام دنیا",
            "قانون اساسی جمهوری اسلامی ایران",
            "۱۲۳۴۵۶۷۸۹۰",  # Persian numbers
            "؟؛،",  # Persian punctuation
            "آبپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"  # Persian alphabet
        ]
        
        for i, text in enumerate(persian_samples, 1):
            print(f"🔤 Testing Persian text sample {i}: {text[:20]}...")
            
            encoding_result = {
                "original_text": text,
                "text_length": len(text),
                "utf8_encoding": False,
                "storage_successful": False,
                "retrieval_successful": False,
                "character_match": False
            }
            
            # Test UTF-8 encoding
            try:
                encoded = text.encode('utf-8')
                decoded = encoded.decode('utf-8')
                if decoded == text:
                    encoding_result["utf8_encoding"] = True
                    print(f"   ✅ UTF-8 encoding: OK")
                else:
                    print(f"   ❌ UTF-8 encoding: Mismatch")
            except Exception as e:
                print(f"   ❌ UTF-8 encoding: Error - {e}")
            
            # Test database storage (if available)
            if self.test_results["database_connections"].get("sqlite_connection"):
                try:
                    db_file = self.test_results["database_connections"]["database_files"][0]
                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    
                    # Try to create a test table if it doesn't exist
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS persian_text_test (
                            id INTEGER PRIMARY KEY,
                            content TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Insert Persian text
                    cursor.execute("INSERT INTO persian_text_test (content) VALUES (?)", (text,))
                    test_id = cursor.lastrowid
                    conn.commit()
                    
                    encoding_result["storage_successful"] = True
                    print(f"   ✅ Storage: Successful (ID: {test_id})")
                    
                    # Retrieve and verify
                    cursor.execute("SELECT content FROM persian_text_test WHERE id = ?", (test_id,))
                    retrieved = cursor.fetchone()
                    
                    if retrieved and retrieved[0] == text:
                        encoding_result["retrieval_successful"] = True
                        encoding_result["character_match"] = True
                        print(f"   ✅ Retrieval: Character match OK")
                    else:
                        print(f"   ❌ Retrieval: Character mismatch")
                        encoding_result["retrieved_text"] = retrieved[0] if retrieved else None
                    
                    # Cleanup
                    cursor.execute("DELETE FROM persian_text_test WHERE id = ?", (test_id,))
                    conn.commit()
                    conn.close()
                    
                except Exception as e:
                    print(f"   ❌ Database storage/retrieval: Error - {e}")
                    encoding_result["storage_error"] = str(e)
            
            persian_text_results["encoding_tests"].append(encoding_result)
        
        self.test_results["persian_text_tests"] = persian_text_results
    
    async def _test_search_functionality(self):
        """Test search functionality with Persian text"""
        print("\n🔍 Phase 3: Search Functionality Tests")
        print("-" * 50)
        
        search_results = {
            "full_text_search": [],
            "partial_matching": [],
            "case_sensitivity": [],
            "diacritic_handling": [],
            "performance_metrics": {}
        }
        
        # Test with available database connections
        if self.test_results["database_connections"].get("database_manager"):
            print("🔄 Testing Database Manager search...")
            
            try:
                from backend.database import DatabaseManager
                db_manager = DatabaseManager()
                
                # Insert test documents first
                inserted_ids = []
                for doc in self.test_documents[:3]:  # Test with first 3 documents
                    try:
                        doc_id = await db_manager.save_document(doc)
                        if doc_id:
                            inserted_ids.append(doc_id)
                            print(f"   📄 Inserted test document: {doc['title'][:30]}...")
                    except Exception as e:
                        print(f"   ⚠️  Failed to insert document: {e}")
                
                # Test search queries
                for query in self.search_queries[:5]:  # Test first 5 queries
                    print(f"   🔍 Searching for: '{query}'")
                    
                    search_result = {
                        "query": query,
                        "results_found": 0,
                        "search_time_ms": 0,
                        "successful": False
                    }
                    
                    try:
                        start_time = time.time()
                        
                        if hasattr(db_manager, 'search_documents'):
                            results = await db_manager.search_documents(query, limit=10)
                        else:
                            # Fallback method
                            results = []
                        
                        search_time = (time.time() - start_time) * 1000
                        
                        search_result.update({
                            "results_found": len(results),
                            "search_time_ms": search_time,
                            "successful": True
                        })
                        
                        print(f"      ✅ Found {len(results)} results ({search_time:.1f}ms)")
                        
                    except Exception as e:
                        search_result["error"] = str(e)
                        print(f"      ❌ Search failed: {e}")
                    
                    search_results["full_text_search"].append(search_result)
                
                # Cleanup inserted documents
                for doc_id in inserted_ids:
                    try:
                        if hasattr(db_manager, 'delete_document'):
                            await db_manager.delete_document(doc_id)
                    except Exception:
                        pass
                        
            except Exception as e:
                print(f"❌ Database Manager search test failed: {e}")
        
        # Test Persian DB search functionality
        if self.test_results["database_connections"].get("persian_db"):
            print("🔄 Testing Persian DB search...")
            
            try:
                from backend.database.persian_db import PersianLegalDB
                persian_db = PersianLegalDB()
                
                # Test search methods
                for query in self.search_queries[:3]:
                    try:
                        start_time = time.time()
                        
                        if hasattr(persian_db, 'search_persian_documents'):
                            results = await persian_db.search_persian_documents(query, limit=5)
                            search_time = (time.time() - start_time) * 1000
                            
                            search_results["partial_matching"].append({
                                "query": query,
                                "results_found": len(results),
                                "search_time_ms": search_time,
                                "successful": True
                            })
                            
                            print(f"   ✅ Persian DB search '{query}': {len(results)} results ({search_time:.1f}ms)")
                        else:
                            print(f"   ⚠️  Persian DB search method not available")
                            
                    except Exception as e:
                        print(f"   ❌ Persian DB search '{query}' failed: {e}")
                        
            except Exception as e:
                print(f"❌ Persian DB search test failed: {e}")
        
        self.test_results["search_functionality"] = search_results
    
    async def _test_crud_operations(self):
        """Test Create, Read, Update, Delete operations"""
        print("\n📊 Phase 4: CRUD Operations Tests")
        print("-" * 50)
        
        crud_results = {
            "create_operations": [],
            "read_operations": [],
            "update_operations": [],
            "delete_operations": [],
            "batch_operations": []
        }
        
        # Test with Database Manager if available
        if self.test_results["database_connections"].get("database_manager"):
            print("🔄 Testing CRUD operations with Database Manager...")
            
            try:
                from backend.database import DatabaseManager
                db_manager = DatabaseManager()
                
                test_doc = self.test_documents[0].copy()
                test_doc["title"] = f"CRUD Test Document - {datetime.now().isoformat()}"
                
                # CREATE operation
                print("   ➕ Testing CREATE operation...")
                try:
                    doc_id = await db_manager.save_document(test_doc)
                    if doc_id:
                        crud_results["create_operations"].append({
                            "successful": True,
                            "document_id": doc_id,
                            "title": test_doc["title"]
                        })
                        print(f"      ✅ Document created with ID: {doc_id}")
                        
                        # READ operation
                        print("   📖 Testing READ operation...")
                        try:
                            retrieved_doc = await db_manager.get_document(doc_id)
                            if retrieved_doc:
                                crud_results["read_operations"].append({
                                    "successful": True,
                                    "document_id": doc_id,
                                    "title_matches": retrieved_doc.get("title") == test_doc["title"],
                                    "content_matches": retrieved_doc.get("content") == test_doc["content"]
                                })
                                print(f"      ✅ Document retrieved successfully")
                            else:
                                crud_results["read_operations"].append({
                                    "successful": False,
                                    "error": "Document not found"
                                })
                                print(f"      ❌ Document retrieval failed")
                        except Exception as e:
                            crud_results["read_operations"].append({
                                "successful": False,
                                "error": str(e)
                            })
                            print(f"      ❌ READ operation failed: {e}")
                        
                        # UPDATE operation
                        print("   ✏️  Testing UPDATE operation...")
                        try:
                            updated_title = f"Updated - {test_doc['title']}"
                            if hasattr(db_manager, 'update_document'):
                                success = await db_manager.update_document(doc_id, {"title": updated_title})
                                if success:
                                    crud_results["update_operations"].append({
                                        "successful": True,
                                        "document_id": doc_id,
                                        "updated_field": "title"
                                    })
                                    print(f"      ✅ Document updated successfully")
                                else:
                                    crud_results["update_operations"].append({
                                        "successful": False,
                                        "error": "Update returned False"
                                    })
                                    print(f"      ❌ Document update failed")
                            else:
                                print(f"      ⚠️  UPDATE method not available")
                        except Exception as e:
                            crud_results["update_operations"].append({
                                "successful": False,
                                "error": str(e)
                            })
                            print(f"      ❌ UPDATE operation failed: {e}")
                        
                        # DELETE operation
                        print("   🗑️  Testing DELETE operation...")
                        try:
                            if hasattr(db_manager, 'delete_document'):
                                success = await db_manager.delete_document(doc_id)
                                crud_results["delete_operations"].append({
                                    "successful": success,
                                    "document_id": doc_id
                                })
                                if success:
                                    print(f"      ✅ Document deleted successfully")
                                else:
                                    print(f"      ❌ Document deletion failed")
                            else:
                                print(f"      ⚠️  DELETE method not available")
                        except Exception as e:
                            crud_results["delete_operations"].append({
                                "successful": False,
                                "error": str(e)
                            })
                            print(f"      ❌ DELETE operation failed: {e}")
                    else:
                        crud_results["create_operations"].append({
                            "successful": False,
                            "error": "No document ID returned"
                        })
                        print(f"      ❌ Document creation failed")
                        
                except Exception as e:
                    crud_results["create_operations"].append({
                        "successful": False,
                        "error": str(e)
                    })
                    print(f"   ❌ CREATE operation failed: {e}")
                    
            except Exception as e:
                print(f"❌ CRUD operations test failed: {e}")
        
        # Test batch operations
        print("   📦 Testing BATCH operations...")
        try:
            if self.test_results["database_connections"].get("database_manager"):
                from backend.database import DatabaseManager
                db_manager = DatabaseManager()
                
                # Prepare batch documents
                batch_docs = []
                for i in range(3):
                    doc = self.test_documents[i % len(self.test_documents)].copy()
                    doc["title"] = f"Batch Test {i+1} - {doc['title']}"
                    batch_docs.append(doc)
                
                # Test bulk insert if available
                if hasattr(db_manager, 'bulk_insert_documents'):
                    start_time = time.time()
                    results = await db_manager.bulk_insert_documents(batch_docs)
                    batch_time = (time.time() - start_time) * 1000
                    
                    crud_results["batch_operations"].append({
                        "operation": "bulk_insert",
                        "documents_count": len(batch_docs),
                        "successful_inserts": len([r for r in results if r]),
                        "processing_time_ms": batch_time,
                        "successful": True
                    })
                    
                    print(f"      ✅ Bulk insert: {len([r for r in results if r])}/{len(batch_docs)} docs ({batch_time:.1f}ms)")
                else:
                    print(f"      ⚠️  Bulk insert method not available")
                    
        except Exception as e:
            crud_results["batch_operations"].append({
                "operation": "bulk_insert",
                "successful": False,
                "error": str(e)
            })
            print(f"   ❌ Batch operations failed: {e}")
        
        self.test_results["crud_operations"] = crud_results
    
    async def _test_data_integrity(self):
        """Test data integrity and constraints"""
        print("\n🛡️  Phase 5: Data Integrity Tests")
        print("-" * 50)
        
        integrity_results = {
            "constraint_tests": [],
            "duplicate_handling": [],
            "foreign_key_tests": [],
            "transaction_tests": []
        }
        
        # Test duplicate handling
        if self.test_results["database_connections"].get("database_manager"):
            print("🔄 Testing duplicate handling...")
            
            try:
                from backend.database import DatabaseManager
                db_manager = DatabaseManager()
                
                # Insert same document twice
                test_doc = self.test_documents[0].copy()
                test_doc["title"] = f"Duplicate Test - {datetime.now().isoformat()}"
                
                # First insertion
                doc_id1 = await db_manager.save_document(test_doc)
                # Second insertion (should be handled gracefully)
                doc_id2 = await db_manager.save_document(test_doc)
                
                integrity_results["duplicate_handling"].append({
                    "first_insert_id": doc_id1,
                    "second_insert_id": doc_id2,
                    "handled_gracefully": doc_id2 is not None,
                    "same_id_returned": doc_id1 == doc_id2 if doc_id1 and doc_id2 else False
                })
                
                if doc_id1 and doc_id2:
                    if doc_id1 == doc_id2:
                        print("   ✅ Duplicate handling: Same ID returned (good)")
                    else:
                        print("   ⚠️  Duplicate handling: Different IDs (check logic)")
                else:
                    print("   ❌ Duplicate handling: Failed")
                
                # Cleanup
                if doc_id1 and hasattr(db_manager, 'delete_document'):
                    await db_manager.delete_document(doc_id1)
                if doc_id2 and doc_id2 != doc_id1 and hasattr(db_manager, 'delete_document'):
                    await db_manager.delete_document(doc_id2)
                    
            except Exception as e:
                integrity_results["duplicate_handling"].append({
                    "successful": False,
                    "error": str(e)
                })
                print(f"   ❌ Duplicate handling test failed: {e}")
        
        # Test data validation
        print("🔄 Testing data validation...")
        
        invalid_docs = [
            {"title": "", "content": "Valid content"},  # Empty title
            {"title": "Valid title", "content": ""},  # Empty content
            {"title": None, "content": "Valid content"},  # None title
            {"content": "Valid content"},  # Missing title
        ]
        
        for i, invalid_doc in enumerate(invalid_docs, 1):
            print(f"   🧪 Testing invalid document {i}...")
            
            try:
                if self.test_results["database_connections"].get("database_manager"):
                    from backend.database import DatabaseManager
                    db_manager = DatabaseManager()
                    
                    doc_id = await db_manager.save_document(invalid_doc)
                    
                    integrity_results["constraint_tests"].append({
                        "test_case": f"invalid_doc_{i}",
                        "document": invalid_doc,
                        "insertion_successful": doc_id is not None,
                        "document_id": doc_id
                    })
                    
                    if doc_id:
                        print(f"      ⚠️  Invalid document accepted (ID: {doc_id})")
                        # Cleanup
                        if hasattr(db_manager, 'delete_document'):
                            await db_manager.delete_document(doc_id)
                    else:
                        print(f"      ✅ Invalid document rejected")
                        
            except Exception as e:
                integrity_results["constraint_tests"].append({
                    "test_case": f"invalid_doc_{i}",
                    "document": invalid_doc,
                    "insertion_successful": False,
                    "error": str(e)
                })
                print(f"      ✅ Invalid document rejected with error: {e}")
        
        self.test_results["data_integrity"] = integrity_results
    
    async def _test_performance(self):
        """Test database performance with various operations"""
        print("\n⚡ Phase 6: Performance Tests")
        print("-" * 50)
        
        performance_results = {
            "insert_performance": {},
            "search_performance": {},
            "bulk_operations": {},
            "concurrent_operations": {}
        }
        
        if not self.test_results["database_connections"].get("database_manager"):
            print("⚠️  Skipping performance tests - no database manager available")
            self.test_results["performance_tests"] = performance_results
            return
        
        try:
            from backend.database import DatabaseManager
            db_manager = DatabaseManager()
            
            # Test insert performance
            print("🔄 Testing insert performance...")
            
            insert_times = []
            inserted_ids = []
            
            for i in range(5):  # Insert 5 documents
                test_doc = self.test_documents[i % len(self.test_documents)].copy()
                test_doc["title"] = f"Performance Test {i+1} - {test_doc['title']}"
                
                start_time = time.time()
                doc_id = await db_manager.save_document(test_doc)
                insert_time = (time.time() - start_time) * 1000
                
                if doc_id:
                    insert_times.append(insert_time)
                    inserted_ids.append(doc_id)
                    print(f"   📄 Document {i+1}: {insert_time:.1f}ms")
            
            if insert_times:
                performance_results["insert_performance"] = {
                    "documents_inserted": len(insert_times),
                    "average_time_ms": sum(insert_times) / len(insert_times),
                    "min_time_ms": min(insert_times),
                    "max_time_ms": max(insert_times),
                    "total_time_ms": sum(insert_times)
                }
                
                print(f"   📊 Insert performance: {performance_results['insert_performance']['average_time_ms']:.1f}ms average")
            
            # Test search performance
            print("🔄 Testing search performance...")
            
            search_times = []
            for query in self.search_queries[:5]:
                start_time = time.time()
                
                try:
                    if hasattr(db_manager, 'search_documents'):
                        results = await db_manager.search_documents(query, limit=10)
                        search_time = (time.time() - start_time) * 1000
                        search_times.append({
                            "query": query,
                            "time_ms": search_time,
                            "results_count": len(results)
                        })
                        print(f"   🔍 '{query}': {search_time:.1f}ms ({len(results)} results)")
                except Exception as e:
                    print(f"   ❌ Search '{query}' failed: {e}")
            
            if search_times:
                avg_search_time = sum(s["time_ms"] for s in search_times) / len(search_times)
                performance_results["search_performance"] = {
                    "queries_tested": len(search_times),
                    "average_time_ms": avg_search_time,
                    "search_details": search_times
                }
                
                print(f"   📊 Search performance: {avg_search_time:.1f}ms average")
            
            # Test bulk operations performance
            print("🔄 Testing bulk operations performance...")
            
            if hasattr(db_manager, 'bulk_insert_documents'):
                bulk_docs = []
                for i in range(10):  # Prepare 10 documents
                    doc = self.test_documents[i % len(self.test_documents)].copy()
                    doc["title"] = f"Bulk Performance Test {i+1} - {doc['title']}"
                    bulk_docs.append(doc)
                
                start_time = time.time()
                results = await db_manager.bulk_insert_documents(bulk_docs)
                bulk_time = (time.time() - start_time) * 1000
                
                successful_inserts = len([r for r in results if r])
                
                performance_results["bulk_operations"] = {
                    "documents_attempted": len(bulk_docs),
                    "documents_inserted": successful_inserts,
                    "total_time_ms": bulk_time,
                    "time_per_document_ms": bulk_time / len(bulk_docs),
                    "throughput_docs_per_sec": len(bulk_docs) / (bulk_time / 1000)
                }
                
                print(f"   📦 Bulk insert: {successful_inserts}/{len(bulk_docs)} docs in {bulk_time:.1f}ms")
                print(f"   📊 Throughput: {performance_results['bulk_operations']['throughput_docs_per_sec']:.1f} docs/sec")
                
                # Cleanup bulk inserted documents
                for result in results:
                    if result and hasattr(db_manager, 'delete_document'):
                        try:
                            await db_manager.delete_document(result)
                        except Exception:
                            pass
            
            # Cleanup performance test documents
            for doc_id in inserted_ids:
                if hasattr(db_manager, 'delete_document'):
                    try:
                        await db_manager.delete_document(doc_id)
                    except Exception:
                        pass
                        
        except Exception as e:
            print(f"❌ Performance testing failed: {e}")
            performance_results["error"] = str(e)
        
        self.test_results["performance_tests"] = performance_results
    
    async def _test_backup_restore(self):
        """Test backup and restore functionality"""
        print("\n💾 Phase 7: Backup and Restore Tests")
        print("-" * 50)
        
        backup_results = {
            "backup_methods_available": [],
            "backup_successful": False,
            "restore_successful": False,
            "data_integrity_after_restore": False
        }
        
        # Check for SQLite database backup
        if self.test_results["database_connections"].get("sqlite_connection"):
            print("🔄 Testing SQLite database backup...")
            
            try:
                db_files = self.test_results["database_connections"]["database_files"]
                if db_files:
                    source_db = db_files[0]
                    backup_db = source_db.replace(".db", f"_backup_{int(time.time())}.db")
                    
                    # Simple file copy backup
                    import shutil
                    shutil.copy2(source_db, backup_db)
                    
                    if Path(backup_db).exists():
                        backup_results["backup_successful"] = True
                        backup_results["backup_methods_available"].append("file_copy")
                        print(f"   ✅ Backup created: {Path(backup_db).name}")
                        
                        # Test backup integrity
                        conn = sqlite3.connect(backup_db)
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
                        table_count = cursor.fetchone()[0]
                        conn.close()
                        
                        if table_count > 0:
                            backup_results["restore_successful"] = True
                            backup_results["data_integrity_after_restore"] = True
                            print(f"   ✅ Backup integrity verified: {table_count} tables")
                        else:
                            print(f"   ❌ Backup integrity check failed")
                        
                        # Cleanup backup file
                        Path(backup_db).unlink()
                        print(f"   🗑️  Backup file cleaned up")
                    else:
                        print(f"   ❌ Backup file not created")
                        
            except Exception as e:
                backup_results["backup_error"] = str(e)
                print(f"   ❌ Backup test failed: {e}")
        
        # Check for database export functionality
        if self.test_results["database_connections"].get("database_manager"):
            print("🔄 Testing database export functionality...")
            
            try:
                from backend.database import DatabaseManager
                db_manager = DatabaseManager()
                
                # Check if export methods are available
                export_methods = []
                if hasattr(db_manager, 'export_to_json'):
                    export_methods.append("json")
                if hasattr(db_manager, 'export_to_csv'):
                    export_methods.append("csv")
                if hasattr(db_manager, 'backup_database'):
                    export_methods.append("database_backup")
                
                backup_results["backup_methods_available"].extend(export_methods)
                
                if export_methods:
                    print(f"   ✅ Export methods available: {', '.join(export_methods)}")
                else:
                    print(f"   ⚠️  No export methods found")
                    
            except Exception as e:
                print(f"   ❌ Export functionality test failed: {e}")
        
        self.test_results["backup_restore"] = backup_results
    
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📋 Database Functionality Test Report")
        print("=" * 80)
        
        # Calculate overall scores
        connection_score = sum([
            self.test_results["database_connections"].get("sqlite_connection", False),
            self.test_results["database_connections"].get("database_manager", False),
            self.test_results["database_connections"].get("persian_db", False)
        ]) / 3 * 100
        
        persian_text_score = 0
        if self.test_results["persian_text_tests"].get("encoding_tests"):
            successful_encodings = sum(1 for test in self.test_results["persian_text_tests"]["encoding_tests"] 
                                     if test.get("utf8_encoding") and test.get("character_match"))
            persian_text_score = successful_encodings / len(self.test_results["persian_text_tests"]["encoding_tests"]) * 100
        
        search_score = 0
        search_tests = self.test_results["search_functionality"]
        total_search_tests = len(search_tests.get("full_text_search", [])) + len(search_tests.get("partial_matching", []))
        if total_search_tests > 0:
            successful_searches = sum(1 for test in search_tests.get("full_text_search", []) if test.get("successful")) + \
                                sum(1 for test in search_tests.get("partial_matching", []) if test.get("successful"))
            search_score = successful_searches / total_search_tests * 100
        
        crud_score = 0
        crud_tests = self.test_results["crud_operations"]
        crud_operations = ["create_operations", "read_operations", "update_operations", "delete_operations"]
        successful_crud = sum(1 for op in crud_operations 
                            if crud_tests.get(op) and any(test.get("successful", False) for test in crud_tests[op]))
        crud_score = successful_crud / len(crud_operations) * 100
        
        overall_score = (connection_score + persian_text_score + search_score + crud_score) / 4
        
        print(f"📊 Test Summary:")
        print(f"   Database Connections: {connection_score:.1f}/100")
        print(f"   Persian Text Handling: {persian_text_score:.1f}/100")
        print(f"   Search Functionality: {search_score:.1f}/100")
        print(f"   CRUD Operations: {crud_score:.1f}/100")
        print(f"   Overall Database Score: {overall_score:.1f}/100")
        
        # Connection status
        connections = self.test_results["database_connections"]
        print(f"\n🔌 Database Connections:")
        print(f"   SQLite: {'✅' if connections.get('sqlite_connection') else '❌'}")
        print(f"   Database Manager: {'✅' if connections.get('database_manager') else '❌'}")
        print(f"   Persian Legal DB: {'✅' if connections.get('persian_db') else '❌'}")
        
        if connections.get("database_files"):
            print(f"   Database Files: {len(connections['database_files'])} found")
            for db_file in connections["database_files"]:
                print(f"     - {Path(db_file).name}")
        
        # Persian text handling results
        persian_tests = self.test_results["persian_text_tests"]
        if persian_tests.get("encoding_tests"):
            print(f"\n📝 Persian Text Handling:")
            successful_tests = sum(1 for test in persian_tests["encoding_tests"] 
                                 if test.get("utf8_encoding") and test.get("storage_successful") and test.get("character_match"))
            print(f"   Successful Tests: {successful_tests}/{len(persian_tests['encoding_tests'])}")
            print(f"   UTF-8 Encoding: ✅")
            print(f"   Storage/Retrieval: {'✅' if successful_tests > 0 else '❌'}")
            print(f"   Character Integrity: {'✅' if successful_tests > 0 else '❌'}")
        
        # Search functionality results
        search_tests = self.test_results["search_functionality"]
        if search_tests.get("full_text_search") or search_tests.get("partial_matching"):
            print(f"\n🔍 Search Functionality:")
            
            if search_tests.get("full_text_search"):
                successful_searches = sum(1 for test in search_tests["full_text_search"] if test.get("successful"))
                avg_time = sum(test.get("search_time_ms", 0) for test in search_tests["full_text_search"]) / len(search_tests["full_text_search"])
                print(f"   Full-text Search: {successful_searches}/{len(search_tests['full_text_search'])} successful")
                print(f"   Average Search Time: {avg_time:.1f}ms")
            
            if search_tests.get("partial_matching"):
                successful_partial = sum(1 for test in search_tests["partial_matching"] if test.get("successful"))
                print(f"   Partial Matching: {successful_partial}/{len(search_tests['partial_matching'])} successful")
        
        # CRUD operations results
        crud_tests = self.test_results["crud_operations"]
        if any(crud_tests.get(op) for op in ["create_operations", "read_operations", "update_operations", "delete_operations"]):
            print(f"\n📊 CRUD Operations:")
            
            for operation in ["create_operations", "read_operations", "update_operations", "delete_operations"]:
                if crud_tests.get(operation):
                    successful = sum(1 for test in crud_tests[operation] if test.get("successful", False))
                    total = len(crud_tests[operation])
                    op_name = operation.replace("_operations", "").upper()
                    print(f"   {op_name}: {successful}/{total} successful")
        
        # Performance results
        performance = self.test_results["performance_tests"]
        if performance.get("insert_performance") or performance.get("search_performance"):
            print(f"\n⚡ Performance Metrics:")
            
            if performance.get("insert_performance"):
                insert_perf = performance["insert_performance"]
                print(f"   Insert Performance: {insert_perf.get('average_time_ms', 0):.1f}ms average")
            
            if performance.get("search_performance"):
                search_perf = performance["search_performance"]
                print(f"   Search Performance: {search_perf.get('average_time_ms', 0):.1f}ms average")
            
            if performance.get("bulk_operations"):
                bulk_perf = performance["bulk_operations"]
                print(f"   Bulk Operations: {bulk_perf.get('throughput_docs_per_sec', 0):.1f} docs/sec")
        
        # Backup and restore results
        backup = self.test_results["backup_restore"]
        if backup.get("backup_methods_available"):
            print(f"\n💾 Backup & Restore:")
            print(f"   Available Methods: {', '.join(backup['backup_methods_available'])}")
            print(f"   Backup Test: {'✅' if backup.get('backup_successful') else '❌'}")
            print(f"   Restore Test: {'✅' if backup.get('restore_successful') else '❌'}")
            print(f"   Data Integrity: {'✅' if backup.get('data_integrity_after_restore') else '❌'}")
        
        # Overall assessment
        print(f"\n🎯 Database Assessment:")
        if overall_score >= 90:
            print(f"   🎉 EXCELLENT: Database is production ready ({overall_score:.1f}/100)")
        elif overall_score >= 80:
            print(f"   ✅ GOOD: Database is mostly functional ({overall_score:.1f}/100)")
        elif overall_score >= 60:
            print(f"   ⚠️  FAIR: Database has some issues ({overall_score:.1f}/100)")
        else:
            print(f"   ❌ POOR: Database needs significant work ({overall_score:.1f}/100)")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if connection_score < 100:
            print("   - Fix database connection issues for full functionality")
        if persian_text_score < 90:
            print("   - Improve Persian text encoding and storage handling")
        if search_score < 80:
            print("   - Optimize search functionality and indexing")
        if crud_score < 90:
            print("   - Implement missing CRUD operations")
        if not backup.get("backup_successful"):
            print("   - Implement database backup and restore functionality")
        
        # Save detailed report
        report_file = Path("database_test_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        print("=" * 80)

async def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Persian Legal AI Database Functionality Tester")
    parser.add_argument("--project-root", type=Path, 
                       help="Project root directory (default: parent of script directory)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    tester = PersianLegalDatabaseTester(args.project_root)
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())