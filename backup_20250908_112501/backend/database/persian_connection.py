"""
Persian Legal AI Database Connection with FTS5 Support
اتصال پایگاه داده هوش مصنوعی حقوقی فارسی با پشتیبانی FTS5
"""

import sqlite3
import aiosqlite
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from contextlib import asynccontextmanager
import json
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class PersianLegalDatabase:
    """Persian-optimized SQLite database with FTS5 full-text search"""
    
    def __init__(self, db_path: str = "persian_legal_ai.db"):
        self.db_path = Path(db_path)
        self._initialized = False
        self.init_persian_database()
    
    def init_persian_database(self):
        """Initialize database with Persian-optimized schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable UTF-8 encoding for Persian text
                conn.execute("PRAGMA encoding = 'UTF-8'")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute("PRAGMA cache_size = 10000")
                conn.execute("PRAGMA temp_store = memory")
                
                # Create main legal documents table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS legal_documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source_url TEXT UNIQUE,
                        document_type TEXT, -- قانون، آیین‌نامه، رأی، مصوبه
                        category TEXT,     -- حقوق مدنی، کیفری، اداری، تجاری
                        subcategory TEXT,  -- زیرمجموعه‌های تخصصی
                        persian_date TEXT, -- تاریخ شمسی
                        gregorian_date TEXT, -- تاریخ میلادی
                        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        content_hash TEXT UNIQUE,
                        classification_confidence REAL,
                        named_entities TEXT, -- JSON array of extracted entities
                        keywords TEXT,       -- JSON array of Persian keywords
                        summary TEXT,        -- خلاصه فارسی
                        word_count INTEGER,
                        char_count INTEGER,
                        quality_score REAL,
                        language_confidence REAL,
                        legal_relevance REAL,
                        is_processed BOOLEAN DEFAULT FALSE,
                        processing_status TEXT DEFAULT 'pending' -- pending, processing, completed, failed
                    )
                """)
                
                # Create Persian FTS5 virtual table for full-text search
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS documents_search USING fts5(
                        title, content, category, document_type, keywords, summary,
                        content='legal_documents',
                        content_rowid='id',
                        tokenize='unicode61 remove_diacritics 1'
                    )
                """)
                
                # Create FTS5 triggers for automatic index updates
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON legal_documents BEGIN
                        INSERT INTO documents_search(rowid, title, content, category, document_type, keywords, summary)
                        VALUES (new.id, new.title, new.content, new.category, new.document_type, new.keywords, new.summary);
                    END
                """)
                
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON legal_documents BEGIN
                        INSERT INTO documents_search(documents_search, rowid, title, content, category, document_type, keywords, summary)
                        VALUES ('delete', old.id, old.title, old.content, old.category, old.document_type, old.keywords, old.summary);
                    END
                """)
                
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON legal_documents BEGIN
                        INSERT INTO documents_search(documents_search, rowid, title, content, category, document_type, keywords, summary)
                        VALUES ('delete', old.id, old.title, old.content, old.category, old.document_type, old.keywords, old.summary);
                        INSERT INTO documents_search(rowid, title, content, category, document_type, keywords, summary)
                        VALUES (new.id, new.title, new.content, new.category, new.document_type, new.keywords, new.summary);
                    END
                """)
                
                # Create performance indexes for Persian queries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_persian_category ON legal_documents(category)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_document_type ON legal_documents(document_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_persian_date ON legal_documents(persian_date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_quality_score ON legal_documents(quality_score)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_processing_status ON legal_documents(processing_status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON legal_documents(content_hash)")
                
                # Create training data table for AI models
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS training_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id INTEGER REFERENCES legal_documents(id),
                        text_chunk TEXT NOT NULL,
                        label TEXT NOT NULL,
                        label_confidence REAL,
                        chunk_position INTEGER,
                        chunk_size INTEGER,
                        model_version TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_validated BOOLEAN DEFAULT FALSE
                    )
                """)
                
                # Create model performance tracking table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        accuracy REAL,
                        precision_score REAL,
                        recall_score REAL,
                        f1_score REAL,
                        test_samples INTEGER,
                        training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        config_hash TEXT,
                        notes TEXT
                    )
                """)
                
                # Create system statistics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        stat_name TEXT NOT NULL,
                        stat_value TEXT NOT NULL,
                        stat_type TEXT NOT NULL, -- counter, gauge, histogram
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                self._initialized = True
                logger.info("✅ Persian database initialized successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Persian database: {e}")
            raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Async context manager for database connections"""
        conn = None
        try:
            conn = await aiosqlite.connect(self.db_path)
            conn.row_factory = aiosqlite.Row
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                await conn.close()
    
    async def insert_document(self, document_data: Dict) -> int:
        """Insert a new legal document with Persian optimization"""
        try:
            # Generate content hash for deduplication
            content_hash = hashlib.sha256(
                document_data['content'].encode('utf-8')
            ).hexdigest()
            
            async with self.get_connection() as conn:
                cursor = await conn.execute("""
                    INSERT OR IGNORE INTO legal_documents (
                        title, content, source_url, document_type, category, subcategory,
                        persian_date, gregorian_date, content_hash, keywords, summary,
                        word_count, char_count, quality_score, language_confidence, legal_relevance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    document_data.get('title', ''),
                    document_data.get('content', ''),
                    document_data.get('source_url'),
                    document_data.get('document_type'),
                    document_data.get('category'),
                    document_data.get('subcategory'),
                    document_data.get('persian_date'),
                    document_data.get('gregorian_date'),
                    content_hash,
                    json.dumps(document_data.get('keywords', []), ensure_ascii=False),
                    document_data.get('summary', ''),
                    document_data.get('word_count', 0),
                    document_data.get('char_count', 0),
                    document_data.get('quality_score', 0.0),
                    document_data.get('language_confidence', 0.0),
                    document_data.get('legal_relevance', 0.0)
                ))
                
                await conn.commit()
                return cursor.lastrowid
                
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            raise
    
    async def search_persian_documents(
        self,
        query: str,
        category: Optional[str] = None,
        document_type: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict]:
        """Persian full-text search with filters"""
        try:
            async with self.get_connection() as conn:
                # Build the search query
                sql_parts = ["""
                    SELECT 
                        d.id, d.title, d.content, d.category, d.document_type,
                        d.persian_date, d.quality_score, d.summary,
                        snippet(documents_search, 1, '<mark>', '</mark>', '...', 32) as snippet,
                        rank
                    FROM documents_search 
                    JOIN legal_documents d ON documents_search.rowid = d.id
                    WHERE documents_search MATCH ?
                """]
                
                params = [query]
                
                if category:
                    sql_parts.append("AND d.category = ?")
                    params.append(category)
                
                if document_type:
                    sql_parts.append("AND d.document_type = ?")
                    params.append(document_type)
                
                sql_parts.append("ORDER BY rank LIMIT ? OFFSET ?")
                params.extend([limit, offset])
                
                sql_query = " ".join(sql_parts)
                
                cursor = await conn.execute(sql_query, params)
                rows = await cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def get_document_stats(self) -> Dict:
        """Get comprehensive document statistics"""
        try:
            async with self.get_connection() as conn:
                # Total documents
                cursor = await conn.execute("SELECT COUNT(*) FROM legal_documents")
                total_docs = (await cursor.fetchone())[0]
                
                # Documents by category
                cursor = await conn.execute("""
                    SELECT category, COUNT(*) as count 
                    FROM legal_documents 
                    WHERE category IS NOT NULL 
                    GROUP BY category
                """)
                by_category = dict(await cursor.fetchall())
                
                # Documents by type
                cursor = await conn.execute("""
                    SELECT document_type, COUNT(*) as count 
                    FROM legal_documents 
                    WHERE document_type IS NOT NULL 
                    GROUP BY document_type
                """)
                by_type = dict(await cursor.fetchall())
                
                # Processing status
                cursor = await conn.execute("""
                    SELECT processing_status, COUNT(*) as count 
                    FROM legal_documents 
                    GROUP BY processing_status
                """)
                by_status = dict(await cursor.fetchall())
                
                # Quality distribution
                cursor = await conn.execute("""
                    SELECT 
                        CASE 
                            WHEN quality_score >= 0.8 THEN 'high'
                            WHEN quality_score >= 0.6 THEN 'medium'
                            WHEN quality_score >= 0.4 THEN 'low'
                            ELSE 'very_low'
                        END as quality_level,
                        COUNT(*) as count
                    FROM legal_documents 
                    WHERE quality_score IS NOT NULL
                    GROUP BY quality_level
                """)
                quality_dist = dict(await cursor.fetchall())
                
                return {
                    "total_documents": total_docs,
                    "by_category": by_category,
                    "by_type": by_type,
                    "by_status": by_status,
                    "quality_distribution": quality_dist,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get document stats: {e}")
            return {}
    
    async def update_document_processing(
        self,
        document_id: int,
        classification_result: Dict,
        named_entities: List = None,
        summary: str = None
    ) -> bool:
        """Update document with AI processing results"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    UPDATE legal_documents 
                    SET 
                        classification_confidence = ?,
                        named_entities = ?,
                        summary = ?,
                        is_processed = TRUE,
                        processing_status = 'completed'
                    WHERE id = ?
                """, (
                    classification_result.get('confidence', 0.0),
                    json.dumps(named_entities or [], ensure_ascii=False),
                    summary,
                    document_id
                ))
                
                await conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to update document processing: {e}")
            return False
    
    async def get_training_data(self, limit: int = 1000) -> List[Dict]:
        """Get training data for AI models"""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.execute("""
                    SELECT 
                        td.text_chunk,
                        td.label,
                        td.label_confidence,
                        d.category,
                        d.document_type
                    FROM training_data td
                    JOIN legal_documents d ON td.document_id = d.id
                    WHERE td.is_validated = TRUE
                    LIMIT ?
                """, (limit,))
                
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return []
    
    async def record_model_performance(self, performance_data: Dict) -> bool:
        """Record model performance metrics"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO model_performance (
                        model_name, model_version, accuracy, precision_score, 
                        recall_score, f1_score, test_samples, config_hash, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    performance_data.get('model_name'),
                    performance_data.get('model_version'),
                    performance_data.get('accuracy'),
                    performance_data.get('precision'),
                    performance_data.get('recall'),
                    performance_data.get('f1_score'),
                    performance_data.get('test_samples'),
                    performance_data.get('config_hash'),
                    performance_data.get('notes')
                ))
                
                await conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to record model performance: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test database connection synchronously"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

# Global instance
persian_db = PersianLegalDatabase()