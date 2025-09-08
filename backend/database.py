import sqlite3
import aiosqlite
from typing import List, Dict, Optional
import hashlib
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "iranian_legal_archive.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with proper schema for Persian legal documents"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable foreign keys
                conn.execute("PRAGMA foreign_keys = ON")
                
                # Create main documents table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT UNIQUE NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source TEXT NOT NULL,
                        category TEXT,
                        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        content_hash TEXT UNIQUE NOT NULL,
                        metadata TEXT,
                        classification_confidence REAL DEFAULT 0.0
                    )
                """)
                
                # Create FTS5 virtual table for Persian full-text search
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                        title, content, category, source,
                        content='documents',
                        content_rowid='id',
                        tokenize='unicode61 remove_diacritics 2'
                    )
                """)
                
                # Create triggers to keep FTS table in sync
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS documents_fts_insert AFTER INSERT ON documents BEGIN
                        INSERT INTO documents_fts(rowid, title, content, category, source)
                        VALUES (new.id, new.title, new.content, new.category, new.source);
                    END
                """)
                
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS documents_fts_delete AFTER DELETE ON documents BEGIN
                        DELETE FROM documents_fts WHERE rowid = old.id;
                    END
                """)
                
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS documents_fts_update AFTER UPDATE ON documents BEGIN
                        UPDATE documents_fts SET 
                            title = new.title,
                            content = new.content,
                            category = new.category,
                            source = new.source
                        WHERE rowid = new.id;
                    END
                """)
                
                # Create indexes for better performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_scraped_at ON documents(scraped_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash)")
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise
    
    async def search_documents(self, query: str, category: Optional[str] = None, 
                             limit: int = 10, offset: int = 0) -> List[Dict]:
        """Full-text search with optional category filter"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Prepare FTS query - escape special characters
                fts_query = query.replace('"', '""')
                
                if category:
                    sql = """
                        SELECT d.*, 
                               snippet(documents_fts, 1, '<mark>', '</mark>', '...', 32) as snippet,
                               bm25(documents_fts) as rank
                        FROM documents_fts 
                        JOIN documents d ON documents_fts.rowid = d.id
                        WHERE documents_fts MATCH ? AND d.category = ?
                        ORDER BY rank ASC
                        LIMIT ? OFFSET ?
                    """
                    params = (f'"{fts_query}"', category, limit, offset)
                else:
                    sql = """
                        SELECT d.*, 
                               snippet(documents_fts, 1, '<mark>', '</mark>', '...', 32) as snippet,
                               bm25(documents_fts) as rank
                        FROM documents_fts 
                        JOIN documents d ON documents_fts.rowid = d.id
                        WHERE documents_fts MATCH ?
                        ORDER BY rank ASC
                        LIMIT ? OFFSET ?
                    """
                    params = (f'"{fts_query}"', limit, offset)
                
                async with conn.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    return [dict(zip(columns, row)) for row in rows]
                    
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
    
    async def count_search_results(self, query: str, category: Optional[str] = None) -> int:
        """Count total search results"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                fts_query = query.replace('"', '""')
                
                if category:
                    sql = """
                        SELECT COUNT(*)
                        FROM documents_fts 
                        JOIN documents d ON documents_fts.rowid = d.id
                        WHERE documents_fts MATCH ? AND d.category = ?
                    """
                    params = (f'"{fts_query}"', category)
                else:
                    sql = """
                        SELECT COUNT(*)
                        FROM documents_fts 
                        WHERE documents_fts MATCH ?
                    """
                    params = (f'"{fts_query}"',)
                
                async with conn.execute(sql, params) as cursor:
                    result = await cursor.fetchone()
                    return result[0] if result else 0
                    
        except Exception as e:
            logger.error(f"Count search results error: {str(e)}")
            return 0
    
    async def get_document(self, document_id: int) -> Optional[Dict]:
        """Get single document by ID"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                async with conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        columns = [description[0] for description in cursor.description]
                        return dict(zip(columns, row))
                    return None
        except Exception as e:
            logger.error(f"Get document error: {str(e)}")
            return None
    
    async def save_document(self, document: Dict) -> Optional[int]:
        """Save document with duplicate detection"""
        try:
            content_hash = hashlib.md5(document["content"].encode('utf-8')).hexdigest()
            
            async with aiosqlite.connect(self.db_path) as conn:
                try:
                    # Insert into main table
                    cursor = await conn.execute("""
                        INSERT INTO documents (url, title, content, source, category, content_hash, metadata, classification_confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        document["url"], 
                        document["title"], 
                        document["content"], 
                        document["source"], 
                        document.get("category"), 
                        content_hash,
                        json.dumps(document.get("metadata", {})),
                        document.get("classification_confidence", 0.0)
                    ))
                    
                    document_id = cursor.lastrowid
                    await conn.commit()
                    
                    logger.info(f"Saved document {document_id}: {document['title'][:50]}...")
                    return document_id
                    
                except sqlite3.IntegrityError as e:
                    if "UNIQUE constraint failed" in str(e):
                        logger.debug(f"Document already exists: {document['url']}")
                        return None
                    else:
                        raise
                        
        except Exception as e:
            logger.error(f"Save document error: {str(e)}")
            return None
    
    async def get_documents_by_category(self, category: str, limit: int = 10) -> List[Dict]:
        """Get documents by category"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                async with conn.execute("""
                    SELECT * FROM documents 
                    WHERE category = ? 
                    ORDER BY scraped_at DESC 
                    LIMIT ?
                """, (category, limit)) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Get documents by category error: {str(e)}")
            return []
    
    async def update_document_category(self, document_id: int, category: str, confidence: float = 0.0):
        """Update document category and classification confidence"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("""
                    UPDATE documents 
                    SET category = ?, classification_confidence = ?
                    WHERE id = ?
                """, (category, confidence, document_id))
                
                await conn.commit()
                logger.info(f"Updated document {document_id} category to {category}")
                
        except Exception as e:
            logger.error(f"Update document category error: {str(e)}")
            raise
    
    async def get_document_stats(self) -> Dict:
        """Get comprehensive document statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                stats = {}
                
                # Total documents
                async with conn.execute("SELECT COUNT(*) FROM documents") as cursor:
                    result = await cursor.fetchone()
                    stats["total_documents"] = result[0] if result else 0
                
                # Documents by category
                async with conn.execute("""
                    SELECT category, COUNT(*) 
                    FROM documents 
                    WHERE category IS NOT NULL 
                    GROUP BY category
                """) as cursor:
                    rows = await cursor.fetchall()
                    stats["documents_by_category"] = {row[0]: row[1] for row in rows}
                
                # Documents by source
                async with conn.execute("""
                    SELECT source, COUNT(*) 
                    FROM documents 
                    GROUP BY source
                """) as cursor:
                    rows = await cursor.fetchall()
                    stats["documents_by_source"] = {row[0]: row[1] for row in rows}
                
                # Recent activity (last 7 days)
                async with conn.execute("""
                    SELECT DATE(scraped_at) as date, COUNT(*) as count
                    FROM documents 
                    WHERE scraped_at >= datetime('now', '-7 days')
                    GROUP BY DATE(scraped_at)
                    ORDER BY date DESC
                """) as cursor:
                    rows = await cursor.fetchall()
                    stats["recent_scraping_activity"] = [
                        {"date": row[0], "documents_added": row[1]} 
                        for row in rows
                    ]
                
                return stats
                
        except Exception as e:
            logger.error(f"Get document stats error: {str(e)}")
            return {}
    
    async def get_category_stats(self) -> Dict:
        """Get detailed category statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                async with conn.execute("""
                    SELECT 
                        category,
                        COUNT(*) as count,
                        AVG(classification_confidence) as avg_confidence,
                        MIN(scraped_at) as first_document,
                        MAX(scraped_at) as last_document
                    FROM documents 
                    WHERE category IS NOT NULL 
                    GROUP BY category
                    ORDER BY count DESC
                """) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    return [dict(zip(columns, row)) for row in rows]
                    
        except Exception as e:
            logger.error(f"Get category stats error: {str(e)}")
            return []