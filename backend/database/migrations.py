"""
Database Migration System for Persian Legal AI
سیستم انتقال پایگاه داده برای هوش مصنوعی حقوقی فارسی
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy import text, MetaData, Table, Column, Integer, String, DateTime, Float, JSON, Boolean, Text, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)

class MigrationManager:
    """Database migration manager for Persian Legal AI"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///persian_legal_ai.db')
        self.is_sqlite = 'sqlite' in self.database_url
        self.is_postgres = 'postgresql' in self.database_url
        
        # Create async engine
        self.engine = create_async_engine(
            self.database_url,
            echo=os.getenv('DB_ECHO', 'false').lower() == 'true',
            future=True,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False} if self.is_sqlite else {}
        )
        
        # Create session factory
        self.AsyncSessionLocal = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        self.migrations = [
            self._migration_001_initial_tables,
            self._migration_002_indexes,
            self._migration_003_fts_search,
            self._migration_004_system_logs,
            self._migration_005_data_sources,
        ]
    
    async def get_current_version(self) -> int:
        """Get current database schema version"""
        try:
            async with self.AsyncSessionLocal() as session:
                # Create migration table if it doesn't exist
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                await session.commit()
                
                # Get latest version
                result = await session.execute(text(
                    "SELECT MAX(version) FROM schema_migrations"
                ))
                version = result.scalar()
                return version or 0
        except Exception as e:
            logger.error(f"Failed to get current version: {e}")
            return 0
    
    async def set_version(self, version: int):
        """Set database schema version"""
        try:
            async with self.AsyncSessionLocal() as session:
                await session.execute(text(
                    "INSERT INTO schema_migrations (version) VALUES (:version)"
                ), {"version": version})
                await session.commit()
                logger.info(f"Set database version to {version}")
        except Exception as e:
            logger.error(f"Failed to set version {version}: {e}")
            raise
    
    async def run_migrations(self) -> bool:
        """Run all pending migrations"""
        try:
            current_version = await self.get_current_version()
            logger.info(f"Current database version: {current_version}")
            
            for i, migration in enumerate(self.migrations, 1):
                if i > current_version:
                    logger.info(f"Running migration {i}: {migration.__name__}")
                    try:
                        await migration()
                        await self.set_version(i)
                        logger.info(f"✅ Migration {i} completed successfully")
                    except Exception as e:
                        logger.error(f"❌ Migration {i} failed: {e}")
                        return False
            
            final_version = await self.get_current_version()
            logger.info(f"✅ All migrations completed. Database version: {final_version}")
            return True
            
        except Exception as e:
            logger.error(f"Migration process failed: {e}")
            return False
    
    async def _migration_001_initial_tables(self):
        """Migration 1: Create initial tables"""
        async with self.AsyncSessionLocal() as session:
            # Training sessions table
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id VARCHAR PRIMARY KEY,
                    model_name VARCHAR NOT NULL,
                    model_type VARCHAR NOT NULL,
                    status VARCHAR NOT NULL DEFAULT 'pending',
                    config JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    current_epoch INTEGER DEFAULT 0,
                    total_epochs INTEGER DEFAULT 0,
                    current_step INTEGER DEFAULT 0,
                    total_steps INTEGER DEFAULT 0,
                    current_loss REAL,
                    best_loss REAL,
                    current_accuracy REAL,
                    best_accuracy REAL,
                    learning_rate REAL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    training_speed REAL,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    data_source VARCHAR,
                    task_type VARCHAR,
                    train_samples INTEGER DEFAULT 0,
                    eval_samples INTEGER DEFAULT 0
                )
            """))
            
            # Model checkpoints table
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS model_checkpoints (
                    id VARCHAR PRIMARY KEY,
                    session_id VARCHAR NOT NULL,
                    epoch INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    checkpoint_type VARCHAR NOT NULL,
                    loss REAL NOT NULL,
                    accuracy REAL,
                    learning_rate REAL,
                    file_path VARCHAR NOT NULL,
                    file_size_bytes INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES training_sessions(id) ON DELETE CASCADE
                )
            """))
            
            # Training metrics table
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id VARCHAR PRIMARY KEY,
                    session_id VARCHAR NOT NULL,
                    epoch INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    loss REAL NOT NULL,
                    accuracy REAL,
                    learning_rate REAL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    gpu_usage REAL,
                    gpu_memory REAL,
                    training_speed REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES training_sessions(id) ON DELETE CASCADE
                )
            """))
            
            # Legal documents table
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS legal_documents (
                    id VARCHAR PRIMARY KEY,
                    title VARCHAR NOT NULL,
                    content TEXT NOT NULL,
                    source VARCHAR NOT NULL,
                    category VARCHAR,
                    word_count INTEGER,
                    char_count INTEGER,
                    language_confidence REAL,
                    legal_relevance REAL,
                    quality_score REAL,
                    document_metadata JSON,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            await session.commit()
            logger.info("Created initial database tables")
    
    async def _migration_002_indexes(self):
        """Migration 2: Create performance indexes"""
        async with self.AsyncSessionLocal() as session:
            # Indexes for training sessions
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_training_sessions_status ON training_sessions(status)"
            ))
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_training_sessions_created ON training_sessions(created_at)"
            ))
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_training_sessions_model_type ON training_sessions(model_type)"
            ))
            
            # Indexes for legal documents
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_legal_documents_source ON legal_documents(source)"
            ))
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_legal_documents_category ON legal_documents(category)"
            ))
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_legal_documents_created ON legal_documents(created_at)"
            ))
            
            # Indexes for training metrics
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_training_metrics_session ON training_metrics(session_id)"
            ))
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_training_metrics_timestamp ON training_metrics(timestamp)"
            ))
            
            await session.commit()
            logger.info("Created performance indexes")
    
    async def _migration_003_fts_search(self):
        """Migration 3: Create FTS search tables (SQLite only)"""
        if not self.is_sqlite:
            logger.info("Skipping FTS migration for non-SQLite database")
            return
        
        async with self.AsyncSessionLocal() as session:
            try:
                # Create FTS5 virtual table for document search
                # Use simple tokenizer instead of porter unicode to avoid compatibility issues
                await session.execute(text("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS document_search USING fts5(
                        title, 
                        content, 
                        keywords,
                        tokenize='simple'
                    )
                """))
                
                # Create triggers to keep FTS table in sync
                await session.execute(text("""
                    CREATE TRIGGER IF NOT EXISTS documents_fts_insert AFTER INSERT ON legal_documents
                    BEGIN
                        INSERT INTO document_search(rowid, title, content, keywords) 
                        VALUES (NEW.rowid, NEW.title, NEW.content, NEW.category);
                    END
                """))
                
                await session.execute(text("""
                    CREATE TRIGGER IF NOT EXISTS documents_fts_update AFTER UPDATE ON legal_documents
                    BEGIN
                        UPDATE document_search SET 
                            title = NEW.title, 
                            content = NEW.content, 
                            keywords = NEW.category
                        WHERE rowid = NEW.rowid;
                    END
                """))
                
                await session.execute(text("""
                    CREATE TRIGGER IF NOT EXISTS documents_fts_delete AFTER DELETE ON legal_documents
                    BEGIN
                        DELETE FROM document_search WHERE rowid = OLD.rowid;
                    END
                """))
                
                await session.commit()
                logger.info("Created FTS search tables and triggers")
                
            except Exception as e:
                logger.warning(f"FTS search creation failed (this is OK): {e}")
                # Continue without FTS if it's not available
                await session.rollback()
    
    async def _migration_004_system_logs(self):
        """Migration 4: Create system logs table"""
        async with self.AsyncSessionLocal() as session:
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id VARCHAR PRIMARY KEY,
                    level VARCHAR NOT NULL,
                    message TEXT NOT NULL,
                    component VARCHAR,
                    session_id VARCHAR,
                    context JSON,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES training_sessions(id) ON DELETE SET NULL
                )
            """))
            
            # Index for log queries
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp)"
            ))
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level)"
            ))
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs(component)"
            ))
            
            await session.commit()
            logger.info("Created system logs table")
    
    async def _migration_005_data_sources(self):
        """Migration 5: Create data sources table"""
        async with self.AsyncSessionLocal() as session:
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS data_sources (
                    id VARCHAR PRIMARY KEY,
                    name VARCHAR NOT NULL UNIQUE,
                    source_type VARCHAR NOT NULL,
                    url VARCHAR,
                    config JSON,
                    total_documents INTEGER DEFAULT 0,
                    processed_documents INTEGER DEFAULT 0,
                    quality_score REAL,
                    is_active BOOLEAN DEFAULT TRUE,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Indexes for data sources
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_data_sources_type ON data_sources(source_type)"
            ))
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_data_sources_active ON data_sources(is_active)"
            ))
            
            await session.commit()
            logger.info("Created data sources table")
    
    async def backup_database(self, backup_path: str = None) -> str:
        """Create database backup"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"persian_legal_ai_backup_{timestamp}.sql"
        
        try:
            if self.is_sqlite:
                # For SQLite, copy the database file
                import shutil
                db_path = self.database_url.replace("sqlite+aiosqlite:///", "")
                if os.path.exists(db_path):
                    backup_db_path = backup_path.replace(".sql", ".db")
                    shutil.copy2(db_path, backup_db_path)
                    logger.info(f"SQLite database backed up to {backup_db_path}")
                    return backup_db_path
                else:
                    logger.warning("SQLite database file not found")
                    return ""
            else:
                # For PostgreSQL, use pg_dump (would need to be implemented)
                logger.warning("PostgreSQL backup not implemented yet")
                return ""
                
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return ""
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            async with self.AsyncSessionLocal() as session:
                stats = {
                    "database_type": "SQLite" if self.is_sqlite else "PostgreSQL",
                    "database_url": self.database_url.split("@")[-1] if "@" in self.database_url else self.database_url,
                    "version": await self.get_current_version(),
                    "tables": {}
                }
                
                # Get table counts
                tables = [
                    "training_sessions",
                    "model_checkpoints", 
                    "training_metrics",
                    "legal_documents",
                    "system_logs",
                    "data_sources"
                ]
                
                for table in tables:
                    try:
                        result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = result.scalar()
                        stats["tables"][table] = count or 0
                    except Exception:
                        stats["tables"][table] = 0
                
                # Get database size (SQLite only)
                if self.is_sqlite:
                    db_path = self.database_url.replace("sqlite+aiosqlite:///", "")
                    if os.path.exists(db_path):
                        size_bytes = os.path.getsize(db_path)
                        stats["size_bytes"] = size_bytes
                        stats["size_mb"] = round(size_bytes / (1024 * 1024), 2)
                    else:
                        stats["size_bytes"] = 0
                        stats["size_mb"] = 0
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Database health check"""
        try:
            start_time = datetime.now()
            
            async with self.AsyncSessionLocal() as session:
                # Test basic connectivity
                await session.execute(text("SELECT 1"))
                
                # Test table access
                await session.execute(text("SELECT COUNT(*) FROM schema_migrations"))
                
                response_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "status": "healthy",
                    "response_time": response_time,
                    "database_type": "SQLite" if self.is_sqlite else "PostgreSQL",
                    "version": await self.get_current_version()
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time": -1
            }
    
    async def close(self):
        """Close database connections"""
        try:
            await self.engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

# Global migration manager
migration_manager = None

async def init_migration_manager(database_url: str = None) -> MigrationManager:
    """Initialize global migration manager"""
    global migration_manager
    migration_manager = MigrationManager(database_url)
    return migration_manager

async def run_migrations(database_url: str = None) -> bool:
    """Run database migrations"""
    manager = await init_migration_manager(database_url)
    return await manager.run_migrations()

async def get_database_health() -> Dict[str, Any]:
    """Get database health status"""
    global migration_manager
    if migration_manager is None:
        migration_manager = await init_migration_manager()
    return await migration_manager.health_check()

async def get_database_stats() -> Dict[str, Any]:
    """Get database statistics"""
    global migration_manager
    if migration_manager is None:
        migration_manager = await init_migration_manager()
    return await migration_manager.get_database_stats()