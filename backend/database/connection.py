"""
Database Connection Manager for Persian Legal AI
مدیر اتصال پایگاه داده برای هوش مصنوعی حقوقی فارسی
"""

import os
import logging
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from .models import Base

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database connection and session management
    """
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or self._get_database_url()
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _get_database_url(self) -> str:
        """Get database URL from environment or default"""
        # Try PostgreSQL first
        postgres_url = os.getenv('DATABASE_URL')
        if postgres_url:
            return postgres_url
        
        # Try individual PostgreSQL components
        postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
        postgres_port = os.getenv('POSTGRES_PORT', '5432')
        postgres_db = os.getenv('POSTGRES_DB', 'persian_legal_ai')
        postgres_user = os.getenv('POSTGRES_USER', 'postgres')
        postgres_password = os.getenv('POSTGRES_PASSWORD', 'password')
        
        if all([postgres_host, postgres_db, postgres_user, postgres_password]):
            return f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
        
        # Fallback to SQLite
        sqlite_path = os.getenv('SQLITE_PATH', 'persian_legal_ai.db')
        return f"sqlite:///{sqlite_path}"
    
    def _initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            # Create engine with appropriate settings
            if self.database_url.startswith('sqlite'):
                # SQLite configuration
                self.engine = create_engine(
                    self.database_url,
                    poolclass=StaticPool,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": 30
                    },
                    echo=False  # Set to True for SQL debugging
                )
            else:
                # PostgreSQL configuration
                self.engine = create_engine(
                    self.database_url,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False  # Set to True for SQL debugging
                )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
            
            # Create all tables
            Base.metadata.create_all(self.engine)
            
            logger.info(f"Database initialized: {self.database_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get database session"""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()
    
    @contextmanager
    def get_session_context(self):
        """Get database session with context manager"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_session_context() as session:
                session.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        try:
            with self.get_session_context() as session:
                # Get database version
                if self.database_url.startswith('postgresql'):
                    result = session.execute(text("SELECT version()"))
                    version = result.fetchone()[0]
                elif self.database_url.startswith('sqlite'):
                    result = session.execute(text("SELECT sqlite_version()"))
                    version = result.fetchone()[0]
                else:
                    version = "Unknown"
                
                # Get table information
                tables = Base.metadata.tables.keys()
                
                return {
                    'database_url': self.database_url.split('@')[-1] if '@' in self.database_url else self.database_url,
                    'version': version,
                    'tables': list(tables),
                    'connection_pool_size': self.engine.pool.size() if hasattr(self.engine.pool, 'size') else None
                }
                
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {}
    
    def backup_database(self, backup_path: str) -> bool:
        """Backup database"""
        try:
            if self.database_url.startswith('sqlite'):
                # SQLite backup
                import shutil
                shutil.copy2(self.database_url.replace('sqlite:///', ''), backup_path)
                logger.info(f"SQLite database backed up to: {backup_path}")
                return True
            
            elif self.database_url.startswith('postgresql'):
                # PostgreSQL backup using pg_dump
                import subprocess
                cmd = [
                    'pg_dump',
                    self.database_url,
                    '-f', backup_path
                ]
                subprocess.run(cmd, check=True)
                logger.info(f"PostgreSQL database backed up to: {backup_path}")
                return True
            
            else:
                logger.error(f"Backup not supported for database type: {self.database_url}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup"""
        try:
            if self.database_url.startswith('sqlite'):
                # SQLite restore
                import shutil
                shutil.copy2(backup_path, self.database_url.replace('sqlite:///', ''))
                logger.info(f"SQLite database restored from: {backup_path}")
                return True
            
            elif self.database_url.startswith('postgresql'):
                # PostgreSQL restore using psql
                import subprocess
                cmd = [
                    'psql',
                    self.database_url,
                    '-f', backup_path
                ]
                subprocess.run(cmd, check=True)
                logger.info(f"PostgreSQL database restored from: {backup_path}")
                return True
            
            else:
                logger.error(f"Restore not supported for database type: {self.database_url}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restore database: {e}")
            return False
    
    def optimize_database(self) -> bool:
        """Optimize database performance"""
        try:
            with self.get_session_context() as session:
                if self.database_url.startswith('sqlite'):
                    # SQLite optimization
                    session.execute(text("VACUUM"))
                    session.execute(text("ANALYZE"))
                    logger.info("SQLite database optimized")
                
                elif self.database_url.startswith('postgresql'):
                    # PostgreSQL optimization
                    session.execute(text("VACUUM ANALYZE"))
                    logger.info("PostgreSQL database optimized")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to optimize database: {e}")
            return False
    
    def get_table_stats(self) -> Dict[str, Any]:
        """Get table statistics"""
        try:
            stats = {}
            
            with self.get_session_context() as session:
                for table_name in Base.metadata.tables.keys():
                    if self.database_url.startswith('sqlite'):
                        result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        count = result.fetchone()[0]
                    elif self.database_url.startswith('postgresql'):
                        result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        count = result.fetchone()[0]
                    else:
                        count = 0
                    
                    stats[table_name] = count
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get table stats: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """Cleanup old data"""
        try:
            from datetime import datetime, timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            with self.get_session_context() as session:
                # Cleanup old training metrics
                from .models import TrainingMetrics
                old_metrics = session.query(TrainingMetrics).filter(
                    TrainingMetrics.timestamp < cutoff_date
                ).delete()
                
                # Cleanup old checkpoints (keep only best and latest)
                from .models import ModelCheckpoint
                old_checkpoints = session.query(ModelCheckpoint).filter(
                    ModelCheckpoint.created_at < cutoff_date,
                    ModelCheckpoint.checkpoint_type.notin_(['best', 'latest'])
                ).delete()
                
                total_cleaned = old_metrics + old_checkpoints
                logger.info(f"Cleaned up {total_cleaned} old records")
                return total_cleaned
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0
    
    def close(self):
        """Close database connections"""
        try:
            if self.engine:
                self.engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Failed to close database connections: {e}")

# Global database manager instance
db_manager = DatabaseManager()

def get_db_session():
    """Get database session (for dependency injection)"""
    return db_manager.get_session()

def get_db_session_context():
    """Get database session context (for dependency injection)"""
    return db_manager.get_session_context()