"""
Real Database Connection for Persian Legal AI
اتصال واقعی پایگاه داده برای هوش مصنوعی حقوقی فارسی
"""

import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Real database manager with SQLite backend"""
    
    def __init__(self, database_url: str = None):
        if database_url is None:
            # Use SQLite for development
            database_path = os.path.join(os.getcwd(), "persian_legal_ai.db")
            database_url = f"sqlite:///{database_path}"
        
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            # Create engine
            if self.database_url.startswith("sqlite"):
                self.engine = create_engine(
                    self.database_url,
                    poolclass=StaticPool,
                    connect_args={"check_same_thread": False},
                    echo=False
                )
            else:
                self.engine = create_engine(self.database_url, echo=False)
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create tables
            self._create_tables()
            
            logger.info(f"Database initialized successfully: {self.database_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables"""
        try:
            from .models import Base
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with context manager"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_database_info(self) -> dict:
        """Get database information"""
        try:
            with self.get_session() as session:
                # Get table information
                if self.database_url.startswith("sqlite"):
                    result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                    tables = [row[0] for row in result.fetchall()]
                else:
                    # For PostgreSQL/MySQL
                    result = session.execute(text("SHOW TABLES"))
                    tables = [row[0] for row in result.fetchall()]
                
                # Get database size (SQLite specific)
                db_size = 0
                if self.database_url.startswith("sqlite"):
                    db_path = self.database_url.replace("sqlite:///", "")
                    if os.path.exists(db_path):
                        db_size = os.path.getsize(db_path)
                
                return {
                    "database_url": self.database_url,
                    "tables": tables,
                    "table_count": len(tables),
                    "database_size_bytes": db_size,
                    "database_size_mb": round(db_size / (1024 * 1024), 2) if db_size > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {
                "database_url": self.database_url,
                "tables": [],
                "table_count": 0,
                "database_size_bytes": 0,
                "database_size_mb": 0,
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def execute_query(self, query: str, params: dict = None) -> list:
        """Execute raw SQL query"""
        try:
            with self.get_session() as session:
                result = session.execute(text(query), params or {})
                return result.fetchall()
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Failed to close database connection: {e}")

# Global database manager instance
db_manager = DatabaseManager()

def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    with db_manager.get_session() as session:
        yield session

def init_database():
    """Initialize database (called at startup)"""
    try:
        # Test connection
        if db_manager.test_connection():
            logger.info("Database connection successful")
            return True
        else:
            logger.error("Database connection failed")
            return False
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False