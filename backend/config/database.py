import os
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
import aiosqlite

class Base(DeclarativeBase):
    pass

# Database URL from environment or default to SQLite
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///persian_legal_ai.db')

# Create async engine with proper configuration
engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv('DB_ECHO', 'false').lower() == 'true',
    future=True,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_database():
    """Dependency for getting database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_database():
    """Initialize database with FTS5 support for Persian text"""
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        
        # Enable FTS5 for Persian text search
        await conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS document_search USING fts5(
                title, 
                content, 
                keywords,
                tokenize='porter unicode'
            )
        """))
        
        # Create performance indexes
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_documents_date ON legal_documents(created_at)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_training_status ON training_sessions(status)"
        ))

async def test_connection():
    """Test database connectivity"""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False