import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///persian_legal_ai.db')

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv('DB_ECHO', 'false').lower() == 'true',
    future=True
)

# Create session factory
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

async def get_database():
    """Dependency for getting database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()