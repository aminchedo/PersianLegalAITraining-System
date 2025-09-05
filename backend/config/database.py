from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from typing import Generator

# REAL PostgreSQL connection
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://persianai:password@localhost:5432/persian_legal_ai')

engine = create_engine(
    DATABASE_URL, 
    pool_size=20, 
    max_overflow=0,
    echo=False  # Set to True for SQL debugging
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db() -> Generator:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")