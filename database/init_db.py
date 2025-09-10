import asyncio
import sqlite3
from pathlib import Path
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def init_database():
    """Initialize database with proper schema"""
    
    # Create basic SQLite database
    db_path = "persian_legal_ai.db"
    conn = sqlite3.connect(db_path)
    
    # Create basic tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS legal_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            keywords TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    """)
    
    # Create FTS5 virtual table for Persian text search
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS document_search USING fts5(
                title, 
                content, 
                keywords,
                tokenize='porter unicode'
            )
        """)
    except sqlite3.OperationalError:
        print("⚠️  FTS5 not available, using regular text search")
    
    # Create indexes for performance
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_date ON legal_documents(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_training_status ON training_sessions(status)")
    
    conn.commit()
    conn.close()
    
    print("✅ Database initialized successfully")

if __name__ == "__main__":
    asyncio.run(init_database())