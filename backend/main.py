#!/usr/bin/env python3
"""
Persian Legal AI Backend Server - Real Data Implementation
سرور Backend برای سیستم هوش مصنوعی حقوقی فارسی با داده‌های واقعی
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import psutil
import os

# Import real data components
from routes.team import router as team_router
from routes.models import router as models_router
from routes.monitoring import router as monitoring_router
from config.database import init_database, get_db
from models import *

# تنظیم لاگینگ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("persian_ai_backend.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# کلاس اصلی سرور
class PersianAIBackend:
    def __init__(self):
        self.app = FastAPI(
            title="Persian Legal AI Backend - Real Data",
            description="سرور Backend برای سیستم هوش مصنوعی حقوقی فارسی با داده‌های واقعی",
            version="2.0.0"
        )
        
        # تنظیم CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Include real data API routers
        self.app.include_router(team_router)
        self.app.include_router(models_router)
        self.app.include_router(monitoring_router)
        
        # متغیرهای سیستم
        self.connected_clients: List[WebSocket] = []
        self.system_active = True
        
        # Initialize database
        self.setup_database()
        
        # Setup routes
        self.setup_routes()
        
        # Setup WebSocket
        self.setup_websocket()
    
    def setup_database(self):
        """Initialize database tables"""
        try:
            init_database()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Persian Legal AI Backend - Real Data System",
                "version": "2.0.0",
                "status": "active",
                "timestamp": datetime.utcnow()
            }
        
        @self.app.get("/api/real/health")
        async def health_check():
            """System health check"""
            try:
                cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                return {
                    "status": "healthy",
                    "timestamp": datetime.utcnow(),
                    "system": {
                        "cpu_usage": cpu_usage,
                        "memory_usage": memory.percent,
                        "disk_usage": (disk.used / disk.total) * 100
                    }
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow()
                }
        
        @self.app.get("/api/real/stats")
        async def get_system_stats(db = Depends(get_db)):
            """Get system statistics from real database"""
            try:
                # Get real statistics from database
                from sqlalchemy import func
                
                team_count = db.query(TeamMember).filter(TeamMember.is_active == True).count()
                model_count = db.query(ModelTraining).count()
                active_models = db.query(ModelTraining).filter(ModelTraining.status == 'training').count()
                
                return {
                    "team_members": team_count,
                    "total_models": model_count,
                    "active_models": active_models,
                    "timestamp": datetime.utcnow()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
    
    def setup_websocket(self):
        """Setup WebSocket for real-time updates"""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.connected_clients.append(websocket)
            logger.info(f"Client connected. Total clients: {len(self.connected_clients)}")
            
            try:
                while True:
                    # Send real-time system metrics
                    try:
                        cpu_usage = psutil.cpu_percent(interval=1)
                        memory = psutil.virtual_memory()
                        
                        metrics = {
                            "type": "system_metrics",
                            "data": {
                                "cpu_usage": cpu_usage,
                                "memory_usage": memory.percent,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                        
                        await websocket.send_text(json.dumps(metrics))
                        await asyncio.sleep(5)  # Send updates every 5 seconds
                        
                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        logger.error(f"WebSocket error: {e}")
                        break
                        
            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self.connected_clients:
                    self.connected_clients.remove(websocket)
                logger.info(f"Client disconnected. Total clients: {len(self.connected_clients)}")
    
    async def broadcast_update(self, message: dict):
        """Broadcast message to all connected clients"""
        if self.connected_clients:
            disconnected = []
            for client in self.connected_clients:
                try:
                    await client.send_text(json.dumps(message))
                except:
                    disconnected.append(client)
            
            # Remove disconnected clients
            for client in disconnected:
                self.connected_clients.remove(client)
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the server"""
        logger.info(f"Starting Persian Legal AI Backend on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="info")

# Create and run the server
if __name__ == "__main__":
    server = PersianAIBackend()
    server.run()