#!/usr/bin/env python3
"""
Persian Legal AI Backend Server - Enhanced Security Implementation
سرور Backend با امنیت پیشرفته برای سیستم هوش مصنوعی حقوقی فارسی
"""

import asyncio
import logging
import time
import ssl
from datetime import datetime
from typing import Dict, List, Optional, Any
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import psutil
import os

# Import real API endpoints
from api.system_endpoints import router as system_router
from api.training_endpoints import router as training_router
from api.enhanced_health import router as health_router
from api.real_data_endpoints import router as real_data_router
from auth.routes import router as auth_router

# Import database and optimization
from database.connection import init_database, db_manager
from optimization.system_optimizer import system_optimizer

# Import security middleware
from middleware.rate_limiter import rate_limit_middleware

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("persian_ai_backend.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PersianAIBackend:
    """Enhanced Persian AI Backend Server with Security"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Persian Legal AI Backend - Enhanced Security",
            description="سرور Backend با امنیت پیشرفته برای سیستم هوش مصنوعی حقوقی فارسی",
            version="2.1.0"
        )
        
        # Setup CORS with environment-based origins
        cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://frontend:80").split(",")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
        
        # Add rate limiting middleware
        self.app.middleware("http")(rate_limit_middleware)
        
        # Include routers
        self.app.include_router(auth_router)
        self.app.include_router(health_router)
        self.app.include_router(system_router)
        self.app.include_router(training_router)
        self.app.include_router(real_data_router)
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Enhanced Persian AI Backend initialized with security features")
    
    def _setup_routes(self):
        """Setup additional routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Persian Legal AI Backend - Real Implementation",
                "status": "running",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "2.0.0"
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                # Check database connection
                db_healthy = db_manager.test_connection()
                
                # Check system resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                return {
                    "status": "healthy" if db_healthy else "degraded",
                    "timestamp": datetime.utcnow().isoformat(),
                    "database": "connected" if db_healthy else "disconnected",
                    "system": {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_available_gb": memory.available / (1024**3)
                    },
                    "optimization": {
                        "active": system_optimizer.monitoring_active,
                        "optimal_batch_size": system_optimizer.get_optimal_batch_size(),
                        "optimal_workers": system_optimizer.get_optimal_num_workers()
                    }
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    # Send periodic system updates
                    await asyncio.sleep(5)
                    
                    # Get current system metrics
                    system_metrics = await self._get_system_metrics()
                    
                    # Send to all connected clients
                    await self._broadcast_to_connections(system_metrics)
                    
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
                logger.info("WebSocket client disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get optimization report
            optimization_report = system_optimizer.get_optimization_report()
            
            return {
                "type": "system_metrics",
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": (disk.used / disk.total) * 100,
                    "disk_free_gb": disk.free / (1024**3)
                },
                "optimization": optimization_report
            }
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {
                "type": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def _broadcast_to_connections(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket connections"""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)
    
    async def startup(self):
        """Startup tasks"""
        try:
            logger.info("Starting Persian AI Backend...")
            
            # Initialize database
            if init_database():
                logger.info("Database initialized successfully")
            else:
                logger.error("Database initialization failed")
            
            # Start system optimization monitoring
            system_optimizer.monitor_system(interval=10.0)
            logger.info("System optimization monitoring started")
            
            logger.info("Persian AI Backend startup completed")
            
        except Exception as e:
            logger.error(f"Startup failed: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown tasks"""
        try:
            logger.info("Shutting down Persian AI Backend...")
            
            # Stop system optimization monitoring
            system_optimizer.stop_monitoring()
            
            # Close database connections
            db_manager.close()
            
            # Close WebSocket connections
            for connection in self.active_connections:
                try:
                    await connection.close()
                except Exception as e:
                    logger.warning(f"Failed to close WebSocket: {e}")
            
            logger.info("Persian AI Backend shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")

# Create backend instance
backend = PersianAIBackend()

# Add startup and shutdown events
@backend.app.on_event("startup")
async def startup_event():
    await backend.startup()

@backend.app.on_event("shutdown")
async def shutdown_event():
    await backend.shutdown()

# Get FastAPI app
app = backend.app

def create_ssl_context():
    """Create SSL context for HTTPS"""
    try:
        cert_file = "/app/certificates/server.crt"
        key_file = "/app/certificates/server.key"
        
        if os.path.exists(cert_file) and os.path.exists(key_file):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(cert_file, key_file)
            logger.info("SSL context created successfully")
            return ssl_context
        else:
            logger.warning("SSL certificates not found, running without HTTPS")
            return None
    except Exception as e:
        logger.error(f"Failed to create SSL context: {e}")
        return None

def main():
    """Main function to run the server"""
    try:
        logger.info("Starting Enhanced Persian Legal AI Backend Server...")
        
        # Check for SSL certificates
        ssl_context = create_ssl_context()
        use_https = ssl_context is not None
        
        # Determine port and protocol
        if use_https:
            port = int(os.getenv("HTTPS_PORT", "8443"))
            logger.info(f"Starting server with HTTPS on port {port}")
        else:
            port = int(os.getenv("HTTP_PORT", "8000"))
            logger.info(f"Starting server with HTTP on port {port}")
        
        # Run server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            ssl_context=ssl_context,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        raise

if __name__ == "__main__":
    main()