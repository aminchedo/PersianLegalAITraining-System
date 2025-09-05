#!/usr/bin/env python3
"""
Persian Legal AI Backend Server
سرور Backend برای داشبورد پیشرفته
"""

import asyncio
import json
import logging
import random
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

# Import new training system components
from api.training_endpoints import router as training_router
from api.model_endpoints import router as model_router
from api.system_endpoints import router as system_router
from services.training_service import training_service
from database.connection import db_manager

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

# مدل‌های داده
class SystemMetrics(BaseModel):
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = 0.0
    temperature: Optional[float] = 45.0
    power_consumption: Optional[float] = 150.0
    timestamp: datetime

class ModelStatus(BaseModel):
    id: int
    name: str
    status: str  # training, completed, pending, error
    progress: int
    accuracy: float
    loss: float
    epochs_completed: int
    time_remaining: str
    dora_rank: int
    learning_rate: float

class DataSource(BaseModel):
    name: str
    documents: int
    quality: int
    status: str
    collection_rate: int  # documents per hour

class TrainingConfig(BaseModel):
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 3
    dora_rank: int = 64
    dora_alpha: float = 16.0
    enable_decomposition: bool = True

class Notification(BaseModel):
    id: int
    type: str  # success, warning, error, info
    title: str
    message: str
    timestamp: datetime
    read: bool = False

class LogEntry(BaseModel):
    id: int
    level: str
    message: str
    timestamp: datetime
    component: str

# کلاس اصلی سرور
class PersianAIBackend:
    def __init__(self):
        self.app = FastAPI(
            title="Persian Legal AI Backend",
            description="سرور Backend برای سیستم هوش مصنوعی حقوقی فارسی",
            version="1.0.0"
        )
        
        # تنظیم CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Include new API routers
        self.app.include_router(training_router)
        self.app.include_router(model_router)
        self.app.include_router(system_router)
        
        # متغیرهای سیستم
        self.connected_clients: List[WebSocket] = []
        self.system_active = True
        self.training_active = False
        self.collection_active = False
        
        # داده‌های نمونه
        self.models = [
            ModelStatus(
                id=1,
                name="PersianMind-v1.0",
                status="training",
                progress=67,
                accuracy=89.2,
                loss=0.23,
                epochs_completed=8,
                time_remaining="2 ساعت 15 دقیقه",
                dora_rank=64,
                learning_rate=1e-4
            ),
            ModelStatus(
                id=2,
                name="ParsBERT-Legal",
                status="completed",
                progress=100,
                accuracy=92.5,
                loss=0.15,
                epochs_completed=12,
                time_remaining="تکمیل شده",
                dora_rank=32,
                learning_rate=5e-5
            ),
            ModelStatus(
                id=3,
                name="Persian-QA-Advanced",
                status="pending",
                progress=0,
                accuracy=0.0,
                loss=0.0,
                epochs_completed=0,
                time_remaining="در انتظار",
                dora_rank=128,
                learning_rate=2e-4
            ),
            ModelStatus(
                id=4,
                name="Legal-NER-v2",
                status="error",
                progress=45,
                accuracy=67.8,
                loss=0.45,
                epochs_completed=3,
                time_remaining="خطا در آموزش",
                dora_rank=64,
                learning_rate=1e-4
            )
        ]
        
        self.data_sources = [
            DataSource(name="پیکره نعب", documents=15420, quality=94, status="active", collection_rate=125),
            DataSource(name="پورتال قوانین", documents=8932, quality=87, status="active", collection_rate=89),
            DataSource(name="مجلس شورای اسلامی", documents=5673, quality=92, status="active", collection_rate=67),
            DataSource(name="پورتال داده ایران", documents=3241, quality=78, status="inactive", collection_rate=0)
        ]
        
        self.notifications = [
            Notification(
                id=1,
                type="success",
                title="آموزش تکمیل شد",
                message="مدل ParsBERT-Legal با دقت 92.5% آموزش داده شد",
                timestamp=datetime.now() - timedelta(minutes=2)
            ),
            Notification(
                id=2,
                type="warning",
                title="هشدار عملکرد",
                message="استفاده از CPU به 85% رسیده - بهینه‌سازی توصیه می‌شود",
                timestamp=datetime.now() - timedelta(minutes=5)
            ),
            Notification(
                id=3,
                type="info",
                title="جمع‌آوری داده",
                message="جمع‌آوری 1,247 سند جدید از پیکره نعب",
                timestamp=datetime.now() - timedelta(minutes=10),
                read=True
            )
        ]
        
        self.logs = []
        self.log_counter = 1
        
        # تنظیم routes
        self.setup_routes()
        
        # شروع background tasks
        asyncio.create_task(self.background_tasks())
        
        logger.info("🚀 سرور Persian AI Backend آماده شد")

    def setup_routes(self):
        """تنظیم API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Persian Legal AI Backend Server",
                "version": "1.0.0",
                "status": "active",
                "timestamp": datetime.now()
            }

        @self.app.get("/api/status")
        async def get_status():
            return {
                "system_active": self.system_active,
                "training_active": self.training_active,
                "collection_active": self.collection_active,
                "connected_clients": len(self.connected_clients),
                "timestamp": datetime.now()
            }

        @self.app.get("/api/metrics")
        async def get_metrics():
            """دریافت معیارهای سیستم"""
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                metrics = SystemMetrics(
                    cpu_usage=cpu_percent,
                    memory_usage=memory.percent,
                    gpu_usage=random.uniform(10, 95),  # شبیه‌سازی GPU
                    temperature=random.uniform(35, 65),
                    power_consumption=random.uniform(100, 250),
                    timestamp=datetime.now()
                )
                
                return metrics.dict()
                
            except Exception as e:
                logger.error(f"خطا در دریافت معیارها: {e}")
                raise HTTPException(status_code=500, detail="خطا در دریافت معیارها")

        @self.app.get("/api/models")
        async def get_models():
            """دریافت وضعیت مدل‌ها"""
            return [model.dict() for model in self.models]

        @self.app.post("/api/training/start")
        async def start_training(config: TrainingConfig):
            """شروع آموزش"""
            try:
                self.training_active = True
                
                # پیدا کردن مدل pending برای شروع آموزش
                for model in self.models:
                    if model.status == "pending":
                        model.status = "training"
                        model.learning_rate = config.learning_rate
                        model.dora_rank = config.dora_rank
                        break
                
                # ثبت لاگ
                await self.add_log("INFO", f"آموزش با تنظیمات جدید شروع شد", "trainer")
                
                # ارسال اعلان
                await self.add_notification(
                    "info",
                    "شروع آموزش",
                    f"آموزش با نرخ یادگیری {config.learning_rate} شروع شد"
                )
                
                return {
                    "success": True,
                    "message": "آموزش با موفقیت شروع شد",
                    "config": config.dict()
                }
                
            except Exception as e:
                logger.error(f"خطا در شروع آموزش: {e}")
                raise HTTPException(status_code=500, detail="خطا در شروع آموزش")

        @self.app.post("/api/training/stop")
        async def stop_training():
            """توقف آموزش"""
            try:
                self.training_active = False
                
                # توقف مدل‌های در حال آموزش
                for model in self.models:
                    if model.status == "training":
                        model.status = "pending"
                
                await self.add_log("WARNING", "آموزش توسط کاربر متوقف شد", "trainer")
                
                await self.add_notification(
                    "warning",
                    "توقف آموزش",
                    "آموزش توسط کاربر متوقف شد"
                )
                
                return {
                    "success": True,
                    "message": "آموزش متوقف شد"
                }
                
            except Exception as e:
                logger.error(f"خطا در توقف آموزش: {e}")
                raise HTTPException(status_code=500, detail="خطا در توقف آموزش")

        @self.app.get("/api/data/stats")
        async def get_data_stats():
            """دریافت آمار جمع‌آوری داده"""
            total_docs = sum(source.documents for source in self.data_sources)
            active_sources = len([s for s in self.data_sources if s.status == "active"])
            
            return {
                "total_documents": total_docs,
                "quality_documents": int(total_docs * 0.87),  # 87% کیفیت متوسط
                "active_sources": active_sources,
                "collection_rate": sum(s.collection_rate for s in self.data_sources),
                "sources": [source.dict() for source in self.data_sources],
                "quality_distribution": {
                    "excellent": int(total_docs * 0.45),
                    "good": int(total_docs * 0.35),
                    "fair": int(total_docs * 0.15),
                    "poor": int(total_docs * 0.05)
                }
            }

        @self.app.post("/api/data/collection/start")
        async def start_data_collection():
            """شروع جمع‌آوری داده"""
            self.collection_active = True
            
            await self.add_log("INFO", "جمع‌آوری داده شروع شد", "data-collector")
            
            return {
                "success": True,
                "message": "جمع‌آوری داده شروع شد"
            }

        @self.app.post("/api/data/collection/stop")
        async def stop_data_collection():
            """توقف جمع‌آوری داده"""
            self.collection_active = False
            
            await self.add_log("WARNING", "جمع‌آوری داده متوقف شد", "data-collector")
            
            return {
                "success": True,
                "message": "جمع‌آوری داده متوقف شد"
            }

        @self.app.get("/api/notifications")
        async def get_notifications():
            """دریافت اعلان‌ها"""
            return [notif.dict() for notif in self.notifications]

        @self.app.post("/api/notifications/{notification_id}/read")
        async def mark_notification_read(notification_id: int):
            """علامت‌گذاری اعلان به عنوان خوانده شده"""
            for notif in self.notifications:
                if notif.id == notification_id:
                    notif.read = True
                    break
            
            return {"success": True}

        @self.app.get("/api/logs")
        async def get_logs(limit: int = 100):
            """دریافت لاگ‌های سیستم"""
            return [log.dict() for log in self.logs[-limit:]]

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket برای ارتباط real-time"""
            await websocket.accept()
            self.connected_clients.append(websocket)
            logger.info(f"کلاینت جدید متصل شد. تعداد کل: {len(self.connected_clients)}")
            
            try:
                while True:
                    # دریافت پیام از کلاینت
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # پردازش پیام
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }))
                    
            except WebSocketDisconnect:
                self.connected_clients.remove(websocket)
                logger.info(f"کلاینت قطع شد. تعداد باقی‌مانده: {len(self.connected_clients)}")

    async def add_log(self, level: str, message: str, component: str):
        """اضافه کردن لاگ جدید"""
        log_entry = LogEntry(
            id=self.log_counter,
            level=level,
            message=message,
            timestamp=datetime.now(),
            component=component
        )
        
        self.logs.append(log_entry)
        self.log_counter += 1
        
        # نگه داشتن فقط 1000 لاگ آخر
        if len(self.logs) > 1000:
            self.logs = self.logs[-1000:]
        
        # ارسال به کلاینت‌ها
        await self.broadcast_to_clients({
            "type": "new_log",
            "data": log_entry.dict()
        })
        
        logger.info(f"[{component}] {level}: {message}")

    async def add_notification(self, notif_type: str, title: str, message: str):
        """اضافه کردن اعلان جدید"""
        notification = Notification(
            id=len(self.notifications) + 1,
            type=notif_type,
            title=title,
            message=message,
            timestamp=datetime.now()
        )
        
        self.notifications.insert(0, notification)  # جدیدترین در بالا
        
        # نگه داشتن فقط 50 اعلان آخر
        if len(self.notifications) > 50:
            self.notifications = self.notifications[:50]
        
        # ارسال به کلاینت‌ها
        await self.broadcast_to_clients({
            "type": "new_notification",
            "data": notification.dict()
        })

    async def broadcast_to_clients(self, message: dict):
        """ارسال پیام به همه کلاینت‌ها"""
        if not self.connected_clients:
            return
        
        message_str = json.dumps(message, default=str, ensure_ascii=False)
        
        disconnected = []
        for client in self.connected_clients:
            try:
                await client.send_text(message_str)
            except:
                disconnected.append(client)
        
        # حذف کلاینت‌های قطع شده
        for client in disconnected:
            self.connected_clients.remove(client)

    async def background_tasks(self):
        """وظایف پس‌زمینه"""
        await asyncio.sleep(2)  # انتظار برای راه‌اندازی کامل
        
        while True:
            try:
                # به‌روزرسانی پیشرفت مدل‌ها
                await self.update_training_progress()
                
                # ارسال معیارهای سیستم
                await self.send_system_metrics()
                
                # شبیه‌سازی رویدادهای تصادفی
                if random.random() < 0.1:  # 10% احتمال
                    await self.simulate_random_event()
                
                await asyncio.sleep(3)  # هر 3 ثانیه
                
            except Exception as e:
                logger.error(f"خطا در background tasks: {e}")
                await asyncio.sleep(5)

    async def update_training_progress(self):
        """به‌روزرسانی پیشرفت آموزش"""
        if not self.training_active:
            return
        
        for model in self.models:
            if model.status == "training":
                # افزایش تدریجی پیشرفت
                if model.progress < 100:
                    model.progress += random.randint(1, 3)
                    model.accuracy += random.uniform(0.1, 0.5)
                    model.loss = max(0.05, model.loss - random.uniform(0.001, 0.01))
                    model.epochs_completed += random.choice([0, 1])
                
                # تکمیل آموزش
                if model.progress >= 100:
                    model.progress = 100
                    model.status = "completed"
                    model.time_remaining = "تکمیل شده"
                    
                    await self.add_notification(
                        "success",
                        "آموزش تکمیل شد",
                        f"مدل {model.name} با دقت {model.accuracy:.1f}% آموزش داده شد"
                    )
                    
                    await self.add_log("SUCCESS", f"آموزش مدل {model.name} تکمیل شد", "trainer")

    async def send_system_metrics(self):
        """ارسال معیارهای سیستم به کلاینت‌ها"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            metrics = {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "gpu_usage": random.uniform(10, 95),
                "temperature": random.uniform(35, 65),
                "power_consumption": random.uniform(100, 250),
                "timestamp": datetime.now().isoformat()
            }
            
            await self.broadcast_to_clients({
                "type": "metrics_update",
                "data": metrics
            })
            
        except Exception as e:
            logger.error(f"خطا در ارسال معیارها: {e}")

    async def simulate_random_event(self):
        """شبیه‌سازی رویدادهای تصادفی"""
        events = [
            ("info", "جمع‌آوری داده", f"جمع‌آوری {random.randint(100, 500)} سند جدید"),
            ("warning", "هشدار سیستم", f"استفاده از {random.choice(['CPU', 'حافظه'])} بالا است"),
            ("success", "ذخیره مدل", "checkpoint جدید ذخیره شد"),
            ("info", "اتصال منبع", f"اتصال به {random.choice(['پیکره نعب', 'پورتال قوانین'])} برقرار شد")
        ]
        
        event_type, title, message = random.choice(events)
        await self.add_notification(event_type, title, message)

def main():
    """تابع اصلی"""
    backend = PersianAIBackend()
    
    # تنظیم سرور
    config = uvicorn.Config(
        app=backend.app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
    
    server = uvicorn.Server(config)
    
    print("🚀 Persian Legal AI Backend Server")
    print("=" * 50)
    print("📊 Dashboard API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔌 WebSocket: ws://localhost:8000/ws")
    print("=" * 50)
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n🛑 سرور توسط کاربر متوقف شد")
    except Exception as e:
        print(f"❌ خطا در اجرای سرور: {e}")

if __name__ == "__main__":
    main()