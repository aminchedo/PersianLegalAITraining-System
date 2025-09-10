"""
Redis Configuration for Persian Legal AI
تنظیمات Redis برای هوش مصنوعی حقوقی فارسی
"""

import os
import asyncio
import logging
from typing import Optional, Any, Dict
from datetime import datetime, timedelta
import json

import redis.asyncio as aioredis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

class RedisConfig:
    """Redis configuration and connection management"""
    
    def __init__(self):
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", "6379"))
        self.password = os.getenv("REDIS_PASSWORD")
        self.db = int(os.getenv("REDIS_DB", "0"))
        self.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
        self.socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
        self.socket_connect_timeout = int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
        
        # Connection URL
        if self.password:
            self.url = f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        else:
            self.url = f"redis://{self.host}:{self.port}/{self.db}"
        
        self._redis: Optional[Redis] = None
        self._connection_pool = None

    async def connect(self) -> Redis:
        """Establish Redis connection"""
        try:
            if self._redis is None:
                self._redis = await aioredis.from_url(
                    self.url,
                    max_connections=self.max_connections,
                    socket_timeout=self.socket_timeout,
                    socket_connect_timeout=self.socket_connect_timeout,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                
                # Test connection
                await self._redis.ping()
                logger.info(f"✅ Redis connected successfully to {self.host}:{self.port}")
                
            return self._redis
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self._redis = None
            raise

    async def disconnect(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Redis connection closed")

    async def get_client(self) -> Optional[Redis]:
        """Get Redis client, connecting if necessary"""
        try:
            if self._redis is None:
                await self.connect()
            return self._redis
        except Exception as e:
            logger.error(f"Failed to get Redis client: {e}")
            return None

    async def health_check(self) -> Dict[str, Any]:
        """Check Redis health"""
        try:
            start_time = datetime.now()
            client = await self.get_client()
            
            if client is None:
                return {
                    "status": "unhealthy",
                    "error": "Cannot connect to Redis",
                    "response_time": -1
                }
            
            # Test ping
            pong = await client.ping()
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Get info
            info = await client.info()
            
            return {
                "status": "healthy" if pong else "unhealthy",
                "response_time": response_time,
                "version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "keyspace": info.get("db0", {})
            }
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time": -1
            }

# Global Redis configuration instance
redis_config = RedisConfig()

class RedisCache:
    """Redis caching operations"""
    
    def __init__(self, redis_config: RedisConfig):
        self.redis_config = redis_config
        self.default_ttl = 3600  # 1 hour default TTL
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            client = await self.redis_config.get_client()
            if client is None:
                return None
            
            value = await client.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value.decode('utf-8') if isinstance(value, bytes) else value
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            client = await self.redis_config.get_client()
            if client is None:
                return False
            
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, ensure_ascii=False)
            else:
                serialized_value = str(value)
            
            ttl = ttl or self.default_ttl
            result = await client.setex(key, ttl, serialized_value)
            return bool(result)
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            client = await self.redis_config.get_client()
            if client is None:
                return False
            
            result = await client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            client = await self.redis_config.get_client()
            if client is None:
                return False
            
            result = await client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        try:
            client = await self.redis_config.get_client()
            if client is None:
                return False
            
            result = await client.expire(key, ttl)
            return bool(result)
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        try:
            client = await self.redis_config.get_client()
            if client is None:
                return 0
            
            keys = await client.keys(pattern)
            if keys:
                deleted = await client.delete(*keys)
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0
    
    # Specialized caching methods for Persian Legal AI
    
    async def cache_classification_result(self, text_hash: str, result: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache AI classification result"""
        key = f"classification:{text_hash}"
        return await self.set(key, result, ttl)
    
    async def get_classification_result(self, text_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached AI classification result"""
        key = f"classification:{text_hash}"
        return await self.get(key)
    
    async def cache_model_info(self, model_name: str, info: Dict[str, Any], ttl: int = 86400) -> bool:
        """Cache model information"""
        key = f"model_info:{model_name}"
        return await self.set(key, info, ttl)
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get cached model information"""
        key = f"model_info:{model_name}"
        return await self.get(key)
    
    async def cache_training_session(self, session_id: str, data: Dict[str, Any], ttl: int = 7200) -> bool:
        """Cache training session data"""
        key = f"training_session:{session_id}"
        return await self.set(key, data, ttl)
    
    async def get_training_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached training session data"""
        key = f"training_session:{session_id}"
        return await self.get(key)
    
    async def cache_user_session(self, user_id: str, session_data: Dict[str, Any], ttl: int = 1800) -> bool:
        """Cache user session data"""
        key = f"user_session:{user_id}"
        return await self.set(key, session_data, ttl)
    
    async def get_user_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user session data"""
        key = f"user_session:{user_id}"
        return await self.get(key)

# Global cache instance
cache = RedisCache(redis_config)

# Utility functions for easy access
async def get_cache_client() -> Optional[Redis]:
    """Get Redis client for direct operations"""
    return await redis_config.get_client()

async def init_redis() -> bool:
    """Initialize Redis connection"""
    try:
        await redis_config.connect()
        logger.info("Redis initialization completed")
        return True
    except Exception as e:
        logger.error(f"Redis initialization failed: {e}")
        return False

async def close_redis():
    """Close Redis connection"""
    await redis_config.disconnect()
    logger.info("Redis connection closed")

# Health check function for external use
async def redis_health_check() -> Dict[str, Any]:
    """External Redis health check"""
    return await redis_config.health_check()