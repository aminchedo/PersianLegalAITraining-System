"""
Rate Limiting Middleware for Persian Legal AI
میان‌افزار محدودسازی نرخ برای هوش مصنوعی حقوقی فارسی
"""

import time
import logging
from typing import Dict, Optional
from collections import defaultdict, deque
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import asyncio

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter using sliding window algorithm"""
    
    def __init__(self):
        # Store request timestamps for each client
        self.requests: Dict[str, deque] = defaultdict(deque)
        # Rate limit configurations per endpoint
        self.limits: Dict[str, Dict[str, int]] = {
            "/api/auth/login": {"requests": 5, "window": 300},  # 5 requests per 5 minutes
            "/api/training/start": {"requests": 3, "window": 3600},  # 3 requests per hour
            "/api/training/stop": {"requests": 10, "window": 60},  # 10 requests per minute
            "/api/model/upload": {"requests": 2, "window": 3600},  # 2 requests per hour
            "/api/system/health": {"requests": 60, "window": 60},  # 60 requests per minute
            "default": {"requests": 100, "window": 60}  # 100 requests per minute
        }
        # Cleanup interval for old entries
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def get_rate_limit(self, path: str) -> Dict[str, int]:
        """Get rate limit configuration for a path"""
        # Check for exact path match first
        if path in self.limits:
            return self.limits[path]
        
        # Check for pattern matches
        for pattern, limit in self.limits.items():
            if pattern != "default" and path.startswith(pattern):
                return limit
        
        # Return default limit
        return self.limits["default"]
    
    def is_allowed(self, client_id: str, path: str) -> tuple[bool, Dict[str, int]]:
        """Check if request is allowed for client"""
        current_time = time.time()
        limit_config = self.get_rate_limit(path)
        max_requests = limit_config["requests"]
        window_seconds = limit_config["window"]
        
        # Clean up old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries(current_time, window_seconds)
            self.last_cleanup = current_time
        
        # Get or create request queue for this client
        client_requests = self.requests[client_id]
        
        # Remove requests outside the window
        cutoff_time = current_time - window_seconds
        while client_requests and client_requests[0] < cutoff_time:
            client_requests.popleft()
        
        # Check if under limit
        if len(client_requests) < max_requests:
            # Add current request
            client_requests.append(current_time)
            return True, {
                "limit": max_requests,
                "remaining": max_requests - len(client_requests),
                "reset_time": int(current_time + window_seconds)
            }
        else:
            # Rate limit exceeded
            return False, {
                "limit": max_requests,
                "remaining": 0,
                "reset_time": int(client_requests[0] + window_seconds) if client_requests else int(current_time + window_seconds)
            }
    
    def _cleanup_old_entries(self, current_time: float, window_seconds: int):
        """Clean up old entries to prevent memory leaks"""
        cutoff_time = current_time - window_seconds
        clients_to_remove = []
        
        for client_id, requests in self.requests.items():
            # Remove old requests
            while requests and requests[0] < cutoff_time:
                requests.popleft()
            
            # Remove empty client entries
            if not requests:
                clients_to_remove.append(client_id)
        
        for client_id in clients_to_remove:
            del self.requests[client_id]
    
    def get_client_id(self, request: Request) -> str:
        """Get client identifier from request"""
        # Try to get real IP from headers (for reverse proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        # Include user agent for additional uniqueness
        user_agent = request.headers.get("User-Agent", "unknown")
        
        return f"{client_ip}:{hash(user_agent) % 10000}"

# Global rate limiter instance
rate_limiter = RateLimiter()

async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    try:
        # Skip rate limiting for health checks and static files
        if request.url.path in ["/health", "/docs", "/openapi.json", "/favicon.ico"]:
            return await call_next(request)
        
        # Get client identifier
        client_id = rate_limiter.get_client_id(request)
        path = request.url.path
        
        # Check rate limit
        is_allowed, limit_info = rate_limiter.is_allowed(client_id, path)
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for client {client_id} on path {path}")
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "limit": limit_info["limit"],
                    "remaining": limit_info["remaining"],
                    "reset_time": limit_info["reset_time"]
                },
                headers={
                    "X-RateLimit-Limit": str(limit_info["limit"]),
                    "X-RateLimit-Remaining": str(limit_info["remaining"]),
                    "X-RateLimit-Reset": str(limit_info["reset_time"])
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(limit_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(limit_info["reset_time"])
        
        return response
        
    except Exception as e:
        logger.error(f"Rate limiting middleware error: {e}")
        # If rate limiting fails, allow the request to proceed
        return await call_next(request)

def get_rate_limiter() -> RateLimiter:
    """Get rate limiter instance"""
    return rate_limiter