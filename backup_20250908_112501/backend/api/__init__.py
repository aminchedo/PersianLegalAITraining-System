"""
API Package for Persian Legal AI
بسته API برای هوش مصنوعی حقوقی فارسی
"""

from .training_endpoints import router as training_router
from .model_endpoints import router as model_router
from .system_endpoints import router as system_router

__all__ = ['training_router', 'model_router', 'system_router']