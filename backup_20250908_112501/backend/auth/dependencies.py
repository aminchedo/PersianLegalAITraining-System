"""
Authentication Dependencies for Persian Legal AI
وابستگی‌های احراز هویت برای هوش مصنوعی حقوقی فارسی
"""

import logging
from typing import Optional, List
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .jwt_handler import jwt_handler, TokenData

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Get current authenticated user from JWT token"""
    token = credentials.credentials
    token_data = jwt_handler.verify_token(token)
    
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return token_data

def get_current_active_user(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """Get current active user"""
    # In a real application, you would check if the user is active in the database
    # For now, we assume all users with valid tokens are active
    return current_user

def require_permission(required_permission: str):
    """Decorator to require specific permission"""
    def permission_dependency(current_user: TokenData = Depends(get_current_active_user)) -> TokenData:
        if required_permission not in current_user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{required_permission}' required"
            )
        return current_user
    return permission_dependency

def require_any_permission(required_permissions: List[str]):
    """Decorator to require any of the specified permissions"""
    def permission_dependency(current_user: TokenData = Depends(get_current_active_user)) -> TokenData:
        if not any(perm in current_user.permissions for perm in required_permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of the following permissions required: {', '.join(required_permissions)}"
            )
        return current_user
    return permission_dependency

def require_admin_permission(current_user: TokenData = Depends(get_current_active_user)) -> TokenData:
    """Require admin permission"""
    if "admin" not in current_user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permission required"
        )
    return current_user

def require_training_permission(current_user: TokenData = Depends(get_current_active_user)) -> TokenData:
    """Require training permission"""
    if "training" not in current_user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Training permission required"
        )
    return current_user

def require_model_permission(current_user: TokenData = Depends(get_current_active_user)) -> TokenData:
    """Require model permission"""
    if "model" not in current_user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Model permission required"
        )
    return current_user

def require_system_permission(current_user: TokenData = Depends(get_current_active_user)) -> TokenData:
    """Require system permission"""
    if "system" not in current_user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="System permission required"
        )
    return current_user

def optional_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[TokenData]:
    """Optional authentication - returns user if token is valid, None otherwise"""
    if credentials is None:
        return None
    
    try:
        token = credentials.credentials
        token_data = jwt_handler.verify_token(token)
        return token_data
    except Exception as e:
        logger.warning(f"Optional auth failed: {e}")
        return None