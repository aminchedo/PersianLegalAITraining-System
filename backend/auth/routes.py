"""
Authentication Routes for Persian Legal AI
مسیرهای احراز هویت برای هوش مصنوعی حقوقی فارسی
"""

import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from .jwt_handler import jwt_handler, User, Token
from .dependencies import get_current_user, TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: User

class UserResponse(BaseModel):
    username: str
    email: str
    full_name: str
    permissions: list

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login endpoint"""
    try:
        # Authenticate user
        user = jwt_handler.authenticate_user(request.username, request.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        token = jwt_handler.create_user_token(user)
        
        # Create user response (without password)
        user_response = User(
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            permissions=user.permissions
        )
        
        logger.info(f"User {request.username} logged in successfully")
        
        return LoginResponse(
            access_token=token.access_token,
            token_type=token.token_type,
            expires_in=token.expires_in,
            user=user_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """Get current user information"""
    try:
        # Get user from database
        user = jwt_handler.get_user(current_user.username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            permissions=user.permissions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user info failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user information"
        )

@router.post("/refresh")
async def refresh_token(current_user: TokenData = Depends(get_current_user)):
    """Refresh access token"""
    try:
        # Get user from database
        user = jwt_handler.get_user(current_user.username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Create new access token
        token = jwt_handler.create_user_token(user)
        
        logger.info(f"Token refreshed for user {current_user.username}")
        
        return {
            "access_token": token.access_token,
            "token_type": token.token_type,
            "expires_in": token.expires_in
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.post("/logout")
async def logout(current_user: TokenData = Depends(get_current_user)):
    """Logout endpoint (client should discard token)"""
    logger.info(f"User {current_user.username} logged out")
    
    return {
        "message": "Successfully logged out",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/users")
async def list_users(current_user: TokenData = Depends(get_current_user)):
    """List all users (admin only)"""
    if "admin" not in current_user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permission required"
        )
    
    try:
        users = []
        for username, user_data in jwt_handler.users_db.items():
            users.append({
                "username": user_data["username"],
                "email": user_data["email"],
                "full_name": user_data["full_name"],
                "is_active": user_data["is_active"],
                "permissions": user_data["permissions"]
            })
        
        return {
            "users": users,
            "total": len(users)
        }
        
    except Exception as e:
        logger.error(f"List users failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )