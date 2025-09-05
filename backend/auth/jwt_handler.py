"""
JWT Authentication Handler for Persian Legal AI
مدیر احراز هویت JWT برای هوش مصنوعی حقوقی فارسی
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "persian_ai_jwt_secret_key_2024_very_secure")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None
    permissions: Optional[list] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class User(BaseModel):
    username: str
    email: str
    full_name: str
    is_active: bool = True
    permissions: list = []

class UserInDB(User):
    hashed_password: str

class JWTHandler:
    """JWT Authentication Handler"""
    
    def __init__(self):
        self.secret_key = JWT_SECRET_KEY
        self.algorithm = JWT_ALGORITHM
        self.access_token_expire_minutes = JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        
        # Default users (in production, use database)
        self.users_db = {
            "admin": {
                "username": "admin",
                "email": "admin@persian-legal-ai.com",
                "full_name": "System Administrator",
                "hashed_password": self.get_password_hash("admin123"),
                "is_active": True,
                "permissions": ["admin", "training", "model", "system"]
            },
            "trainer": {
                "username": "trainer",
                "email": "trainer@persian-legal-ai.com",
                "full_name": "Model Trainer",
                "hashed_password": self.get_password_hash("trainer123"),
                "is_active": True,
                "permissions": ["training", "model"]
            },
            "viewer": {
                "username": "viewer",
                "email": "viewer@persian-legal-ai.com",
                "full_name": "System Viewer",
                "hashed_password": self.get_password_hash("viewer123"),
                "is_active": True,
                "permissions": ["view"]
            }
        }
        
        logger.info("JWT Handler initialized")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def get_user(self, username: str) -> Optional[UserInDB]:
        """Get user by username"""
        if username in self.users_db:
            user_dict = self.users_db[username]
            return UserInDB(**user_dict)
        return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate user with username and password"""
        user = self.get_user(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return user
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify JWT token and return token data"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            permissions: list = payload.get("permissions", [])
            
            if username is None:
                return None
            
            token_data = TokenData(
                username=username,
                user_id=user_id,
                permissions=permissions
            )
            return token_data
        except JWTError as e:
            logger.error(f"JWT verification failed: {e}")
            return None
    
    def get_token_expiration(self) -> int:
        """Get token expiration time in seconds"""
        return self.access_token_expire_minutes * 60
    
    def create_user_token(self, user: UserInDB) -> Token:
        """Create access token for user"""
        access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
        access_token = self.create_access_token(
            data={
                "sub": user.username,
                "user_id": user.username,
                "permissions": user.permissions
            },
            expires_delta=access_token_expires
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=self.get_token_expiration()
        )

# Global JWT handler instance
jwt_handler = JWTHandler()

def get_jwt_handler() -> JWTHandler:
    """Get JWT handler instance"""
    return jwt_handler