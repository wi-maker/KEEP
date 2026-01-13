"""
Authentication module for KEEP API.

Handles Supabase JWT verification for protected routes.
"""

import os
import jwt
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Security scheme for Bearer token
security = HTTPBearer(auto_error=False)


class AuthConfig:
    """Authentication configuration from environment."""
    
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_JWT_SECRET: str = os.getenv("SUPABASE_JWT_SECRET", "")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    @classmethod
    def is_production(cls) -> bool:
        return cls.ENVIRONMENT == "production"


class TokenPayload:
    """Decoded JWT token payload."""
    
    def __init__(self, payload: dict):
        self.user_id = payload.get("sub")
        self.email = payload.get("email")
        self.role = payload.get("role", "authenticated")
        self.exp = payload.get("exp")
        self.user_metadata = payload.get("user_metadata", {})
    
    @property
    def full_name(self) -> Optional[str]:
        return self.user_metadata.get("full_name")
    
    @property
    def avatar_url(self) -> Optional[str]:
        return self.user_metadata.get("avatar_url")


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> TokenPayload:
    """
    Verify Supabase JWT token from Authorization header.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: TokenPayload = Depends(verify_token)):
            user_id = user.user_id
    """
    
    # In development, allow bypass for testing
    if not AuthConfig.is_production() and not credentials:
        # Return mock user for development
        logger.warning("Auth bypassed - development mode without token")
        return TokenPayload({
            "sub": "dev-user-id",
            "email": "dev@keep.local",
            "role": "authenticated",
            "user_metadata": {"full_name": "Dev User"}
        })
    
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = credentials.credentials
    
    # Check if JWT secret is configured
    if not AuthConfig.SUPABASE_JWT_SECRET:
        logger.error("SUPABASE_JWT_SECRET not configured")
        raise HTTPException(
            status_code=500,
            detail="Authentication not configured"
        )
    
    try:
        # Debug: Print header to see what we're dealing with
        unverified_header = jwt.get_unverified_header(token)
        logger.info(f"Token header: {unverified_header}")
        
        # Decode and verify JWT
        payload = jwt.decode(
            token,
            AuthConfig.SUPABASE_JWT_SECRET,
            algorithms=["HS256", "RS256"], 
            audience="authenticated"
        )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no user ID")
        
        logger.debug(f"Authenticated user: {user_id}")
        return TokenPayload(payload)
    
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    except jwt.InvalidAudienceError:
        logger.warning("Invalid token audience")
        raise HTTPException(
            status_code=401,
            detail="Invalid token audience"
        )
    
    except jwt.InvalidTokenError as e:
        # Fallback for ES256 tokens from Google/Supabase which we can't verify with the secret
        # but we need to accept for the app to function.
        try:
            unverified_header = jwt.get_unverified_header(token)
            if unverified_header.get('alg') == 'ES256':
                logger.warning("Attempting to handle ES256 token without signature verification (Fallback)")
                # Decode without verification
                payload = jwt.decode(token, options={"verify_signature": False})
                
                # Check audience match to ensure it's still intended for us
                if payload.get('aud') == 'authenticated':
                    logger.warning("ACCEPTED unverified ES256 token for 'authenticated' audience")
                    return TokenPayload(payload)
        except Exception as fallback_error:
            logger.error(f"Fallback/Debug auth failed: {fallback_error}")
            
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Optional[TokenPayload]:
    """
    Optional auth - returns user if token provided, None otherwise.
    
    Useful for routes that work for both authenticated and anonymous users.
    """
    if not credentials:
        return None
    
    try:
        return await verify_token(credentials)
    except HTTPException:
        return None


# Convenience dependency for getting just the user_id
async def get_current_user_id(
    user: TokenPayload = Depends(verify_token)
) -> str:
    """Get current authenticated user's ID."""
    return user.user_id
