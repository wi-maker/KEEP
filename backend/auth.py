import os
import jwt
import json
import time
import urllib.request
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Security scheme for Bearer token
security = HTTPBearer(auto_error=False)

# ── JWKS cache ────────────────────────────────────────────────────────
_jwks_cache: dict = {"keys": [], "fetched_at": 0}
_JWKS_CACHE_TTL = 3600  # 1 hour


def _get_jwks_url() -> str:
    """Build the JWKS URL from the Supabase project URL."""
    base = AuthConfig.SUPABASE_URL.rstrip("/")
    return f"{base}/auth/v1/.well-known/jwks.json"


def _fetch_jwks() -> list:
    """Fetch JWKS keys from Supabase, with caching."""
    now = time.time()
    if _jwks_cache["keys"] and (now - _jwks_cache["fetched_at"]) < _JWKS_CACHE_TTL:
        return _jwks_cache["keys"]
    
    try:
        url = _get_jwks_url()
        logger.info(f"Fetching JWKS from {url}")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        
        _jwks_cache["keys"] = data.get("keys", [])
        _jwks_cache["fetched_at"] = now
        logger.info(f"Fetched {len(_jwks_cache['keys'])} JWKS key(s)")
        return _jwks_cache["keys"]
    except Exception as e:
        logger.error(f"Failed to fetch JWKS: {e}")
        # Return stale cache if available
        if _jwks_cache["keys"]:
            logger.warning("Using stale JWKS cache")
            return _jwks_cache["keys"]
        return []


def _get_public_key_for_kid(kid: str):
    """Find the public key matching the given Key ID from JWKS."""
    keys = _fetch_jwks()
    for key_data in keys:
        if key_data.get("kid") == kid:
            return jwt.algorithms.ECAlgorithm.from_jwk(json.dumps(key_data))
    return None


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
        self.phone_claim = payload.get("phone")
    
    @property
    def full_name(self) -> Optional[str]:
        return self.user_metadata.get("full_name")
    
    @property
    def avatar_url(self) -> Optional[str]:
        return self.user_metadata.get("avatar_url")
    
    @property
    def phone(self) -> Optional[str]:
        """Get phone number from user_metadata or direct claim."""
        return self.user_metadata.get("phone") or self.phone_claim


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> TokenPayload:
    """
    Verify Supabase JWT token from Authorization header.
    
    Supports both HS256 (symmetric, uses SUPABASE_JWT_SECRET) and
    ES256 (asymmetric, fetches public key from Supabase JWKS endpoint).
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: TokenPayload = Depends(verify_token)):
            user_id = user.user_id
    """
    
    # In development, allow bypass for testing
    if not AuthConfig.is_production() and not credentials:
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
    
    try:
        # Read the token header to determine algorithm and key ID
        unverified_header = jwt.get_unverified_header(token)
        alg = unverified_header.get("alg", "HS256")
        kid = unverified_header.get("kid")
        logger.info(f"Token header: alg={alg}, kid={kid}")
        
        # ── Choose the right key based on the algorithm ──
        if alg == "ES256":
            # Asymmetric: fetch the public key from Supabase JWKS
            if not kid:
                raise jwt.InvalidTokenError("ES256 token missing 'kid' header")
            
            public_key = _get_public_key_for_kid(kid)
            if not public_key:
                # Force refresh cache and retry once
                _jwks_cache["fetched_at"] = 0
                public_key = _get_public_key_for_kid(kid)
            
            if not public_key:
                logger.error(f"No JWKS key found for kid={kid}")
                raise jwt.InvalidTokenError(f"Unknown signing key: {kid}")
            
            signing_key = public_key
            algorithms = ["ES256"]
        
        else:
            # HS256 / RS256: use the shared secret
            if not AuthConfig.SUPABASE_JWT_SECRET:
                logger.error("SUPABASE_JWT_SECRET not configured")
                raise HTTPException(status_code=500, detail="Authentication not configured")
            
            signing_key = AuthConfig.SUPABASE_JWT_SECRET
            algorithms = ["HS256", "RS256"]
        
        # Decode and verify
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=algorithms,
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
        logger.error(f"JWT verification failed: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
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


# Admin verification dependency (requires db session)
def get_current_admin(db_dependency):
    """
    Factory function to create admin dependency.
    
    Usage in router:
        from db import get_db
        get_admin = get_current_admin(get_db)
        
        @router.get("/admin/endpoint")
        async def admin_endpoint(user: TokenPayload = Depends(get_admin)):
            ...
    """
    async def verify_admin(
        user: TokenPayload = Depends(verify_token),
        db = Depends(db_dependency)
    ) -> TokenPayload:
        """Verify current user is an admin. Raises 403 if not."""
        from models import User
        
        db_user = db.query(User).filter_by(id=user.user_id).first()
        if not db_user or not db_user.is_admin:
            raise HTTPException(
                status_code=403,
                detail="Admin privileges required"
            )
        return user
    
    return verify_admin

