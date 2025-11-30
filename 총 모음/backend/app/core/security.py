"""
API   
API     
"""

import re
from typing import Optional
from fastapi import HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from backend.app.models.user import User
from backend.app.core.database import get_db

# API   
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class SecurityValidator:
    """  """

    # XSS  
    XSS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'onerror\s*=',
        r'onclick\s*=',
        r'onload\s*=',
    ]

    # SQL Injection 
    SQL_INJECTION_PATTERNS = [
        r"(\s|^)(DROP|DELETE|INSERT|UPDATE|SELECT|UNION|ALTER|CREATE|TRUNCATE)\s",
        r"--",
        r";.*?(\s|$)",
        r"'.*?OR.*?'",
        r'".*?OR.*?"',
    ]

    @classmethod
    def validate_text_input(cls, text: str, max_length: int = 5000) -> bool:
        """   - XSS SQL Injection """
        #  
        if not text or len(text) == 0:
            raise HTTPException(status_code=400, detail=" ")

        if len(text) > max_length:
            raise HTTPException(
                status_code=400,
                detail=f"   ( {max_length})"
            )

        # XSS 
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                raise HTTPException(
                    status_code=400,
                    detail="XSS   "
                )

        # SQL Injection 
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                raise HTTPException(
                    status_code=400,
                    detail="SQL Injection  "
                )

        return True

    @classmethod
    def validate_api_key_format(cls, api_key: str) -> bool:
        """API     (ked_  )"""
        if not api_key:
            return False

        # ked_  
        if not api_key.startswith("ked_"):
            raise HTTPException(
                status_code=400,
                detail="API    - ked_ "
            )

        #  
        if len(api_key) < 40 or len(api_key) > 60:
            raise HTTPException(
                status_code=400,
                detail="API   "
            )

        #    
        token_part = api_key[4:]
        if not re.match(r'^[A-Za-z0-9_-]+$', token_part):
            raise HTTPException(
                status_code=400,
                detail="API    "
            )

        return True

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """  - HTML    """
        # HTML  
        text = re.sub(r'<[^>]+>', '', text)

        #    
        text = re.sub(r'\s+', ' ', text)

        #   
        text = text.strip()

        return text


async def verify_api_key(
    api_key: str = Security(api_key_header),
    request: Request = None,
    db: Session = None
) -> User:
    """
    API   
    FastAPI Dependency 
    """
    # API   
    if not api_key:
        raise HTTPException(
            status_code=403,
            detail="API   (X-API-Key )"
        )

    # API   
    SecurityValidator.validate_api_key_format(api_key)

    # DB  
    if db is None:
        db = next(get_db())

    # DB  
    user = db.query(User).filter(
        User.api_key == api_key
    ).first()

    if not user:
        raise HTTPException(
            status_code=403,
            detail="  API "
        )

    #    
    if not user.is_active:
        raise HTTPException(
            status_code=401,
            detail=" "
        )

    # Rate limit 
    if user.is_rate_limited():
        raise HTTPException(
            status_code=429,
            detail=f"  .  {user.request_limit} , "
                   f" {user.request_count} "
        )

    #   
    user.increment_request_count()
    db.commit()

    return user


async def verify_api_key_optional(
    api_key: Optional[str] = Security(api_key_header),
    db: Session = None
) -> Optional[User]:
    """  - API   ,  """
    if not api_key:
        return None

    try:
        return await verify_api_key(api_key=api_key, db=db)
    except HTTPException:
        return None


def get_client_ip(request: Request) -> str:
    """ IP   (  )"""
    # X-Forwarded-For  
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    # X-Real-IP  
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    #  
    return request.client.host if request.client else "unknown"


def get_user_agent(request: Request) -> str:
    """User-Agent """
    return request.headers.get("User-Agent", "unknown")


class RateLimitExceeded(Exception):
    """Rate limit  """
    pass


class InvalidAPIKey(Exception):
    """  API  """
    pass


class InactiveAccount(Exception):
    """  """
    pass
