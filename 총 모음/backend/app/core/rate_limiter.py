"""
Rate Limiting Module - API  

    :
- IP  rate limiting
- (API )  rate limiting
-   
- Redis   rate limiting
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException
from typing import Callable
import hashlib

# Global limiter instance
limiter = Limiter(key_func=get_remote_address)


def get_api_key_from_request(request: Request) -> str:
    """
     API   (Rate limiting )

    Args:
        request: FastAPI Request 

    Returns:
        str: API   IP 
    """
    api_key = request.headers.get("X-API-Key")

    if api_key:
        # API    
        return f"apikey:{hashlib.md5(api_key.encode()).hexdigest()}"

    # API   IP  
    return f"ip:{get_remote_address(request)}"


def get_user_id_from_request(request: Request) -> str:
    """
      ID  (state)

    Args:
        request: FastAPI Request 

    Returns:
        str:  ID  IP 
    """
    # FastAPI request.state user  
    if hasattr(request.state, "user") and request.state.user:
        return f"user:{request.state.user.id}"

    #    API   IP 
    return get_api_key_from_request(request)


# Rate limit  

#   ( )
default_limit = "100/minute"

#   (  )
analysis_limit = "30/minute"

#    (   )
batch_limit = "10/minute"

# WebSocket  
websocket_limit = "5/minute"

#  API 
admin_limit = "50/minute"


class RateLimiterConfig:
    """Rate limiter  """

    #   
    ENDPOINT_LIMITS = {
        "/api/analyze": "30/minute",
        "/api/batch": "10/minute",
        "/ws/realtime": "5/minute",
        "/api/stats": "100/minute",
        "/health": "1000/minute",  # Health check  
    }

    #  tier 
    USER_TIER_LIMITS = {
        "free": {
            "per_minute": 30,
            "per_hour": 1000,
            "per_day": 10000,
        },
        "basic": {
            "per_minute": 60,
            "per_hour": 3000,
            "per_day": 50000,
        },
        "premium": {
            "per_minute": 120,
            "per_hour": 10000,
            "per_day": 200000,
        },
        "enterprise": {
            "per_minute": 300,
            "per_hour": 50000,
            "per_day": 1000000,
        }
    }

    @classmethod
    def get_limit_for_endpoint(cls, endpoint: str) -> str:
        """
            

        Args:
            endpoint: API  

        Returns:
            str: Rate limit  (: "30/minute")
        """
        return cls.ENDPOINT_LIMITS.get(endpoint, default_limit)

    @classmethod
    def get_limit_for_user_tier(cls, tier: str, period: str = "per_minute") -> int:
        """
         tier   

        Args:
            tier:  tier (free, basic, premium, enterprise)
            period:   (per_minute, per_hour, per_day)

        Returns:
            int:  
        """
        tier_config = cls.USER_TIER_LIMITS.get(tier, cls.USER_TIER_LIMITS["free"])
        return tier_config.get(period, 30)


def create_custom_limiter(
    key_func: Callable = None,
    default_limits: list = None
) -> Limiter:
    """
     limiter 

    Args:
        key_func:    (: get_remote_address)
        default_limits:   

    Returns:
        Limiter:  limiter 
    """
    if key_func is None:
        key_func = get_api_key_from_request

    if default_limits is None:
        default_limits = [default_limit]

    return Limiter(
        key_func=key_func,
        default_limits=default_limits
    )


def handle_rate_limit_exceeded(request: Request, exc: RateLimitExceeded):
    """
    Rate limit   

    Args:
        request: FastAPI Request 
        exc: RateLimitExceeded 

    Raises:
        HTTPException: 429  with  
    """
    #   
    retry_after = exc.detail.split("Retry after ")[1] if "Retry after" in exc.detail else "unknown"

    raise HTTPException(
        status_code=429,
        detail={
            "error": "Rate limit exceeded",
            "message": f"Too many requests. Please try again later.",
            "retry_after": retry_after,
            "endpoint": str(request.url.path),
            "limit": exc.detail
        }
    )


# IP  ( IP )
IP_BLACKLIST = set()


def is_ip_blacklisted(ip: str) -> bool:
    """
    IP  

    Args:
        ip:  IP 

    Returns:
        bool:   True
    """
    return ip in IP_BLACKLIST


def add_to_blacklist(ip: str, reason: str = "abuse"):
    """
    IP  

    Args:
        ip:  IP 
        reason:  
    """
    IP_BLACKLIST.add(ip)
    print(f"[SECURITY] IP {ip} added to blacklist. Reason: {reason}")


def remove_from_blacklist(ip: str):
    """
    IP  

    Args:
        ip:  IP 
    """
    if ip in IP_BLACKLIST:
        IP_BLACKLIST.remove(ip)
        print(f"[SECURITY] IP {ip} removed from blacklist")


# Rate limit  ( ,  Redis  )
rate_limit_stats = {}


def track_rate_limit_hit(identifier: str, endpoint: str):
    """
    Rate limit  

    Args:
        identifier: /IP 
        endpoint: API 
    """
    key = f"{identifier}:{endpoint}"

    if key not in rate_limit_stats:
        rate_limit_stats[key] = {
            "hits": 0,
            "first_hit": None,
            "last_hit": None
        }

    from datetime import datetime
    now = datetime.utcnow()

    rate_limit_stats[key]["hits"] += 1
    rate_limit_stats[key]["last_hit"] = now

    if rate_limit_stats[key]["first_hit"] is None:
        rate_limit_stats[key]["first_hit"] = now


def get_rate_limit_stats(identifier: str = None) -> dict:
    """
    Rate limit  

    Args:
        identifier:   (None  )

    Returns:
        dict:  
    """
    if identifier:
        return {k: v for k, v in rate_limit_stats.items() if k.startswith(identifier)}

    return rate_limit_stats


def reset_rate_limit_stats():
    """Rate limit  """
    global rate_limit_stats
    rate_limit_stats = {}


# FastAPI    

def setup_rate_limiter(app):
    """
    FastAPI  rate limiter 

    Usage:
        ```python
        from fastapi import FastAPI
        from app.core.rate_limiter import setup_rate_limiter, limiter

        app = FastAPI()
        setup_rate_limiter(app)
        ```

    Args:
        app: FastAPI  
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, handle_rate_limit_exceeded)

    print("[RATE LIMITER] Rate limiting enabled")
    print(f"[RATE LIMITER] Default limit: {default_limit}")
    print(f"[RATE LIMITER] Analysis limit: {analysis_limit}")
    print(f"[RATE LIMITER] Batch limit: {batch_limit}")
