"""
  
 , IP ,  
"""

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.gzip import GZipMiddleware
from typing import Callable
import time
import logging
from datetime import datetime

from app.core.rate_limiter import is_ip_blacklisted, track_rate_limit_hit

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """   """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        #   
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # HTTPS 
        if not request.url.hostname.startswith("localhost"):
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # CSP 
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self' data:; "
            "connect-src 'self'"
        )

        response.headers["X-Powered-By"] = "K-Euphemism-Detector"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=()"
        )

        return response


class IPBlacklistMiddleware(BaseHTTPMiddleware):
    """ IP """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # IP  
        client_ip = request.client.host if request.client else "unknown"

        #   
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        #  
        if is_ip_blacklisted(client_ip):
            logger.warning(f" IP : {client_ip}")
            return JSONResponse(
                status_code=403,
                content={
                    "error": " ",
                    "message": " IP ",
                    "ip": client_ip
                }
            )

        response = await call_next(request)
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """  """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        #   
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        url = str(request.url)

        # 
        logger.info(f": {method} {url} from {client_ip}")

        #  
        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            #  
            logger.info(
                f": {response.status_code} ({process_time:.3f}s)"
            )

            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f" : {method} {url} - {str(e)} ({process_time:.3f}s)"
            )
            raise


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """  -   """

    SLOW_REQUEST_THRESHOLD = 1.0  # 1  

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        #   
        if process_time > self.SLOW_REQUEST_THRESHOLD:
            logger.warning(
                f" : {request.method} {request.url.path} "
                f"  {process_time:.3f}"
            )

        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """   (DoS )"""

    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")

        if content_length:
            content_length = int(content_length)

            if content_length > self.MAX_REQUEST_SIZE:
                logger.warning(f"  : {content_length} bytes")
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "  ",
                        "message": f"   ( 10MB)",
                        "max_size": self.MAX_REQUEST_SIZE
                    }
                )

        response = await call_next(request)
        return response


class RateLimitTrackingMiddleware(BaseHTTPMiddleware):
    """Rate limit  """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        api_key = request.headers.get("X-API-Key")
        client_ip = request.client.host if request.client else "unknown"

        identifier = f"apikey:{api_key}" if api_key else f"ip:{client_ip}"
        track_rate_limit_hit(identifier, request.url.path)

        response = await call_next(request)
        return response


def setup_middleware(app):
    """  """
    # GZip 
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    #  
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(IPBlacklistMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware)
    app.add_middleware(RateLimitTrackingMiddleware)
    app.add_middleware(PerformanceMonitoringMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    logger.info("[]     ")
