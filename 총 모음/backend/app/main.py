"""
FastAPI  
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging

from app.core.config import settings
from app.api.routes import analyze, websocket, labeling
from app.api.models.response import HealthResponse

# Import security modules
from app.core.rate_limiter import setup_rate_limiter, limiter
from app.core.middleware import setup_middleware

#  
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI  
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="    API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "analysis", "description": " "},
        {"name": "health", "description": " "},
        {"name": "websocket", "description": " "},
        {"name": "labeling", "description": "   "}
    ]
)

# CORS 
allowed_origins = settings.CORS_ORIGINS

#    origin 
if settings.DEBUG:
    allowed_origins = ["*"]
    logger.warning("[]   -  origin ")
else:
    logger.info(f"[] CORS : {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    max_age=3600,
)

#   
setup_middleware(app)

# Rate limiter 
setup_rate_limiter(app)

#  
app.include_router(analyze.router, prefix=settings.API_PREFIX, tags=["analysis"])
app.include_router(websocket.router, tags=["websocket"])
app.include_router(labeling.router, prefix=settings.API_PREFIX, tags=["labeling"])

logger.info("[]   ")
logger.info("[]    ")


@app.on_event("startup")
async def startup_event():
    """  """
    logger.info("="*60)
    logger.info(f"[] K-Euphemism Detector API v{settings.API_VERSION}")
    logger.info(f"[] : {'' if settings.DEBUG else ''}")
    logger.info(f"[] DB: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else ''}")
    logger.info(f"[] Redis: {settings.REDIS_URL.split('@')[1] if '@' in settings.REDIS_URL else ''}")
    logger.info("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """  """
    logger.info("[] DB   ...")
    logger.info("[] API  ")


@app.get("/", tags=["health"])
@limiter.limit("100/minute")
async def root(request: Request):
    """ """
    return {
        "message": "K-Euphemism Detector API",
        "version": settings.API_VERSION,
        "status": "operational",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "security": {
            "authentication": "API Key (X-API-Key header)",
            "rate_limiting": "enabled",
            "cors": "configured"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
@limiter.limit("1000/minute")
async def health_check(request: Request):
    """  (/Cloud Run)"""
    from app.ml.detector import EuphemismDetector
    from app.core.database import engine
    from app.core.cache import redis_client

    #  
    health_info = {
        "status": "healthy",
        "version": settings.API_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        # ML   
        detector = EuphemismDetector()
        health_info["model_loaded"] = detector.is_loaded()
        health_info["trained_model"] = detector.ml_detector._using_trained_model if hasattr(detector.ml_detector, '_using_trained_model') else False
        health_info["gemini_enabled"] = detector.gemini_detector is not None
    except Exception as e:
        logger.error(f"  : {e}")
        health_info["model_loaded"] = False
        health_info["trained_model"] = False
        health_info["gemini_enabled"] = False
        health_info["status"] = "degraded"

    try:
        # DB  
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        health_info["database"] = "connected"
    except Exception as e:
        logger.error(f"DB  : {e}")
        health_info["database"] = "disconnected"
        health_info["status"] = "degraded"

    try:
        # Redis/ 
        from app.core.cache import cache
        redis_client.ping()
        if cache._connected:
            health_info["cache"] = "redis_connected"
        else:
            health_info["cache"] = "in_memory_fallback"
        #     degraded 
    except Exception as e:
        logger.warning(f" : {e}")
        health_info["cache"] = "in_memory_only"
        #   API   degraded 

    return HealthResponse(**health_info)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """  """
    #  
    logger.error(
        f"  : {type(exc).__name__}: {str(exc)} "
        f": {request.method} {request.url.path}"
    )

    #    
    error_detail = str(exc) if settings.DEBUG else "  "

    return JSONResponse(
        status_code=500,
        content={
            "error": " ",
            "detail": error_detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path,
            "method": request.method
        }
    )


@app.get("/api/info", tags=["health"])
@limiter.limit("100/minute")
async def api_info(request: Request):
    """API """
    return {
        "api": {
            "name": "K-Euphemism Detector",
            "version": settings.API_VERSION,
            "description": "Korean euphemistic expression detection and resolution API"
        },
        "features": {
            "pattern_detection": "Rule-based pattern matching",
            "ml_detection": "KoELECTRA NER-based detection",
            "entity_resolution": "Context-aware entity resolution",
            "batch_processing": "Batch analysis support",
            "real_time": "WebSocket streaming support"
        },
        "limits": {
            "max_text_length": settings.MAX_TEXT_LENGTH,
            "rate_limit_default": "30/minute",
            "batch_limit": "100 texts per request"
        },
        "authentication": {
            "type": "API Key",
            "header": "X-API-Key",
            "format": "ked_<token>"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("[STARTUP] Starting server...")

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
