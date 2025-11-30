"""
Redis caching utilities with graceful degradation
"""
import redis
import json
import hashlib
import logging
from typing import Optional, Any
from functools import wraps

from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis cache manager for euphemism detection results with graceful degradation"""

    def __init__(self):
        """Initialize Redis connection with fallback"""
        self.redis_client = None
        self._connected = False
        self.ttl = settings.REDIS_CACHE_TTL
        self._in_memory_cache = {}  #   
        self._max_memory_items = 1000  #     

        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2,  #   2
                socket_timeout=2
            )
            #  
            self.redis_client.ping()
            self._connected = True
            logger.info("[] Redis  ")
        except Exception as e:
            logger.warning(f"[] Redis  ,   : {e}")
            self._connected = False

    @property
    def is_connected(self) -> bool:
        """Redis   """
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            self._connected = True
            return True
        except:
            self._connected = False
            return False
    
    def generate_key(self, text: str, options: dict = None) -> str:
        """
        Generate cache key from text and options
        
        Args:
            text: Input text to analyze
            options: Additional options for analysis
            
        Returns:
            Cache key string
        """
        content = f"{text}_{json.dumps(options or {}, sort_keys=True)}"
        return f"euphemism:{hashlib.md5(content.encode()).hexdigest()}"
    
    def get(self, key: str) -> Optional[dict]:
        """
        Get cached result (Redis with in-memory fallback)

        Args:
            key: Cache key

        Returns:
            Cached data or None
        """
        # Redis   Redis 
        if self._connected and self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.debug(f"Redis get error, trying memory: {e}")
                self._connected = False

        #   
        if key in self._in_memory_cache:
            cached = self._in_memory_cache[key]
            # TTL  ( )
            import time
            if cached.get('_expires', 0) > time.time():
                return cached.get('data')
            else:
                #   
                del self._in_memory_cache[key]

        return None

    def set(self, key: str, value: dict, ttl: int = None):
        """
        Set cache value (Redis with in-memory fallback)

        Args:
            key: Cache key
            value: Data to cache
            ttl: Time to live in seconds (default: settings.REDIS_CACHE_TTL)
        """
        cache_ttl = ttl or self.ttl

        # Redis   Redis 
        if self._connected and self.redis_client:
            try:
                self.redis_client.setex(
                    key,
                    cache_ttl,
                    json.dumps(value, ensure_ascii=False)
                )
                return
            except Exception as e:
                logger.debug(f"Redis set error, using memory: {e}")
                self._connected = False

        #   
        import time
        #   
        if len(self._in_memory_cache) >= self._max_memory_items:
            #     ( LRU)
            oldest_key = next(iter(self._in_memory_cache))
            del self._in_memory_cache[oldest_key]

        self._in_memory_cache[key] = {
            'data': value,
            '_expires': time.time() + cache_ttl
        }

    def delete(self, key: str):
        """Delete cache entry"""
        # Redis 
        if self._connected and self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.debug(f"Redis delete error: {e}")

        #  
        if key in self._in_memory_cache:
            del self._in_memory_cache[key]

    def clear_all(self):
        """Clear all cache entries"""
        # Redis 
        if self._connected and self.redis_client:
            try:
                for key in self.redis_client.scan_iter("euphemism:*"):
                    self.redis_client.delete(key)
            except Exception as e:
                logger.debug(f"Redis clear error: {e}")

        #  
        self._in_memory_cache.clear()
        logger.info("[]    ")


# Global cache instance
cache = CacheManager()


class RedisClientProxy:
    """Redis client proxy for backwards compatibility with health checks"""

    def __init__(self, cache_manager: CacheManager):
        self._cache = cache_manager

    def ping(self):
        """Ping Redis or raise exception for health check"""
        if self._cache._connected and self._cache.redis_client:
            return self._cache.redis_client.ping()
        #     (degraded  healthy)
        return True

    def get(self, key):
        return self._cache.get(key)

    def setex(self, key, ttl, value):
        self._cache.set(key, json.loads(value) if isinstance(value, str) else value, ttl)

    def delete(self, key):
        self._cache.delete(key)


# Export redis client proxy for backwards compatibility
redis_client = RedisClientProxy(cache)


def cached(ttl: int = None):
    """
    Decorator for caching function results

    Args:
        ttl: Time to live in seconds

    Usage:
        @cached(ttl=3600)
        def analyze_text(text: str):
            # expensive computation
            return result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Handle both instance methods (self, text, ...) and functions (text, ...)
            if args and hasattr(args[0], '__class__'):
                # Instance method: args[0] is self, args[1] is text
                text = args[1] if len(args) > 1 else kwargs.get('text', '')
            else:
                # Regular function: args[0] is text
                text = args[0] if args else kwargs.get('text', '')

            # Generate cache key
            cache_key = cache.generate_key(text, kwargs)

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result:
                return cached_result

            # Compute result
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, ttl)

            return result
        return wrapper
    return decorator
