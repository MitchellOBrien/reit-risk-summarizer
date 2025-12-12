"""Simple in-memory cache for REIT risk data.

This cache stores fetched 10-K HTML and extracted risk text to avoid
redundant SEC API calls and processing during a session.

The cache is purely in-memory and automatically cleared when the program exits.
For persistent caching across sessions, consider Redis or file-based storage.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MemoryCache:
    """In-memory cache using a Python dictionary.
    
    Thread-safe for single-threaded applications. For multi-threaded use,
    consider adding locks or using a thread-safe dict implementation.
    
    Examples:
        >>> cache = MemoryCache()
        >>> cache.set("AMT_10k_2024", "<html>...</html>")
        >>> html = cache.get("AMT_10k_2024")
        >>> cache.clear()
    """
    
    def __init__(self):
        """Initialize empty cache."""
        self._cache = {}
        logger.debug("Initialized MemoryCache")
    
    def get(self, key: str) -> Optional[str]:
        """Get cached value by key.
        
        Args:
            key: Cache key (e.g., "AMT_10k_2024")
            
        Returns:
            Cached value if exists, None otherwise
        """
        value = self._cache.get(key)
        if value is not None:
            logger.debug(f"Cache HIT: {key}")
        else:
            logger.debug(f"Cache MISS: {key}")
        return value
    
    def set(self, key: str, value: str) -> None:
        """Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache (typically HTML or extracted text)
        """
        self._cache[key] = value
        logger.debug(f"Cache SET: {key} ({len(value):,} chars)")
    
    def has(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        return key in self._cache
    
    def delete(self, key: str) -> bool:
        """Delete specific key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted, False if it didn't exist
        """
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Cache DELETE: {key}")
            return True
        return False
    
    def clear(self) -> None:
        """Clear entire cache."""
        count = len(self._cache)
        self._cache.clear()
        logger.debug(f"Cache CLEAR: removed {count} entries")
    
    def size(self) -> int:
        """Get number of cached items.
        
        Returns:
            Number of entries in cache
        """
        return len(self._cache)
    
    def keys(self):
        """Get all cache keys.
        
        Returns:
            View of all cache keys
        """
        return self._cache.keys()


# Global cache instance (shared across application)
_global_cache = None


def get_cache() -> MemoryCache:
    """Get or create global cache instance.
    
    This provides a singleton cache that can be shared across the application.
    
    Returns:
        Shared MemoryCache instance
        
    Examples:
        >>> from reit_risk_summarizer.services.cache import get_cache
        >>> cache = get_cache()
        >>> cache.set("key", "value")
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = MemoryCache()
    return _global_cache


def reset_cache() -> None:
    """Reset global cache instance.
    
    Useful for testing or when you want to ensure a fresh cache.
    """
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
    _global_cache = None
    logger.debug("Global cache reset")
