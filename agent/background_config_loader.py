"""
Background configuration loader that periodically loads configuration from YAML files to Redis
"""
import asyncio
import logging
from typing import Optional
from agent.config import load_config_from_files
from agent.cache.redis_client import get_redis_client
from dataclasses import asdict

logger = logging.getLogger(__name__)

class BackgroundConfigLoader:
    """Background service that loads configuration to Redis periodically"""
    
    def __init__(self, interval_minutes: int = 1):
        self.interval_minutes = interval_minutes
        self.interval_seconds = interval_minutes * 60
        self.running = False
        self.task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the background configuration loader"""
        if self.running:
            logger.warning("Background config loader is already running")
            return
        
        self.running = True
        self.task = asyncio.create_task(self._load_config_loop())
        logger.info(f"Background config loader started (interval: {self.interval_minutes} minutes)")
    
    async def stop(self):
        """Stop the background configuration loader"""
        if not self.running:
            return
        
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Background config loader stopped")
    
    async def _load_config_loop(self):
        """Main loop that loads configuration periodically"""
        while self.running:
            try:
                await self._load_config_to_redis()
                await asyncio.sleep(self.interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background config loader: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(30)
    
    async def _load_config_to_redis(self):
        """Load configuration from YAML files to Redis"""
        try:
            logger.debug("Loading configuration from YAML files to Redis...")
            
            # Load configuration from YAML files
            config = load_config_from_files()
            
            # Get Redis client
            redis_client = await get_redis_client()
            if not redis_client:
                logger.error("Redis client not available")
                return
            
            # Ensure Redis connection is active
            if not await redis_client.is_connected():
                logger.warning("Redis connection lost, attempting to reconnect...")
                if not await redis_client.connect():
                    logger.error("Failed to reconnect to Redis")
                    return
            
            # Cache different sections of configuration
            config_sections = {
                'universe': asdict(config.universe),
                'signals': asdict(config.signals),
                'guardrails': asdict(config.guardrails),
                'safety': asdict(config.safety),
                'technical_analysis': asdict(config.technical_analysis),
                'sentiment_analysis': asdict(config.sentiment_analysis),
                'exchanges': asdict(config.exchanges),
                'risk_management': asdict(config.risk_management),
                'app': asdict(config.app),
                'database': asdict(config.database),
                'security': asdict(config.security),
                'logging': asdict(config.logging),
                'models': asdict(config.models),
            }
            
            # Cache each section with 1 hour TTL
            for section_name, section_data in config_sections.items():
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        success = await redis_client.cache_config(section_name, section_data, ttl=3600)
                        if success:
                            logger.debug(f"Cached config section: {section_name}")
                            break
                        else:
                            logger.warning(f"Failed to cache config section: {section_name} (attempt {attempt + 1}/{max_retries})")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(1)  # Wait before retry
                    except Exception as e:
                        logger.error(f"Exception caching config section {section_name} (attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1)  # Wait before retry
                        else:
                            logger.error(f"Section data type: {type(section_data)}, data: {section_data}")
            
            # Cache the full configuration as well
            full_config = asdict(config)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    success = await redis_client.cache_config('full', full_config, ttl=3600)
                    if success:
                        logger.debug("Cached full configuration")
                        break
                    else:
                        logger.warning(f"Failed to cache full configuration (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1)  # Wait before retry
                except Exception as e:
                    logger.error(f"Exception caching full configuration (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)  # Wait before retry
            
            logger.info("Configuration loaded to Redis successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration to Redis: {e}")
    
    async def load_config_once(self):
        """Load configuration once (useful for startup)"""
        await self._load_config_to_redis()

# Global instance
_config_loader: Optional[BackgroundConfigLoader] = None

async def start_config_loader(interval_minutes: int = 1):
    """Start the global configuration loader"""
    global _config_loader
    if _config_loader is None:
        _config_loader = BackgroundConfigLoader(interval_minutes)
    await _config_loader.start()

async def stop_config_loader():
    """Stop the global configuration loader"""
    global _config_loader
    if _config_loader:
        await _config_loader.stop()
        _config_loader = None

async def load_config_once():
    """Load configuration once (useful for startup)"""
    global _config_loader
    if _config_loader is None:
        _config_loader = BackgroundConfigLoader()
    await _config_loader.load_config_once()
