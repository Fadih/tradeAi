"""
Redis-based configuration manager that reads configuration from Redis cache
"""
import logging
from typing import Optional, Dict, Any
from agent.cache.redis_client import get_redis_client
from agent.config import AgentConfig, UniverseConfig, SignalsConfig, Guardrails, SafetyConfig, TechnicalAnalysisConfig, SentimentAnalysisConfig, ExchangeConfig, RiskManagementConfig, AppConfig, DatabaseConfig, SecurityConfig, LoggingConfig, ModelConfig
from dataclasses import asdict

logger = logging.getLogger(__name__)

class RedisConfigManager:
    """Configuration manager that reads from Redis cache"""
    
    def __init__(self):
        self._cached_config: Optional[AgentConfig] = None
    
    async def get_config(self) -> Optional[AgentConfig]:
        """Get configuration from Redis cache"""
        try:
            redis_client = await get_redis_client()
            if not redis_client:
                logger.error("Redis client not available")
                return None
            
            # Try to get full configuration first
            full_config_data = await redis_client.get_cached_config('full')
            if full_config_data:
                logger.debug("Retrieved full configuration from Redis")
                return self._deserialize_config(full_config_data)
            
            # If full config not available, build from sections
            logger.debug("Full config not available, building from sections")
            return await self._build_config_from_sections()
            
        except Exception as e:
            logger.error(f"Failed to get configuration from Redis: {e}")
            return None
    
    async def _build_config_from_sections(self) -> Optional[AgentConfig]:
        """Build configuration from individual sections in Redis"""
        try:
            redis_client = await get_redis_client()
            if not redis_client:
                return None
            
            # Get all configuration sections
            sections = {
                'universe': await redis_client.get_cached_config('universe'),
                'signals': await redis_client.get_cached_config('signals'),
                'guardrails': await redis_client.get_cached_config('guardrails'),
                'safety': await redis_client.get_cached_config('safety'),
                'technical_analysis': await redis_client.get_cached_config('technical_analysis'),
                'sentiment_analysis': await redis_client.get_cached_config('sentiment_analysis'),
                'exchanges': await redis_client.get_cached_config('exchanges'),
                'risk_management': await redis_client.get_cached_config('risk_management'),
                'app': await redis_client.get_cached_config('app'),
                'database': await redis_client.get_cached_config('database'),
                'security': await redis_client.get_cached_config('security'),
                'logging': await redis_client.get_cached_config('logging'),
                'models': await redis_client.get_cached_config('models'),
            }
            
            # Check if all required sections are available
            missing_sections = [name for name, data in sections.items() if data is None]
            if missing_sections:
                logger.warning(f"Missing configuration sections in Redis: {missing_sections}")
                return None
            
            # Build configuration object
            config_data = {
                'universe': sections['universe'],
                'signals': sections['signals'],
                'guardrails': sections['guardrails'],
                'safety': sections['safety'],
                'technical_analysis': sections['technical_analysis'],
                'sentiment_analysis': sections['sentiment_analysis'],
                'exchanges': sections['exchanges'],
                'risk_management': sections['risk_management'],
                'app': sections['app'],
                'database': sections['database'],
                'security': sections['security'],
                'logging': sections['logging'],
                'models': sections['models'],
            }
            
            return self._deserialize_config(config_data)
            
        except Exception as e:
            logger.error(f"Failed to build configuration from sections: {e}")
            return None
    
    def _deserialize_config(self, config_data: Dict[str, Any]) -> AgentConfig:
        """Deserialize configuration data to AgentConfig object"""
        try:
            # Create configuration objects from data
            universe = UniverseConfig(**config_data.get('universe', {}))
            signals = SignalsConfig(**config_data.get('signals', {}))
            guardrails = Guardrails(**config_data.get('guardrails', {}))
            safety = SafetyConfig(**config_data.get('safety', {}))
            technical_analysis = TechnicalAnalysisConfig(**config_data.get('technical_analysis', {}))
            sentiment_analysis = SentimentAnalysisConfig(**config_data.get('sentiment_analysis', {}))
            exchanges = ExchangeConfig(**config_data.get('exchanges', {}))
            risk_management = RiskManagementConfig(**config_data.get('risk_management', {}))
            app = AppConfig(**config_data.get('app', {}))
            database = DatabaseConfig(**config_data.get('database', {}))
            security = SecurityConfig(**config_data.get('security', {}))
            logging = LoggingConfig(**config_data.get('logging', {}))
            models = ModelConfig(**config_data.get('models', {}))
            
            # Create main configuration object
            config = AgentConfig(
                universe=universe,
                signals=signals,
                guardrails=guardrails,
                safety=safety,
                technical_analysis=technical_analysis,
                sentiment_analysis=sentiment_analysis,
                exchanges=exchanges,
                risk_management=risk_management,
                app=app,
                database=database,
                security=security,
                logging=logging,
                models=models
            )
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to deserialize configuration: {e}")
            # Return default configuration if deserialization fails
            return AgentConfig()
    
    async def get_universe_config(self) -> Optional[Dict[str, Any]]:
        """Get universe configuration from Redis"""
        try:
            redis_client = await get_redis_client()
            if not redis_client:
                return None
            
            return await redis_client.get_cached_config('universe')
        except Exception as e:
            logger.error(f"Failed to get universe config from Redis: {e}")
            return None
    
    async def get_signals_config(self) -> Optional[Dict[str, Any]]:
        """Get signals configuration from Redis"""
        try:
            redis_client = await get_redis_client()
            if not redis_client:
                return None
            
            return await redis_client.get_cached_config('signals')
        except Exception as e:
            logger.error(f"Failed to get signals config from Redis: {e}")
            return None

# Global instance
_redis_config_manager: Optional[RedisConfigManager] = None

def get_redis_config_manager() -> RedisConfigManager:
    """Get global Redis configuration manager instance"""
    global _redis_config_manager
    if _redis_config_manager is None:
        _redis_config_manager = RedisConfigManager()
    return _redis_config_manager
