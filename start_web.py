#!/usr/bin/env python3
"""
Trading Agent Web Interface Startup Script
Simple script to start the web interface with proper configuration
"""

import os
import sys
import uvicorn
from pathlib import Path

def main():
    """Start the web interface"""
    
    # Add the current directory to Python path
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Set default environment variables if not set
    os.environ.setdefault('AGENT_LOG_LEVEL', 'info')
    os.environ.setdefault('AGENT_LOG_FORMAT', 'simple')
    
    # Load configuration from app.yaml
    try:
        from agent.config import get_config
        config = get_config()
        
        # Use server configuration from app.yaml
        host = config.server.host
        port = config.server.port
        reload = config.server.reload
        debug = config.server.debug
        workers = config.server.workers
        
        print(f"🔧 Using server configuration from app.yaml")
        
    except Exception as e:
        print(f"⚠️ Failed to load config, using environment variables: {e}")
        
        # Fallback to environment variables
        host = os.getenv('WEB_HOST', '0.0.0.0')
        port = int(os.getenv('WEB_PORT', '8000'))
        reload = os.getenv('WEB_RELOAD', 'false').lower() == 'true'
        debug = os.getenv('WEB_DEBUG', 'false').lower() == 'true'
        workers = int(os.getenv('WEB_WORKERS', '1'))
    
    log_level = os.getenv('WEB_LOG_LEVEL', 'info')
    
    print(f"🚀 Starting Trading Agent Web Interface")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print(f"🐛 Debug: {debug}")
    print(f"👥 Workers: {workers}")
    print(f"📝 Log Level: {log_level}")
    print(f"🌐 Dashboard: http://{host}:{port}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/api/health")
    print("=" * 50)
    
    try:
        # Start the web server with configuration
        uvicorn.run(
            "web.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,  # Workers ignored in reload mode
            log_level=log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped by user")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
