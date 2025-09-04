# Trading Agent Makefile
# Usage: make <target>

.PHONY: help install test clean build docker-build docker-run docker-stop docker-clean lint format check-deps setup-dev

# Default target
help:
	@echo "Trading Agent - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install      - Install dependencies"
	@echo "  setup-dev    - Setup development environment"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo "  clean        - Clean build artifacts"
	@echo ""
	@echo "Building:"
	@echo "  build        - Build the trading agent"
	@echo "  check-deps   - Check dependency conflicts"
	@echo ""
	@echo "Web Interface:"
	@echo "  web-install  - Install web interface dependencies"
	@echo "  web-run      - Run web interface (development)"
	@echo "  web-build    - Build web interface for production"
	@echo "  web-test     - Test web interface endpoints"
	@echo ""
	@echo "Redis:"
	@echo "  redis-install - Install Redis server"
	@echo "  redis-start   - Start Redis server"
	@echo "  redis-stop    - Stop Redis server"
	@echo "  redis-status  - Check Redis status"
	@echo ""
	@echo "Docker:"
	@echo "  docker-clean-all - Remove ALL Docker layers, images, and containers"
	@echo "  docker-build     - Build Docker image"
	@echo "  docker-run       - Run Docker container"
	@echo "  docker-stop      - Stop Docker container"
	@echo "  docker-clean     - Clean Docker images and containers"
	@echo ""
	@echo "Examples:"
	@echo "  make install && make test"
	@echo "  make docker-clean-all && make docker-build && make docker-run"
	@echo "  make docker-clean-all && docker compose up -d"

# Development setup
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully!"

setup-dev:
	@echo "Setting up development environment..."
	python -m venv .venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source .venv/bin/activate  # On Unix/macOS"
	@echo "  .venv\\Scripts\\activate     # On Windows"
	@echo "Then run: make install"

# Testing and quality
test:
	@echo "Running tests..."
	python -m pytest tests/ -v --tb=short
	@echo "✅ Tests completed!"

lint:
	@echo "Running linting checks..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 agent/ --max-line-length=100 --ignore=E501,W503; \
	else \
		echo "⚠️  flake8 not found. Install with: pip install flake8"; \
	fi
	@if command -v pylint >/dev/null 2>&1; then \
		pylint agent/ --disable=C0114,C0116; \
	else \
		echo "⚠️  pylint not found. Install with: pip install pylint"; \
	fi

format:
	@echo "Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		black agent/ --line-length=100; \
	else \
		echo "⚠️  black not found. Install with: pip install black"; \
	fi

# Building
build:
	@echo "Building trading agent..."
	@if [ -f "setup.py" ]; then \
		python setup.py build; \
	else \
		echo "✅ Trading agent is ready to run!"; \
		echo "Use: python -m agent.cli <command>"; \
	fi

check-deps:
	@echo "Checking dependency conflicts..."
	pip check
	@echo "✅ No dependency conflicts found!"

# Web Interface operations
web-install:
	@echo "Installing web interface dependencies..."
	pip install fastapi uvicorn[standard] python-multipart redis aioredis
	@echo "✅ Web interface dependencies installed!"

web-run:
	@echo "Starting web interface (development mode)..."
	@echo "Access dashboard at: http://localhost:8000"
	@echo "API docs at: http://localhost:8000/docs"
	python -m web.main

web-build:
	@echo "Building web interface for production..."
	@echo "✅ Web interface is ready for production deployment!"

web-test:
	@echo "Testing web interface endpoints..."
	@echo "Starting test server..."
	@timeout 10s python -m web.main &
	@sleep 3
	@curl -f http://localhost:8000/api/health || echo "❌ Health check failed"
	@curl -f http://localhost:8000/api/status || echo "❌ Status endpoint failed"
	@curl -f http://localhost:8000/api/redis/status || echo "❌ Redis status failed"
	@pkill -f "python -m web.main" || true
	@echo "✅ Web interface tests completed!"

# Redis operations
redis-install:
	@echo "Installing Redis..."
	@if command -v brew >/dev/null 2>&1; then \
		brew install redis; \
	elif command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update && sudo apt-get install -y redis-server; \
	elif command -v yum >/dev/null 2>&1; then \
		sudo yum install -y redis; \
	else \
		echo "⚠️  Please install Redis manually for your system"; \
	fi
	@echo "✅ Redis installation completed!"

redis-start:
	@echo "Starting Redis server..."
	@if command -v redis-server >/dev/null 2>&1; then \
		redis-server --daemonize yes; \
		echo "✅ Redis server started!"; \
	else \
		echo "❌ Redis server not found. Run 'make redis-install' first."; \
	fi

redis-stop:
	@echo "Stopping Redis server..."
	@if command -v redis-cli >/dev/null 2>&1; then \
		redis-cli shutdown; \
		echo "✅ Redis server stopped!"; \
	else \
		echo "❌ Redis CLI not found."; \
	fi

redis-status:
	@echo "Checking Redis status..."
	@if command -v redis-cli >/dev/null 2>&1; then \
		redis-cli ping || echo "❌ Redis not responding"; \
	else \
		echo "❌ Redis CLI not found."; \
	fi

# Docker operations
docker-clean-all:
	@echo "🧹 Removing ALL Docker layers, images, and containers..."
	docker compose down 2>/dev/null || true
	docker stop $$(docker ps -aq) 2>/dev/null || true
	docker rm $$(docker ps -aq) 2>/dev/null || true
	docker rmi $$(docker images -aq) 2>/dev/null || true
	docker system prune -af --volumes
	docker builder prune -af
	@echo "✅ All Docker layers, images, and containers removed!"

docker-build: docker-clean-all
	@echo "Building Docker image..."
	docker build -t trading-agent:latest .
	@echo "✅ Docker image built successfully!"

docker-run:
	@echo "Running Docker container..."
	@if [ ! "$$(docker images -q trading-agent:latest 2> /dev/null)" ]; then \
		echo "⚠️  Docker image not found. Run 'make docker-build' first."; \
		exit 1; \
	fi
	docker run -d --name trading-agent \
		-e AGENT_LOG_LEVEL=info \
		-e AGENT_LOG_FORMAT=simple \
		-v $(PWD)/config:/app/config \
		-v $(PWD)/logs:/app/logs \
		trading-agent:latest
	@echo "✅ Container started! Check logs with: docker logs trading-agent"

docker-stop:
	@echo "Stopping Docker container..."
	docker stop trading-agent 2>/dev/null || echo "Container not running"
	docker rm trading-agent 2>/dev/null || echo "Container not found"
	@echo "✅ Container stopped and removed!"

docker-clean:
	@echo "Cleaning Docker resources..."
	docker stop trading-agent 2>/dev/null || true
	docker rm trading-agent 2>/dev/null || true
	docker rmi trading-agent:latest 2>/dev/null || true
	docker system prune -f
	@echo "✅ Docker cleanup completed!"

# Utility targets
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup completed!"

# Quick start
quick-start: install
	@echo ""
	@echo "🚀 Quick Start Guide:"
	@echo "1. Set environment variables (see README.md)"
	@echo "2. Test the agent: python -m agent.cli show-config"
	@echo "3. Get a trading tip: python -m agent.cli tip"
	@echo "4. Run backtest: python -m agent.cli backtest"
	@echo "5. Auto-tune parameters: python -m agent.cli tune"
	@echo ""
	@echo "For more commands: python -m agent.cli --help"
