# Trading Agent Multi-Stage Dockerfile

# =============================================================================
# Stage 1: Build Stage - Install dependencies and build the application
# =============================================================================
FROM python:3.9-slim as builder

# Set environment variables for build
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Development Stage - For development and testing
# =============================================================================
FROM python:3.9-slim as development

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV AGENT_LOG_LEVEL=debug
ENV AGENT_LOG_FORMAT=detailed

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/config

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash trading && \
    chown -R trading:trading /app
USER trading

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import agent; print('OK')" || exit 1

# Expose web interface port
EXPOSE 8000

# Default command for development
CMD ["python", "-m", "web.main"]

# =============================================================================
# Stage 3: Production Stage - Optimized for production deployment
# =============================================================================
FROM python:3.9-slim as production

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV AGENT_LOG_LEVEL=info
ENV AGENT_LOG_FORMAT=simple
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only necessary application files
COPY agent/ ./agent/
COPY web/ ./web/
COPY config/ ./config/
COPY requirements.txt .
COPY *.md .

# Create necessary directories
RUN mkdir -p /app/logs /app/config

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash trading && \
    chown -R trading:trading /app
USER trading

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import agent; print('OK')" || exit 1

# Expose web interface port
EXPOSE 8000

# Default command for production
CMD ["python", "-m", "web.main"]
