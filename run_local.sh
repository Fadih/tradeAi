#!/bin/bash

# Trading AI - Local Development Startup Script
# This script sets up and runs the Trading AI application locally for development

set -e  # Exit on any error

echo "🚀 Starting Trading AI Local Development Environment"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if Redis is running
echo "🔍 Checking Redis status..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo "📦 Starting Redis server..."
    if command -v brew > /dev/null 2>&1; then
        brew services start redis
    else
        echo "❌ Redis not found. Please install Redis manually."
        exit 1
    fi
else
    echo "✅ Redis is running"
fi

# Check environment variables
echo "🔧 Checking environment configuration..."
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Using defaults."
else
    echo "✅ Environment file found"
fi

# Test CLI
echo "🧪 Testing CLI configuration..."
python -m agent.cli show-config

echo ""
echo "🎉 Setup complete! You can now:"
echo ""
echo "1. Run the web interface:"
echo "   python start_web.py"
echo "   Then open: http://localhost:8000"
echo ""
echo "2. Run CLI commands:"
echo "   python -m agent.cli run-once    # Generate trading tips"
echo "   python -m agent.cli backtest    # Run backtest"
echo "   python -m agent.cli tune        # Auto-tune parameters"
echo ""
echo "3. Use IntelliJ IDEA:"
echo "   - Open this project in IntelliJ"
echo "   - Configure Python interpreter to use .venv/bin/python"
echo "   - Create run configurations as described in INTELLIJ_SETUP.md"
echo ""
echo "📚 For detailed setup instructions, see: INTELLIJ_SETUP.md"
echo ""
echo "Happy trading! 🚀"
