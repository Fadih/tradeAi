# IntelliJ IDEA Setup Guide for Trading AI

This guide will help you set up the Trading AI application to run and debug in IntelliJ IDEA without Docker.

## Prerequisites

✅ **Completed Setup:**
- Python virtual environment created (`.venv`)
- Dependencies installed (`pip install -r requirements.txt`)
- Environment variables configured (`.env`)
- Redis server running locally
- Application tested and working

## IntelliJ IDEA Configuration

### 1. Open Project in IntelliJ

1. Open IntelliJ IDEA
2. Choose "Open" and select the `/Users/fhussein/Documents/repositories/huggingface/tradeAi` directory
3. IntelliJ should automatically detect the Python project

### 2. Configure Python Interpreter

1. Go to **File → Project Structure** (or `Cmd+;` on Mac)
2. Select **Project → Project SDK**
3. Click the gear icon → **Add...**
4. Choose **Existing Environment**
5. Navigate to and select: `/Users/fhussein/Documents/repositories/huggingface/tradeAi/.venv/bin/python`
6. Click **OK**

### 3. Create Run Configurations

#### Configuration 1: Web Interface (Main Application)

1. Go to **Run → Edit Configurations...**
2. Click the **+** button → **Python**
3. Configure as follows:

**Name:** `Trading AI Web Interface`

**Script path:** `/Users/fhussein/Documents/repositories/huggingface/tradeAi/start_web.py`

**Python interpreter:** `/Users/fhussein/Documents/repositories/huggingface/tradeAi/.venv/bin/python`

**Working directory:** `/Users/fhussein/Documents/repositories/huggingface/tradeAi`

**Environment variables:**
```
PYTHONPATH=/Users/fhussein/Documents/repositories/huggingface/tradeAi
```

**Environment variables file:** `/Users/fhussein/Documents/repositories/huggingface/tradeAi/.env`

4. Click **OK**

#### Configuration 2: CLI Commands

1. Go to **Run → Edit Configurations...**
2. Click the **+** button → **Python**
3. Configure as follows:

**Name:** `Trading AI CLI - Show Config`

**Module name:** `agent.cli`

**Parameters:** `show-config`

**Python interpreter:** `/Users/fhussein/Documents/repositories/huggingface/tradeAi/.venv/bin/python`

**Working directory:** `/Users/fhussein/Documents/repositories/huggingface/tradeAi`

**Environment variables:**
```
PYTHONPATH=/Users/fhussein/Documents/repositories/huggingface/tradeAi
```

**Environment variables file:** `/Users/fhussein/Documents/repositories/huggingface/tradeAi/.env`

4. Click **OK**

#### Configuration 3: CLI - Generate Trading Tips

1. Go to **Run → Edit Configurations...**
2. Click the **+** button → **Python**
3. Configure as follows:

**Name:** `Trading AI CLI - Generate Tips`

**Module name:** `agent.cli`

**Parameters:** `run-once`

**Python interpreter:** `/Users/fhussein/Documents/repositories/huggingface/tradeAi/.venv/bin/python`

**Working directory:** `/Users/fhussein/Documents/repositories/huggingface/tradeAi`

**Environment variables:**
```
PYTHONPATH=/Users/fhussein/Documents/repositories/huggingface/tradeAi
```

**Environment variables file:** `/Users/fhussein/Documents/repositories/huggingface/tradeAi/.env`

4. Click **OK**

#### Configuration 4: CLI - Backtest

1. Go to **Run → Edit Configurations...**
2. Click the **+** button → **Python**
3. Configure as follows:

**Name:** `Trading AI CLI - Backtest`

**Module name:** `agent.cli`

**Parameters:** `backtest --bars 100`

**Python interpreter:** `/Users/fhussein/Documents/repositories/huggingface/tradeAi/.venv/bin/python`

**Working directory:** `/Users/fhussein/Documents/repositories/huggingface/tradeAi`

**Environment variables:**
```
PYTHONPATH=/Users/fhussein/Documents/repositories/huggingface/tradeAi
```

**Environment variables file:** `/Users/fhussein/Documents/repositories/huggingface/tradeAi/.env`

4. Click **OK**

### 4. Debug Configuration

To debug the application:

1. Set breakpoints in your code by clicking in the left margin
2. Select the run configuration you want to debug
3. Click the **Debug** button (🐛) instead of **Run**
4. The debugger will stop at your breakpoints

### 5. Environment Variables

The application uses environment variables from the `.env` file. Key variables include:

```bash
# Core Configuration
AGENT_TICKERS=BTC/USDT,ETH/USDT,SPY
AGENT_TIMEFRAME=1h
AGENT_NOTIFIER=console
AGENT_LOG_LEVEL=debug

# Web Interface
WEB_HOST=0.0.0.0
WEB_PORT=8000
WEB_RELOAD=true
WEB_DEBUG=true

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Hugging Face (for sentiment analysis)
HF_TOKEN=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
```

### 6. Running the Application

#### Web Interface (Recommended for Development)
1. Select **"Trading AI Web Interface"** configuration
2. Click **Run** or **Debug**
3. Open browser to: http://localhost:8000
4. API documentation: http://localhost:8000/docs

#### CLI Commands
1. Select any CLI configuration
2. Click **Run** or **Debug**
3. View output in IntelliJ console

### 7. Debugging Tips

#### Setting Breakpoints
- Click in the left margin next to line numbers
- Red dots indicate active breakpoints
- Right-click breakpoints for conditional breakpoints

#### Debug Console
- Use the debug console to evaluate expressions
- Inspect variables in the Variables panel
- Step through code with F8 (Step Over), F7 (Step Into), F9 (Resume)

#### Common Debug Locations
- `agent/cli.py` - CLI entry point
- `web/main.py` - Web interface entry point
- `agent/engine.py` - Trading signal generation
- `agent/models/sentiment.py` - Sentiment analysis
- `agent/indicators.py` - Technical indicators

### 8. Troubleshooting

#### Import Errors
- Ensure `PYTHONPATH` is set correctly
- Check that the virtual environment is selected as interpreter

#### Redis Connection Issues
- Ensure Redis is running: `brew services start redis`
- Check Redis status: `redis-cli ping`

#### Port Already in Use
- Change `WEB_PORT` in `.env` file
- Or kill existing process: `lsof -ti:8000 | xargs kill`

#### Environment Variables Not Loading
- Ensure `.env` file is in the project root
- Check file permissions
- Verify variable names match exactly

### 9. Project Structure

```
tradeAi/
├── agent/                 # Core trading agent modules
│   ├── cli.py            # CLI entry point
│   ├── engine.py         # Signal generation
│   ├── indicators.py     # Technical indicators
│   ├── models/           # ML models
│   └── ...
├── web/                  # Web interface
│   ├── main.py          # FastAPI application
│   └── ...
├── config/              # Configuration files
├── .env                 # Environment variables
├── start_web.py         # Web server startup script
└── requirements.txt     # Python dependencies
```

### 10. Next Steps

1. **Start with the Web Interface** - Most comprehensive for development
2. **Set breakpoints** in key functions to understand the flow
3. **Use the API documentation** at http://localhost:8000/docs
4. **Monitor logs** in the IntelliJ console
5. **Test different configurations** using the CLI run configurations

## Quick Start Commands

```bash
# Start Redis (if not running)
brew services start redis

# Activate virtual environment
source .venv/bin/activate

# Test CLI
python -m agent.cli show-config

# Start web interface
python start_web.py
```

## Support

If you encounter issues:
1. Check the logs in IntelliJ console
2. Verify all environment variables are set
3. Ensure Redis is running
4. Check that the virtual environment is activated
5. Verify Python interpreter is set correctly

Happy debugging! 🚀
