#!/bin/bash

# GP MLOps MLOps Dashboard Launcher
# Starts backend (8888) and frontend (2222)

set -e

echo "LAUNCH GP MLOps MLOps Dashboard"
echo "================================="
echo "Backend: http://localhost:2233"
echo "Frontend: http://localhost:2222"
echo "================================="
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "FAIL Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Install dependencies
echo "PKG Installing dependencies..."
pip3 install -r requirements.txt --quiet

echo "PASS Dependencies installed"
echo

# Create log directory
mkdir -p logs

# Start backend in background
echo "FIX Starting backend server on port 2233..."
python3 backend.py > logs/backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Check if backend is running
if ps -p $BACKEND_PID > /dev/null; then
    echo "PASS Backend server started successfully"
else
    echo "FAIL Backend server failed to start"
    exit 1
fi

# Start frontend
echo "🎨 Starting frontend dashboard on port 2222..."
echo "🌐 Dashboard will open automatically in your browser"
echo

# Keep track of background process
echo $BACKEND_PID > backend.pid

# Start frontend (this will block)
python3 frontend.py

# Cleanup on exit
cleanup() {
    echo
    echo "🛑 Shutting down servers..."
    if [ -f backend.pid ]; then
        kill $(cat backend.pid) 2>/dev/null || true
        rm -f backend.pid
    fi
    echo "PASS Shutdown complete"
}

trap cleanup EXIT