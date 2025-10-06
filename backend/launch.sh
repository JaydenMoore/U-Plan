#!/bin/bash

# Urban Planner AI - Launch Script
# Starts both the main API server and the real-time dashboard

echo "🌍 Starting Urban Planner AI System..."
echo "======================================"

# Function to check if port is available
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $1 is already in use"
        return 1
    else
        return 0
    fi
}

# Function to kill process on port
kill_port() {
    echo "🔄 Stopping existing process on port $1..."
    lsof -ti:$1 | xargs kill -9 2>/dev/null || true
    sleep 2
}

# Change to backend directory
cd "$(dirname "$0")"

# Check Python environment
echo "🐍 Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check ports
MAIN_API_PORT=8001
DASHBOARD_PORT=8050

if ! check_port $MAIN_API_PORT; then
    kill_port $MAIN_API_PORT
fi

if ! check_port $DASHBOARD_PORT; then
    kill_port $DASHBOARD_PORT
fi

echo ""
echo "🚀 Starting services..."
echo "======================"

# Start main API server in background
echo "🌐 Starting main API server on port $MAIN_API_PORT..."
python main.py &
MAIN_PID=$!

# Wait a moment for main server to start
sleep 3

# Check if main server started successfully
if ! lsof -Pi :$MAIN_API_PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "❌ Failed to start main API server"
    kill $MAIN_PID 2>/dev/null || true
    exit 1
fi

echo "✅ Main API server started successfully (PID: $MAIN_PID)"

# Start dashboard
echo "📊 Starting real-time dashboard on port $DASHBOARD_PORT..."
python dashboard.py &
DASHBOARD_PID=$!

# Wait a moment for dashboard to start
sleep 5

# Check if dashboard started successfully
if ! lsof -Pi :$DASHBOARD_PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "❌ Failed to start dashboard"
    kill $MAIN_PID $DASHBOARD_PID 2>/dev/null || true
    exit 1
fi

echo "✅ Dashboard started successfully (PID: $DASHBOARD_PID)"

echo ""
echo "🎉 Urban Planner AI System is now running!"
echo "=========================================="
echo ""
echo "📱 Services:"
echo "   • Main API:        http://localhost:$MAIN_API_PORT"
echo "   • API Docs:        http://localhost:$MAIN_API_PORT/docs"
echo "   • Dashboard:       http://localhost:$DASHBOARD_PORT"
echo ""
echo "🛠️  Features Available:"
echo "   • Environmental risk assessment"
echo "   • Probabilistic risk modeling"
echo "   • AI model interpretability (SHAP)"
echo "   • Real-time data validation"
echo "   • Water risk assessment (WRI Aqueduct)"
echo "   • Population density analysis"
echo "   • Interactive dashboard monitoring"
echo ""
echo "📊 Dashboard Features:"
echo "   • Real-time system metrics"
echo "   • Global risk assessment map"
echo "   • Data quality monitoring"
echo "   • Model performance tracking"
echo "   • Recent assessments table"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $MAIN_PID $DASHBOARD_PID 2>/dev/null || true
    wait $MAIN_PID $DASHBOARD_PID 2>/dev/null || true
    echo "✅ All services stopped"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Wait for either process to exit
wait $MAIN_PID $DASHBOARD_PID