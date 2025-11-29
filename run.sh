#!/bin/bash

echo "🚀 Launching Fire Detection System GUI..."

# Navigate to the project directory
cd "$(dirname "$0")"

# Activate the virtual environment
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ venv not found! Run ./setup.sh first."
    exit 1
fi

# Run the GUI application
echo "📂 Running gui.py..."
python gui.py
