#!/bin/zsh
# -----------------------------
# Run Fire Detection GUI
# -----------------------------

# 1️⃣ Activate the virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# 2️⃣ Run the GUI
python gui.py

# 3️⃣ Deactivate the virtual environment after closing
deactivate
