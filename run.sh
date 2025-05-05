#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
else
  echo "Warning: Virtual environment 'venv' not found. Running in global environment."
fi

# Check if core dependencies are installed (basic check)
echo "Checking core dependencies..."
python -c "import torch; import flask; import flask_socketio; import eventlet; import numpy"
if [ $? -ne 0 ]; then
  echo "Core dependencies missing. Please install them first:"
  echo "pip install torch flask flask-socketio eventlet numpy"
  # Consider adding installation commands here if desired
  # pip install torch flask flask-socketio eventlet numpy
  # exit 1 # Optional: exit if install fails
fi

# Check optional AV dependencies (don't exit if missing)
echo "Checking optional AV dependencies (warnings only)..."
python -c "import cupy" > /dev/null 2>&1 || echo "Warning: cupy not found."
python -c "import numba.cuda" > /dev/null 2>&1 || echo "Warning: numba.cuda not found."
python -c "import vispy" > /dev/null 2>&1 || echo "Warning: vispy not found."
python -c "import sounddevice" > /dev/null 2>&1 || echo "Warning: sounddevice not found."
python -c "import cuml" > /dev/null 2>&1 || echo "Warning: cuml (RAPIDS) not found."


# Run the server
echo "Starting server.py..."
python server.py

# Deactivate environment if activated
if [ -d "venv" ]; then
  echo "Deactivating virtual environment."
  deactivate
fi

echo "Script finished."

