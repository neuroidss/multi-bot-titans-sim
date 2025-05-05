#!/bin/bash

echo "Creating Python virtual environment (venv)..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Remind user about PyTorch installation if needed
echo ""
echo "---------------------------------------------------------------------"
echo "IMPORTANT: Ensure you have the correct PyTorch version installed"
echo "for your system (CPU or CUDA-enabled GPU)."
echo "If the installed version is incorrect, uninstall it and visit:"
echo "https://pytorch.org/get-started/locally/"
echo "to get the correct installation command."
echo "Example (check the PyTorch site for current commands):"
echo "# pip uninstall torch torchvision torchaudio"
echo "# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
echo "---------------------------------------------------------------------"
echo ""


echo "Setup complete. Activate the environment using 'source venv/bin/activate'"
