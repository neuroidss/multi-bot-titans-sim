# Multi-Bot Titans Simulator (v5.15 - AV Process Fix)

```text
🌌 A complex simulation of AI-powered bots competing in a dynamic grid environment with real-time AV processing
```

![Simulation Screenshot](https://github.com/neuroidss/multi-bot-titans-sim/blob/main/Screenshot%20from%202025-05-05%2002-56-48.png?raw=true) <!-- Replace with actual screenshot path -->

## 🚀 Features

```text
- 🤖 Multi-type bot ecosystem (Learning/Hardcoded/Player-controlled)
- 🎮 Real-time web interface with interactive controls
- 🧠 Neural Memory MLP for learning bot decision-making
- 🔊 Real-time audio/visual processing (VisPy/Matplotlib/Web modes)
- 🔧 Configurable parameters via UI and server config
- 🌐 WebSocket-based communication
- 📊 Detailed performance statistics and anomaly tracking
- 🕹️ Mobile-friendly control interface
```

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/neuroidss/multi-bot-titans-sim.git
cd multi-bot-titans-sim

# Install Python dependencies
pip install flask flask-socketio eventlet numpy torch vispy matplotlib sounddevice python-socketio

# For CUDA support (recommended):
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## 🖥️ Usage

```bash
# Start the server
python server.py

# Access the web interface at:
http://localhost:5001
```

**Basic Controls:**
```text
- Click/Drag on grid to set bot targets
- Use mobile buttons or keyboard for direct control
- Adjust parameters in real-time via control panel
- Switch between visualization modes (VisPy/Matplotlib/Web)
```

## ⚡ Configuration

Key configuration options (`server.py`):
```python
DEFAULT_CONFIG = {
    "GRID_SIZE": 25,
    "NUM_LEARNING_BOTS": 4,
    "SIMULATION_SPEED_MS": 30,
    "VISUALIZATION_MODE": "web",  # vispy/matplotlib/web/none
    "ENABLE_AV_OUTPUT": True,
    "LEARNING_BOT_DIM": 128,
    "AUTOSTART_SIMULATION": True
}
```

## 🛠️ Troubleshooting

**Common Issues:**
```text
1. Missing Dependencies: Ensure all Python packages are installed
2. CUDA Errors: Verify PyTorch CUDA installation with torch.cuda.is_available()
3. AV Library Issues: Check console for missing visualization/audio libs
4. Port Conflicts: Change default port in server.py if needed
```

**Debugging Command:**
```bash
python -m server --debug  # Add debug flags as needed
```

## 🤝 Contributing

```text
1. Fork the repository
2. Create feature branch (git checkout -b feature/amazing-feature)
3. Commit changes (git commit -m 'Add amazing feature')
4. Push to branch (git push origin feature/amazing-feature)
5. Open Pull Request
```

## 📜 License

```text
Distributed under MIT License. See LICENSE for more information.
```

## 📬 Contact

```text
Project Maintainer: Your Name - your.email@example.com
GitHub Profile: https://github.com/neuroidss
```
