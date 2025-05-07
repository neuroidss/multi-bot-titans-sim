# Multi-Bot Titans Simulator (v5.17 - EEG/Hallucination/Rules)

🌌 A complex simulation of AI-powered bots competing in a dynamic grid environment, featuring Hierarchical Neural Memory, EEG-like data integration, mental attacks, rule-based guidance, and real-time audiovisualization.

[![Simulation Screencast](https://img.youtube.com/vi/Lgj_rm51TZ8/0.jpg)](https://www.youtube.com/watch?v=Lgj_rm51TZ8) 
*Note: Screencast may be from an older version.*

![Simulation Screenshot](https://github.com/neuroidss/multi-bot-titans-sim/blob/main/Screenshot%20from%202025-05-05%2002-56-48.png?raw=true)
*Note: Screenshot may be from an older version.*

## 🚀 Features

-   🤖 **Multi-Bot Ecosystem**: Learning, Hardcoded, and Player-controlled bots.
-   🧠 **Hierarchical Neural Memory (HNM)**: Learning Bots utilize a sophisticated, configurable HNM system (inspired by predictive coding and Titans paper).
-   🤯 **Mental Attacks & Hallucination**: Bots can mentally attack others, causing them to act based on internal reconstructions (hallucinations) rather than real senses.
-   룰 **Rule-Based Guidance**: Hardcoded bot logic can be fed as an external input signal to the HNM of Learning Bots.
-   📈 **EEG-like Feature Integration**: Simulate or input EEG-like data (band power, coherence) as an external signal to the HNM.
-   🎮 **Real-Time Web Interface**: Interactive controls for simulation, parameters, and player interaction.
-   🔊 **Audiovisualization**: Real-time audio and visual feedback (VisPy/Matplotlib/Web modes) based on bot internal states (anomaly, memory vector, etc.).
-   🔧 **Deeply Configurable**: Extensive parameters tunable via UI and server config, including HNM architecture, EEG processing, and bot behaviors.
-   💬 **Bot Chat**: Bots can "chat" about their current state and decisions, providing insights into their internal workings.
-   🌐 **WebSocket Communication**: Efficient real-time updates between server and client.
-   📊 **Detailed Stats**: Performance statistics and per-level anomaly/weight change tracking for HNM bots.
-   📱 **Mobile-Friendly Controls**: Joystick-like controls for player interaction on mobile devices.

## ⚙️ Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/neuroidss/multi-bot-titans-sim.git
    cd multi-bot-titans-sim
    ```
2.  **Ensure Custom Libraries are Present**:
    Make sure `hierarchical_neural_memory_lib_v5.py`, `audiovisualization.py`, and `eeg_feature_extractor_lib.py` are in the same directory as `server.py`.

3.  **Install Python Dependencies**:
    It's recommended to use a virtual environment (e.g., conda or venv).
    ```bash
    pip install flask flask-socketio eventlet numpy torch vispy matplotlib sounddevice python-socketio scikit-learn mne
    ```
    *   `mne`: For EEG feature extraction (MNE-Python).
    *   `scikit-learn`: For PCA in audiovisualization.
    *   For specific MNE-Connectivity features (if enabled in EEG config), ensure `mne-connectivity` is installed: `pip install mne-connectivity`.

4.  **For CUDA Support (Recommended for PyTorch)**:
    If you have an NVIDIA GPU, install PyTorch with CUDA support for significantly better performance. Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for the correct command based on your CUDA version. Example for CUDA 12.1:
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

## 🖥️ Usage

1.  **Start the Server**:
    ```bash
    python server.py
    ```
    The server console will output status messages, including library availability and CUDA status.

2.  **Access the Web Interface**:
    Open your browser and navigate to `http://localhost:5001`.

**Basic Controls:**
-   **Simulation**: Start/Stop simulation, trigger new rounds, or perform a full reset.
-   **Parameters**: Adjust various simulation, environment, bot behavior, HNM, and EEG parameters in real-time (some may require a round or full reset).
-   **Player Control**: Join as a player to control a Learning Bot. Click/Drag on the grid to set the bot's target, or use mobile buttons for direct actions (Up, Down, Left, Right, Punch, Claim Goal, Mental Attack).
-   **Visualization**: If AV Output is enabled, choose between `VisPy` or `Matplotlib` for a separate 3D visualization window, or `Web` to stream data to the `/visuals` endpoint (useful for custom web renderers).

## ⚡ Configuration

Key configuration options are found in `DEFAULT_CONFIG` within `server.py`. Many of these can also be adjusted via the web UI.

```python
DEFAULT_CONFIG = {
    # --- Simulation Params ---
    "GRID_SIZE": 25,
    "NUM_HC_BOTS": 1,
    "NUM_LEARNING_BOTS": 4,
    "NUM_GOALS": 5,
    "MAX_STEPS_PER_ROUND": 1000,
    "SIMULATION_SPEED_MS": 30, # Lower is faster
    "NUM_ACTIONS": 7, # 0:Up, 1:Left, 2:Right, 3:Down, 4:Punch, 5:ClaimGoal, 6:MentalAttack

    # --- Bot Abilities ---
    "FREEZE_DURATION": 15, # Steps a bot is frozen after being punched
    "MENTAL_ATTACK_RANGE": 3,
    "MENTAL_ATTACK_DURATION": 30, # Steps the mental attack effect lasts
    "MENTAL_ATTACK_USES_RECONSTRUCTION": True, # If true, attacked bot uses L0 HNM reconstruction for senses

    # --- Learning Bot Behavior ---
    "LEARNING_BOT_BASE_EXPLORATION_RATE": 15.0, # Base % chance to explore
    "LEARNING_BOT_RULE_EXPLORE_PERCENT": 60.0, # % of exploration using rules
    "PLAYER_CONTROL_PERCENT": 100.0, # Influence of player input vs AI
    "USE_RULES_AS_HNS_INPUT": True, # Feed hardcoded rules to Learning Bots' HNM

    # --- Hierarchical Neural Memory Configuration ---
    "SENSORY_INPUT_DIM": 22, # Dimension of raw game state features for L0
    "POLICY_HEAD_INPUT_LEVEL_NAME": "L2_Executive", # HNM level output feeding bot's policy
    "AV_DATA_SOURCE_LEVEL_NAME": "L2_Executive", # HNM level feeding AV system

    "HIERARCHY_LEVEL_CONFIGS": [
        {
            "name": "L0_SensoryMotor", "dim": 64, "raw_sensory_input_dim": 22, # Must match SENSORY_INPUT_DIM
            "bu_source_level_names": [], # Root sensory level
            "td_source_level_names": ["L1_Context"],
            "external_input_config": [ # List of external signals
                 {"source_signal_name": "motor_efference_copy", "dim": 7}, # Matches NUM_ACTIONS
                 # {"source_signal_name": "hardcoded_rule_guidance", "dim": 7}, # Added if USE_RULES_AS_HNS_INPUT=True
            ],
            "nmm_params": { "mem_model_depth": 2, "learning_rate": 0.001, "external_signal_role": "add_to_bu", ... }
        },
        {
            "name": "L1_Context", "dim": 96,
            "bu_source_level_names": ["L0_SensoryMotor"],
            "td_source_level_names": ["L2_Executive"],
            "external_input_config": {
                 # "source_signal_name": "eeg_features", "dim": <calculated_dim> # Added if EEG_SETTINGS.enabled=True
            },
            "nmm_params": { "mem_model_depth": 2, "learning_rate": 0.001, "external_signal_role": "add_to_td", ... }
        },
        {
            "name": "L2_Executive", "dim": 128,
            "bu_source_level_names": ["L1_Context"],
            "td_source_level_names": [],
            "nmm_params": { "mem_model_depth": 3, "learning_rate": 0.0005, ... }
        }
    ],

    # --- EEG Configuration ---
    "EEG_SETTINGS": {
        "enabled": False,
        "simulate_eeg": True, # If true, generate random EEG data
        "sfreq": 250, # Sample frequency (Hz)
        "channel_names": ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "P3", "P4"], # Example
        "processing_window_samples": 250, # e.g., 1 second window @ 250Hz
        "processing_interval_new_samples": 60, # Recalculate features every ~240ms
        "processing_config": {
            "filtering": {"highpass_hz": 1.0, "highpass_order": 5},
            "feature_bands": {"Delta": [1.0, 4.0], "Theta": [4.0, 8.0], "Alpha": [8.0, 13.0], ...},
            "coherence": { "enable": False, "methods_to_calculate": ["coh"], ... }
        },
        "hns_signal_name": "eeg_features", # Name for HNM external input
        "hns_target_level": "L1_Context",  # Target HNM level for EEG features
        "flattening_order": ["band_power", "coherence"], # Order of features in vector
        "feature_dim": 0 # Dynamically calculated by server
    },

    # --- Audio/Visual Output Config ---
    "ENABLE_AV": True,
    "ENABLE_AV_OUTPUT": True,
    "VISUALIZATION_MODE": "web", # vispy/matplotlib/web/none
    "AUTOSTART_SIMULATION": True,
}
```

**Key Configuration Notes:**
-   `SENSORY_INPUT_DIM`: Must match `raw_sensory_input_dim` for the root sensory level(s) in `HIERARCHY_LEVEL_CONFIGS`.
-   `NUM_ACTIONS`: The `dim` for `motor_efference_copy` external signal should match this.
-   `external_input_config` in `HIERARCHY_LEVEL_CONFIGS`: Can be a list of dictionaries, each defining an external signal source name and its dimension. The HNM will attempt to use these based on its internal setup.
-   `EEG_SETTINGS.feature_dim`: This is calculated by the server based on the EEG configuration and should not be set manually.

## 🛠️ Troubleshooting

-   **Missing Dependencies**: Ensure all Python packages listed under Installation are correctly installed. Check for errors during `pip install`.
-   **CUDA Errors**: If using GPU, verify your PyTorch and CUDA installation. Run a simple PyTorch CUDA check script. If issues persist, the simulation will fall back to CPU.
-   **AV Library Issues**: The server console will print warnings if `VisPy`, `Matplotlib`, or `SoundDevice` are not found or fail to import. AV features will be disabled accordingly.
-   **Port Conflicts**: If port `5001` is in use, change it in the `server.py` near the end of the file.
-   **HNM Configuration Errors**: Carefully review `HIERARCHY_LEVEL_CONFIGS`. Mismatched level names, incorrect `dim` specifications, or invalid source/target connections can cause errors during HNM initialization or processing steps. The server console will provide detailed tracebacks.
-   **EEG Feature Dimension Mismatch**: If `EEG_SETTINGS` are changed in a way that alters the `feature_dim` (e.g., number of channels, enabled coherence), a full reset is usually required for the HNM to adapt.

## 🤝 Contributing

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/YourAmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/YourAmazingFeature`).
5.  Open a Pull Request.

## 📜 License

Distributed under the AGPL-3.0 License. See `LICENSE` for more information.

## 📬 Contact

Project Maintainer: Dmitry Sukhoruchkin - dmitry.sukhoruchkin@neuroidss.com

GitHub Profile: [https://github.com/neuroidss](https://github.com/neuroidss)

