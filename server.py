# Filename: server.py
# coding: utf-8
# Version: 5.17 (EEG, Refined Freeze/Attack, Rules Input, Chatbot)
import os
import time
import math
import random
import numpy as np
from collections import deque, namedtuple
from functools import partial
import copy # For deep copying states if needed
import traceback # For detailed error logging
import sys # For stderr printing
import multiprocessing as mp # Import multiprocessing
from typing import List, Dict, Any, Optional, Tuple # Added Tuple

# --- Eventlet Monkey Patching (IMPORTANT: Must be done early) ---
import eventlet
eventlet.monkey_patch()
print("Eventlet monkey patching applied.")

# --- Flask & SocketIO Setup ---
from flask import Flask, render_template, request, jsonify, Response # Added Response
from flask_socketio import SocketIO, emit, Namespace # Added Namespace

app = Flask(__name__, template_folder='.') # Look for index.html in the same directory
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")
print("Flask and SocketIO initialized.")

# --- PyTorch Setup ---
import torch
from torch import nn, Tensor, is_tensor, cat, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, Parameter, ParameterList

# --- Import Custom Libraries ---
try:
    # Use Hierarchical Neural Memory Library
    from hierarchical_neural_memory_lib_v5 import HierarchicalSystemV5, NeuralMemState, mem_state_detach
    print("Hierarchical Neural Memory Library (v5.0.8) imported successfully.")
    HIERARCHICAL_NEURAL_LIB_AVAILABLE = True
except ImportError as e:
    print(f"FATAL ERROR: Failed to import hierarchical_neural_memory_lib_v5.py: {e}")
    traceback.print_exc()
    HIERARCHICAL_NEURAL_LIB_AVAILABLE = False
    # Dummy classes/functions if import fails
    HierarchicalSystemV5 = None
    NeuralMemState = namedtuple('DummyNeuralMemState', ['seq_index', 'weights', 'optim_state'])
    mem_state_detach = lambda state: state

try:
    # Import EEG processing library
    from eeg_feature_extractor_lib import SignalProcessor, MNE_AVAILABLE, MNE_CONNECTIVITY_AVAILABLE, SCIPY_AVAILABLE
    EEG_LIB_AVAILABLE = MNE_AVAILABLE # Basic requirement
    print(f"EEG Feature Extractor Library loaded. MNE: {MNE_AVAILABLE}, MNE-Conn: {MNE_CONNECTIVITY_AVAILABLE}, SciPy: {SCIPY_AVAILABLE}")
except ImportError as e:
    print(f"Warning: Failed to import eeg_feature_extractor_lib.py: {e}. EEG features disabled.")
    EEG_LIB_AVAILABLE = False
    SignalProcessor = None
    MNE_AVAILABLE = False
    MNE_CONNECTIVITY_AVAILABLE = False
    SCIPY_AVAILABLE = False

# --- SoundDevice Setup ---
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    print("SoundDevice library imported successfully.")
except ImportError as e:
    print(f"Warning: Failed to import sounddevice: {e}. Audio output disabled.")
    SOUNDDEVICE_AVAILABLE = False
    sd = None
    
try:
    from audiovisualization import setup_av_system, update_av_system, stop_av_system, AVManager
    print("Audiovisualization Library imported successfully.")
    AV_LIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Failed to import audiovisualization.py: {e}. AV features disabled.")
    AV_LIB_AVAILABLE = False
    setup_av_system = lambda *args, **kwargs: None
    update_av_system = lambda *args, **kwargs: None
    stop_av_system = lambda *args, **kwargs: None
    AVManager = None

# --- Determine Device ---
try:
    if torch.cuda.is_available():
        if sys.platform != 'win32': 
            if mp.get_start_method(allow_none=True) is None or mp.get_start_method(allow_none=True) == 'fork':
                pass 
        device = torch.device("cuda")
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        try: _ = torch.tensor([1.0], device=device); torch.cuda.empty_cache(); print("CUDA device test successful.")
        except Exception as e: print(f"Warning: CUDA device error: {e}. Falling back to CPU."); device = torch.device("cpu")
    else: device = torch.device("cpu"); print("CUDA not available. Using CPU.")
except Exception as e: print(f"Error during PyTorch device setup: {e}. Falling back to CPU."); device = torch.device("cpu")

# ================================================================
# --- DEFAULT CONFIGURATION ---
# ================================================================
DEFAULT_CONFIG = {
    # Simulation Params
    "GRID_SIZE": 25,
    "NUM_HC_BOTS": 1,
    "NUM_LEARNING_BOTS": 4,
    "NUM_GOALS": 5,
    "OBSTACLES_FACTOR_MIN": 0.03,
    "OBSTACLES_FACTOR_MAX": 0.08,
    "MAX_STEPS_PER_ROUND": 1000,
    "SIMULATION_SPEED_MS": 30,
    "FREEZE_DURATION": 15,
    "VISIBILITY_RANGE": 8,
    "NUM_ACTIONS": 7, # 0:Up, 1:Left, 2:Right, 3:Down, 4:Punch, 5:ClaimGoal, 6:MentalAttack
    "RANDOMIZE_ENV_PER_ROUND": True,
    "MENTAL_ATTACK_RANGE": 3,
    "MENTAL_ATTACK_DURATION": 30,
    "MENTAL_ATTACK_USES_RECONSTRUCTION": True,

    # --- Learning Bot Behavior ---
    "LEARNING_BOT_BASE_EXPLORATION_RATE": 15.0,
    "LEARNING_BOT_RULE_EXPLORE_PERCENT": 60.0,
    "PLAYER_CONTROL_PERCENT": 100.0,
    "USE_RULES_AS_HNS_INPUT": True, 

    # --- Environment Generation ---
    "MIN_GOAL_START_DISTANCE_FACTOR": 0.15,
    "MIN_BOT_START_DISTANCE_FACTOR": 0.25,
    "MIN_BOT_GOAL_DISTANCE_FACTOR": 0.15,

    # --- Hierarchical Neural Memory Configuration ---
    "SENSORY_INPUT_DIM": 22, 
    "POLICY_HEAD_INPUT_LEVEL_NAME": "L2_Executive",
    "AV_DATA_SOURCE_LEVEL_NAME": "L2_Executive", 

    "HIERARCHY_LEVEL_CONFIGS": [
        { 
            "name": "L0_SensoryMotor", "dim": 64, "raw_sensory_input_dim": 22,
            "bu_source_level_names": [],
            "td_source_level_names": ["L1_Context"],
            "external_input_config": [ 
                 {"source_signal_name": "motor_efference_copy", "dim": 7}, 
                 # {"source_signal_name": "hardcoded_rule_guidance", "dim": 7}, # Added by configure_hns_external_inputs
            ],
            "nmm_params": {
                "mem_model_depth": 2, "learning_rate": 0.001, "weight_decay": 0.01,
                "momentum_beta": 0.9, "max_grad_norm": 1.0, "external_signal_role": "add_to_bu", "verbose": False
            }
        },
        {   
            "name": "L1_Context", "dim": 96,
            "bu_source_level_names": ["L0_SensoryMotor"],
            "td_source_level_names": ["L2_Executive"],
            "external_input_config": { 
                 # "source_signal_name": "eeg_features", "dim": 0 # Added by configure_hns_external_inputs
            },
            "nmm_params": {
                "mem_model_depth": 2, "learning_rate": 0.001, "weight_decay": 0.01,
                "momentum_beta": 0.9, "max_grad_norm": 1.0, "verbose": False,
                # "external_signal_role": "add_to_td" # Set by configure_hns_external_inputs for EEG
            }
        },
        {   
            "name": "L2_Executive", "dim": 128,
            "bu_source_level_names": ["L1_Context"],
            "td_source_level_names": [],
            "nmm_params": {
                "mem_model_depth": 3, "learning_rate": 0.0005, "weight_decay": 0.01,
                "momentum_beta": 0.9, "max_grad_norm": 1.0, "verbose": False
            }
        }
    ],

    # --- EEG Configuration ---
    "EEG_SETTINGS": {
        "enabled": False, 
        "simulate_eeg": True, 
        "sfreq": 250, 
        "channel_names": ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "P3", "P4"], 
        "processing_window_samples": 250, 
        "processing_interval_new_samples": 60, 
        "processing_config": {
            "filtering": {"highpass_hz": 1.0, "highpass_order": 5},
            "feature_bands": {"Delta": [1.0, 4.0], "Theta": [4.0, 8.0], "Alpha": [8.0, 13.0], "Beta": [13.0, 30.0], "Gamma": [30.0, 45.0]},
            "coherence": {
                "enable": False, 
                "methods_to_calculate": ["coh"],
                "bands": ["Alpha", "Beta"],
                "use_all_channel_pairs": True,
            }
        },
        "hns_signal_name": "eeg_features", 
        "hns_target_level": "L1_Context", 
        "flattening_order": ["band_power", "coherence"], 
        "feature_dim": 0 
    },

    # --- Audio/Visual Output Config ---
    "ENABLE_AV": True,
    "ENABLE_AV_OUTPUT": True,
    "VISUALIZATION_MODE": "web", 
    "AUTOSTART_SIMULATION": True,
}
current_config = copy.deepcopy(DEFAULT_CONFIG)

# --- Global State ---
bots = {}
players = {}
environment = None
hierarchical_neural_system = None
av_manager = None
signal_processor = None 
simulation_running = False
simulation_loop_task = None
round_number = 0
stats = {'hc_total_goals': 0, 'learning_total_goals': 0}
player_direct_actions = {}

def flatten_nf_metrics(nf_metrics: Dict[str, float], processor: SignalProcessor, order: List[str]) -> List[float]:
    """ Flattens the neurofeedback metrics into a consistent vector based on config order. """
    feature_vector = []
    if not processor: return feature_vector

    for feature_type in order:
        if feature_type == "band_power":
            for band_name in processor.feature_band_names:
                for ch_name in processor.input_channel_names:
                    key = f"{band_name}_power_{ch_name}"
                    feature_vector.append(nf_metrics.get(key, 0.0))
        elif feature_type == "coherence" and processor.calculate_coherence:
            for method in processor.coherence_methods_to_calc:
                for band_name in processor.coherence_band_map:
                    for ch1, ch2 in processor.coherence_pairs_names_list:
                        key = f"{band_name}_{method}_{ch1}-{ch2}"
                        feature_vector.append(nf_metrics.get(key, 0.0))
    return feature_vector


def calculate_eeg_feature_dim(settings: Dict, processor: SignalProcessor) -> int:
    """ Calculates the expected dimension of the flattened EEG feature vector. """
    dim = 0
    if not processor: return 0
    order = settings.get("flattening_order", ["band_power", "coherence"])

    for feature_type in order:
        if feature_type == "band_power":
            n_bands = len(processor.feature_band_names)
            n_channels = processor.n_input_channels
            dim += n_bands * n_channels
        elif feature_type == "coherence" and processor.calculate_coherence:
            n_coh_bands = len(processor.coherence_band_map)
            n_coh_methods = len(processor.coherence_methods_to_calc)
            n_pairs = len(processor.coherence_pairs_names_list)
            dim += n_coh_bands * n_coh_methods * n_pairs
    return dim

def configure_hns_external_inputs():
    """ Updates HIERARCHY_LEVEL_CONFIGS based on enabled features (EEG, Rules). """
    global current_config
    config_changed = False

    # --- Rule Guidance ---
    use_rules = current_config.get("USE_RULES_AS_HNS_INPUT", False)
    rule_signal_name = "hardcoded_rule_guidance"
    rule_dim = current_config['NUM_ACTIONS'] 
    target_level_rules = "L0_SensoryMotor" 

    for lvl_cfg in current_config['HIERARCHY_LEVEL_CONFIGS']:
        if lvl_cfg["name"] == target_level_rules:
            current_ext_config = lvl_cfg.get("external_input_config", [])
            if not isinstance(current_ext_config, list):
                current_ext_config = [current_ext_config] if current_ext_config and isinstance(current_ext_config, dict) else []

            has_rule_config = any(isinstance(c, dict) and c.get("source_signal_name") == rule_signal_name for c in current_ext_config)

            if use_rules and not has_rule_config:
                current_ext_config.append({"source_signal_name": rule_signal_name, "dim": rule_dim})
                lvl_cfg["external_input_config"] = current_ext_config
                config_changed = True
                print(f"HNS Config: Added '{rule_signal_name}' (dim={rule_dim}) to level '{target_level_rules}'.")
            elif not use_rules and has_rule_config:
                lvl_cfg["external_input_config"] = [c for c in current_ext_config if not (isinstance(c, dict) and c.get("source_signal_name") == rule_signal_name)]
                config_changed = True
                print(f"HNS Config: Removed '{rule_signal_name}' from level '{target_level_rules}'.")
            lvl_cfg["external_input_config"] = [c for c in lvl_cfg.get("external_input_config", []) if isinstance(c, dict)]


    # --- EEG Features ---
    eeg_settings = current_config.get("EEG_SETTINGS", {"enabled": False})
    eeg_enabled = eeg_settings.get("enabled", False) and EEG_LIB_AVAILABLE
    eeg_feature_dim = current_config.get("EEG_SETTINGS",{}).get("feature_dim", 0) 
    eeg_signal_name = eeg_settings.get("hns_signal_name")
    eeg_target_level = eeg_settings.get("hns_target_level")

    for lvl_cfg in current_config['HIERARCHY_LEVEL_CONFIGS']:
         current_ext_config = lvl_cfg.get("external_input_config", [])
         if not isinstance(current_ext_config, list): current_ext_config = [current_ext_config] if current_ext_config and isinstance(current_ext_config, dict) else []
         initial_len = len(current_ext_config)
         current_ext_config = [c for c in current_ext_config if not (isinstance(c, dict) and c.get("source_signal_name") == eeg_signal_name)]
         if len(current_ext_config) != initial_len:
             lvl_cfg["external_input_config"] = current_ext_config
             config_changed = True
             print(f"HNS Config: Cleared previous EEG signal '{eeg_signal_name}' from level '{lvl_cfg['name']}'.")
         lvl_cfg["external_input_config"] = [c for c in lvl_cfg.get("external_input_config", []) if isinstance(c, dict)]


    if eeg_enabled and eeg_feature_dim > 0 and eeg_signal_name and eeg_target_level:
        for lvl_cfg in current_config['HIERARCHY_LEVEL_CONFIGS']:
            if lvl_cfg.get("name") == eeg_target_level:
                current_ext_config = lvl_cfg.get("external_input_config", [])
                if not isinstance(current_ext_config, list): current_ext_config = [current_ext_config] if current_ext_config and isinstance(current_ext_config, dict) else []
                
                if not any(isinstance(c,dict) and c.get("source_signal_name") == eeg_signal_name for c in current_ext_config):
                    current_ext_config.append({
                        "source_signal_name": eeg_signal_name,
                        "dim": eeg_feature_dim
                    })
                    if "nmm_params" not in lvl_cfg: lvl_cfg["nmm_params"] = {} 
                    lvl_cfg["nmm_params"]["external_signal_role"] = lvl_cfg["nmm_params"].get("external_signal_role", "add_to_td") 
                    lvl_cfg["external_input_config"] = [c for c in current_ext_config if isinstance(c, dict)]
                    config_changed = True
                    print(f"HNS Config: Added '{eeg_signal_name}' (dim={eeg_feature_dim}) to level '{eeg_target_level}'.")
                break 
    
    return config_changed


def get_level_dim_by_name(level_name: str, config_levels: List[Dict]) -> Optional[int]:
    for lvl_cfg in config_levels:
        if lvl_cfg.get("name") == level_name:
            return lvl_cfg.get("dim")
    return None

def update_hierarchical_neural_system_instance():
    global hierarchical_neural_system, current_config
    if not HIERARCHICAL_NEURAL_LIB_AVAILABLE:
        print("Error: Cannot update Hierarchical NN System, library not available.")
        hierarchical_neural_system = None
        return False

    print("Updating Hierarchical Neural System...")
    configure_hns_external_inputs() 

    try:
        needs_new_system = True
        if hierarchical_neural_system:
            cached_levels = getattr(hierarchical_neural_system, '_cached_level_configs_ref', None)
            cached_sensory_dim = getattr(hierarchical_neural_system, '_cached_sensory_input_dim_ref', None)
            cached_num_actions = getattr(hierarchical_neural_system, '_cached_num_actions_ref', None)
            cached_eeg_dim = getattr(hierarchical_neural_system, '_cached_eeg_dim_ref', None) 

            current_eeg_dim = current_config.get("EEG_SETTINGS",{}).get("feature_dim", 0) if current_config.get("EEG_SETTINGS",{}).get("enabled") else 0

            if (current_config['HIERARCHY_LEVEL_CONFIGS'] == cached_levels and
                current_config['SENSORY_INPUT_DIM'] == cached_sensory_dim and
                current_config['NUM_ACTIONS'] == cached_num_actions and
                current_eeg_dim == cached_eeg_dim and 
                hierarchical_neural_system.target_device == device):
                needs_new_system = False
            else:
                print("HNS parameters, dimensions, or device changed, recreating...")
                if current_config['HIERARCHY_LEVEL_CONFIGS'] != cached_levels: print("  Reason: HIERARCHY_LEVEL_CONFIGS changed.")
                if current_config['SENSORY_INPUT_DIM'] != cached_sensory_dim: print("  Reason: SENSORY_INPUT_DIM changed.")
                if current_config['NUM_ACTIONS'] != cached_num_actions: print("  Reason: NUM_ACTIONS changed.")
                if current_eeg_dim != cached_eeg_dim: print(f"  Reason: EEG feature_dim changed from {cached_eeg_dim} to {current_eeg_dim}.")
                if hierarchical_neural_system.target_device != device: print("  Reason: Target device changed.")


        if needs_new_system:
            print("Creating new Hierarchical NN System instance...")
            if hierarchical_neural_system: del hierarchical_neural_system; torch.cuda.empty_cache() if device.type == 'cuda' else None

            hierarchical_neural_system = HierarchicalSystemV5(
                level_configs=current_config['HIERARCHY_LEVEL_CONFIGS'], 
                target_device=device,
                verbose=False 
            )
            setattr(hierarchical_neural_system, '_cached_level_configs_ref', copy.deepcopy(current_config['HIERARCHY_LEVEL_CONFIGS']))
            setattr(hierarchical_neural_system, '_cached_sensory_input_dim_ref', current_config['SENSORY_INPUT_DIM'])
            setattr(hierarchical_neural_system, '_cached_num_actions_ref', current_config['NUM_ACTIONS'])
            current_eeg_dim = current_config.get("EEG_SETTINGS",{}).get("feature_dim", 0) if current_config.get("EEG_SETTINGS",{}).get("enabled") else 0
            setattr(hierarchical_neural_system, '_cached_eeg_dim_ref', current_eeg_dim) 

            print(f"New Hierarchical NN System ready on device: {device}")
            return True
        else:
            print("Existing Hierarchical NN System is compatible, reusing.")
            return False

    except Exception as e:
         print(f"FATAL: Failed to create/update Hierarchical NN System: {e}")
         traceback.print_exc()
         hierarchical_neural_system = None
         return False

def update_av_manager_instance():
    global av_manager, current_config
    enable_av = current_config.get("ENABLE_AV", False)
    enable_av_output = current_config.get("ENABLE_AV_OUTPUT", False)
    visualization_mode = current_config.get("VISUALIZATION_MODE", "none")

    if not AV_LIB_AVAILABLE and enable_av:
        print("Warning: AV Library not available, cannot enable AV Manager.")
        enable_av = False

    state_changed = False
    if enable_av:
        num_av_bots = current_config.get('NUM_LEARNING_BOTS', 0)
        av_data_source_level_name = current_config.get("AV_DATA_SOURCE_LEVEL_NAME", current_config.get("POLICY_HEAD_INPUT_LEVEL_NAME"))
        av_dim = get_level_dim_by_name(av_data_source_level_name, current_config['HIERARCHY_LEVEL_CONFIGS'])
        if av_dim is None:
            print(f"Warning: AV_DATA_SOURCE_LEVEL_NAME '{av_data_source_level_name}' not found. AV disabled.")
            enable_av = False; av_dim = 0

        needs_new_av_manager = True
        if av_manager and isinstance(av_manager, AVManager):
            if (av_manager.num_bots == num_av_bots and av_manager.dim == av_dim and
                av_manager.enable_audio_output == (enable_av_output and SOUNDDEVICE_AVAILABLE) and
                av_manager.visualization_mode == (visualization_mode if enable_av_output else 'none') and
                str(av_manager.device) == str(device)):
                needs_new_av_manager = False
            else:
                print("AV parameters changed, recreating AV Manager...")
                stop_av_system(av_manager); av_manager = None

        if needs_new_av_manager and enable_av :
            print(f"Creating new AV Manager instance (Mode: {visualization_mode}, Dim: {av_dim})...")
            if av_manager: stop_av_system(av_manager)
            sio_instance = socketio if visualization_mode == 'web' else None
            av_manager = setup_av_system(num_bots=num_av_bots, dim=av_dim, device=device, enable_output=enable_av_output, visualization_mode=visualization_mode, socketio_instance=sio_instance)
            if av_manager: print("New AV Manager created."); state_changed = True
            else: print("Warning: Failed to create new AV Manager instance.")
    else:
        if av_manager:
            print("Disabling and stopping AV Manager..."); stop_av_system(av_manager); av_manager = None; state_changed = True
    return state_changed

def update_eeg_processor_instance():
    """Creates or updates the global signal_processor instance."""
    global signal_processor, current_config
    eeg_settings = current_config.get("EEG_SETTINGS", {"enabled": False})
    needs_new_processor = False
    config_changed_due_to_feature_dim = False

    if eeg_settings.get("enabled", False) and EEG_LIB_AVAILABLE:
        if signal_processor is None:
            needs_new_processor = True
        else:
            if (signal_processor.sfreq != eeg_settings.get("sfreq") or
                signal_processor.input_channel_names != eeg_settings.get("channel_names") or
                signal_processor.processing_window_samples != eeg_settings.get("processing_window_samples") or
                str(signal_processor.processing_config) != str(eeg_settings.get("processing_config"))):
                 needs_new_processor = True
                 print("EEG parameters changed, recreating SignalProcessor...")

        if needs_new_processor:
            print("Creating new SignalProcessor instance...")
            try:
                signal_processor = SignalProcessor(
                    sfreq=eeg_settings["sfreq"],
                    channel_names=eeg_settings["channel_names"],
                    processing_config=eeg_settings["processing_config"],
                    processing_window_samples=eeg_settings["processing_window_samples"],
                    processing_interval_new_samples=eeg_settings.get("processing_interval_new_samples", 0)
                )
                feature_dim = calculate_eeg_feature_dim(eeg_settings, signal_processor)
                if current_config['EEG_SETTINGS'].get('feature_dim') != feature_dim:
                     current_config['EEG_SETTINGS']['feature_dim'] = feature_dim
                     config_changed_due_to_feature_dim = True
                print(f"New SignalProcessor created. Feature dimension: {feature_dim}")
            except Exception as e:
                print(f"Error creating SignalProcessor: {e}"); traceback.print_exc(); signal_processor = None
                if current_config['EEG_SETTINGS'].get('feature_dim') != 0:
                     current_config['EEG_SETTINGS']['feature_dim'] = 0
                     config_changed_due_to_feature_dim = True
    else:
        if signal_processor is not None:
            print("Disabling EEG SignalProcessor.")
            signal_processor = None
            if current_config['EEG_SETTINGS'].get('feature_dim', 0) != 0:
                 current_config['EEG_SETTINGS']['feature_dim'] = 0
                 config_changed_due_to_feature_dim = True 
    return config_changed_due_to_feature_dim


# ================================================================
# --- Simulation Environment ---
# ================================================================
class GridEnvironment:
    def __init__(self, size, num_goals, obstacles_factor_range, num_hc_bots, num_learning_bots, config_factors):
        self.size = max(10, int(size))
        self.num_goals = max(0, int(num_goals))
        self.min_obstacles_factor, self.max_obstacles_factor = obstacles_factor_range
        self.num_hc_bots = max(0, int(num_hc_bots)); self.num_learning_bots = max(0, int(num_learning_bots))
        self.config_factors = config_factors 
        self.obstacles = set(); self.goals = []; self.claimed_goals = set()
        self.start_positions = []; self._initial_goals = []
        try: self.randomize()
        except Exception as e: print(f"FATAL ERROR during environment init: {e}"); traceback.print_exc(); self.size=10; self.num_goals=0; self.num_hc_bots=0; self.num_learning_bots=0; self.goals=[]; self.obstacles=set(); self.start_positions=[]; self._initial_goals = []

    def _manhattan_distance(self, pos1, pos2):
        if not pos1 or not pos2 or 'x' not in pos1 or 'y' not in pos1 or 'x' not in pos2 or 'y' not in pos2: return float('inf')
        return abs(pos1['x'] - pos2['x']) + abs(pos1['y'] - pos2['y'])

    def randomize(self):
        self.obstacles.clear(); self.goals = []; self.claimed_goals.clear(); self.start_positions = []
        total_bots = self.num_hc_bots + self.num_learning_bots; total_cells = self.size * self.size
        required_items = total_bots + self.num_goals
        if total_cells <= 0: raise ValueError("Grid size must be positive.")
        occupied = set(); max_placement_attempts = max(required_items * 100, total_cells * 20); attempts = 0
        def is_valid_placement(pos_tuple, occupied_set, check_dists={}):
            if not (0 <= pos_tuple[0] < self.size and 0 <= pos_tuple[1] < self.size): return False
            if pos_tuple in occupied_set: return False
            pos_dict = {'x': pos_tuple[0], 'y': pos_tuple[1]}
            if 'goal_min_dist' in check_dists and any(self._manhattan_distance(pos_dict, g) < check_dists['goal_min_dist'] for g in self.goals): return False
            if 'bot_min_dist' in check_dists and any(self._manhattan_distance(pos_dict, sp) < check_dists['bot_min_dist'] for sp in self.start_positions): return False
            if 'bot_goal_min_dist' in check_dists and any(self._manhattan_distance(pos_dict, g) < check_dists['bot_goal_min_dist'] for g in self.goals): return False
            return True
        
        min_goal_dist = max(2, int(self.size * self.config_factors.get('goal_dist', 0.15)))
        goal_id_counter = 0; attempts = 0
        while len(self.goals) < self.num_goals and attempts < max_placement_attempts:
            attempts += 1; pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if is_valid_placement(pos, occupied, {'goal_min_dist': min_goal_dist}):
                self.goals.append({'x': pos[0], 'y': pos[1], 'id': f'G{goal_id_counter}'}); occupied.add(pos); goal_id_counter += 1
        
        min_bot_dist = max(3, int(self.size * self.config_factors.get('bot_dist', 0.25)))
        min_bot_goal_dist = max(3, int(self.size * self.config_factors.get('bot_goal_dist', 0.15)))
        attempts = 0; temp_bot_positions = []
        while len(temp_bot_positions) < total_bots and attempts < max_placement_attempts:
            attempts += 1; pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            temp_occupied = occupied.union(set(temp_bot_positions))
            if is_valid_placement(pos, temp_occupied, {'bot_min_dist': min_bot_dist, 'bot_goal_min_dist': min_bot_goal_dist}): temp_bot_positions.append(pos)
        
        hc_placed = 0; ln_placed = 0
        for idx, pos_tuple in enumerate(temp_bot_positions): 
             pos = pos_tuple 
             bot_type = 'Hardcoded' if idx < self.num_hc_bots else 'Learning'
             if bot_type == 'Hardcoded': bot_num = hc_placed; hc_placed += 1
             else: bot_num = ln_placed; ln_placed += 1
             self.start_positions.append({'x': pos[0], 'y': pos[1], 'type': bot_type, 'id': f'{bot_type[0]}{bot_num}'}); occupied.add(pos)

        if len(self.start_positions) < total_bots:
             print(f"CRITICAL Warning: Placed only {len(self.start_positions)}/{total_bots} bots. Adjusting counts."); self.num_hc_bots = hc_placed; self.num_learning_bots = ln_placed
        
        num_obstacles_to_place = random.randint(int(total_cells*self.min_obstacles_factor), int(total_cells*self.max_obstacles_factor)) if total_cells > 0 else 0
        attempts = 0; placed_obstacles = 0
        while placed_obstacles < num_obstacles_to_place and attempts < max_placement_attempts:
             attempts += 1; pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
             if is_valid_placement(pos, occupied): self.obstacles.add(pos); occupied.add(pos); placed_obstacles += 1
        
        self._initial_goals = [{'x': g['x'], 'y': g['y'], 'id': g['id']} for g in self.goals]
        print(f"Environment randomized: {len(self.goals)} goals, {len(self.start_positions)} bots, {len(self.obstacles)} obstacles.")

    def reset_round_state(self):
        self.claimed_goals.clear(); self.goals = [{'x': g['x'], 'y': g['y'], 'id': g['id']} for g in self._initial_goals]
    def is_valid(self, pos):
        if not pos or 'x' not in pos or 'y' not in pos: return False
        x, y = pos['x'], pos['y']; return 0 <= x < self.size and 0 <= y < self.size and (x, y) not in self.obstacles
    def find_path(self, start_pos, goal_pos, all_bots_dict=None, moving_bot_id=None):
        if not self.is_valid(start_pos) or goal_pos is None or not (0 <= goal_pos.get('x', -1) < self.size and 0 <= goal_pos.get('y', -1) < self.size): return None
        start_tuple = (start_pos['x'], start_pos['y']); goal_tuple = (goal_pos['x'], goal_pos['y']); current_obstacles = self.obstacles.copy()
        if all_bots_dict: current_obstacles.update({(b['pos']['x'], b['pos']['y']) for bid, b in all_bots_dict.items() if bid != moving_bot_id})
        queue = deque([(start_tuple, [])]); visited = {start_tuple}; deltas = [(0, -1, 0), (-1, 0, 1), (1, 0, 2), (0, 1, 3)]; max_path_len = self.size * self.size
        while queue:
            current_pos_tuple, path = queue.popleft();
            if len(path) >= max_path_len: continue
            for dx, dy, action in deltas:
                next_x, next_y = current_pos_tuple[0] + dx, current_pos_tuple[1] + dy; next_pos_tuple = (next_x, next_y)
                if next_pos_tuple == goal_tuple: return path + [action]
                if (0 <= next_x < self.size and 0 <= next_y < self.size and next_pos_tuple not in current_obstacles and next_pos_tuple not in visited):
                    visited.add(next_pos_tuple); queue.append((next_pos_tuple, path + [action]))
        return None

    def get_sensory_data(self, acting_bot, all_bots_dict, visibility_range):
        bot_pos = acting_bot['pos']; vis_range = max(1, int(visibility_range))
        senses = {'wall_distance_N': bot_pos['y'], 'wall_distance_S': self.size - 1 - bot_pos['y'], 'wall_distance_W': bot_pos['x'], 'wall_distance_E': self.size - 1 - bot_pos['x'], 'nearest_goal_dist': vis_range + 1, 'nearest_goal_dx': 0, 'nearest_goal_dy': 0, 'num_visible_goals': 0, 'nearest_opponent_dist': vis_range + 1, 'nearest_opponent_dx': 0, 'nearest_opponent_dy': 0, 'opponent_is_frozen': 0.0, 'opponent_type_HC': 0.0, 'opponent_type_LN': 0.0, 'opponent_type_PL': 0.0, 'self_is_frozen': 1.0 if acting_bot['freezeTimer'] > 0 else 0.0, '_visibleGoals': [], '_nearestOpponent': None, 'is_hallucinating': acting_bot.get('is_hallucinating_state', False) } 
        min_goal_dist = vis_range + 1; nearest_goal_obj = None
        for goal in self.goals:
             if goal['id'] not in self.claimed_goals:
                 dist = self._manhattan_distance(bot_pos, goal)
                 if dist <= vis_range: senses['num_visible_goals'] += 1; senses['_visibleGoals'].append({'x': goal['x'], 'y': goal['y'], 'id': goal['id'], 'dist': dist});
                 if dist < min_goal_dist: min_goal_dist = dist; nearest_goal_obj = goal
        senses['nearest_goal_dist'] = min_goal_dist
        if nearest_goal_obj: senses['nearest_goal_dx'] = nearest_goal_obj['x'] - bot_pos['x']; senses['nearest_goal_dy'] = nearest_goal_obj['y'] - bot_pos['y']
        min_opp_dist = vis_range + 1; nearest_opponent_obj = None
        for opp_id, opponent_bot in all_bots_dict.items():
             if opp_id == acting_bot['id']: continue
             dist = self._manhattan_distance(bot_pos, opponent_bot['pos'])
             if dist <= vis_range and dist < min_opp_dist: min_opp_dist = dist; nearest_opponent_obj = opponent_bot
        senses['nearest_opponent_dist'] = min_opp_dist
        if nearest_opponent_obj:
            senses['_nearestOpponent'] = nearest_opponent_obj; senses['nearest_opponent_dx'] = nearest_opponent_obj['pos']['x'] - bot_pos['x']; senses['nearest_opponent_dy'] = nearest_opponent_obj['pos']['y'] - bot_pos['y']; senses['opponent_is_frozen'] = 1.0 if nearest_opponent_obj['freezeTimer'] > 0 else 0.0;
            is_player = nearest_opponent_obj.get('is_player_controlled', False);
            if is_player: senses['opponent_type_PL'] = 1.0
            elif nearest_opponent_obj['type'] == 'Hardcoded': senses['opponent_type_HC'] = 1.0
            elif nearest_opponent_obj['type'] == 'Learning': senses['opponent_type_LN'] = 1.0
        senses['_visibleGoals'].sort(key=lambda g: g['dist']); return senses

    def perform_move_action(self, bot_pos, action_index):
        next_pos = bot_pos.copy(); delta = [(0, -1), (-1, 0), (1, 0), (0, 1)]
        if 0 <= action_index <= 3: dx, dy = delta[action_index]; next_pos['x'] += dx; next_pos['y'] += dy
        return next_pos
    def get_adjacent_unclaimed_goal(self, bot_pos):
        for goal in self.goals:
            if goal['id'] not in self.claimed_goals and self._manhattan_distance(bot_pos, goal) == 1: return goal
        return None
    def claim_goal(self, goal_id, bot_id):
        if goal_id in self.claimed_goals: return False
        if any(g['id'] == goal_id for g in self.goals): self.claimed_goals.add(goal_id); return True
        return False
    def are_all_goals_claimed(self): return len(self._initial_goals) > 0 and len(self.claimed_goals) >= len(self._initial_goals)
    def get_state(self):
        active_goals = [g for g in self.goals if g['id'] not in self.claimed_goals]
        return {'size': self.size, 'goals': active_goals, 'obstacles': list(self.obstacles), 'claimedGoals': list(self.claimed_goals)}

# ================================================================
# --- Bot Logic ---
# ================================================================
def get_hardcoded_action(bot_state, senses, env, all_bots_dict):
    """ Returns suggested_action (int) and mode_string """
    bot_id, pos = bot_state['id'], bot_state['pos']
    bot_state.setdefault('stuckCounter', 0); bot_state.setdefault('currentPath', None); bot_state.setdefault('lastAction', -1); bot_state.setdefault('targetGoalId', None); bot_state.setdefault('lastPos', {'x': -1, 'y': -1}); bot_state.setdefault('randomMoveCounter', 0)
    if pos == bot_state['lastPos']: bot_state['stuckCounter'] += 1
    else: bot_state['stuckCounter'] = 0; bot_state['lastPos'] = pos.copy(); bot_state['randomMoveCounter'] = 0
    
    adjacent_goal = env.get_adjacent_unclaimed_goal(pos)
    if adjacent_goal: bot_state['stuckCounter'] = 0; bot_state['currentPath'] = None; bot_state['targetGoalId'] = None; bot_state['randomMoveCounter'] = 0; return 5, f"Rule:Claim {adjacent_goal['id']}"
    nearest_opponent = senses.get('_nearestOpponent')
    if nearest_opponent and senses.get('nearest_opponent_dist', 99) == 1 and not senses.get('opponent_is_frozen'): bot_state['stuckCounter'] = 0; bot_state['currentPath'] = None; bot_state['targetGoalId'] = None; bot_state['randomMoveCounter'] = 0; return 4, f"Rule:Punch {nearest_opponent['id']}"
    
    if nearest_opponent and senses.get('nearest_opponent_dist', 99) <= current_config.get("MENTAL_ATTACK_RANGE", 3) and senses.get('nearest_opponent_dist', 99) >= 1 and not nearest_opponent.get('mental_attack_timer', 0) > 0 :
         bot_state['stuckCounter'] = 0; bot_state['currentPath'] = None; bot_state['targetGoalId'] = None; bot_state['randomMoveCounter'] = 0
         return 6, f"Rule:MAtt {nearest_opponent['id']}" 

    if bot_state['stuckCounter'] >= 5 and bot_state['randomMoveCounter'] < 3:
         bot_state['randomMoveCounter'] += 1; bot_state['currentPath'] = None; bot_state['targetGoalId'] = None; valid_moves = []
         for action_idx in range(4):
             next_p = env.perform_move_action(pos, action_idx); occupied_by_active = any(bid != bot_id and b['pos'] == next_p and b.get('freezeTimer', 0) <= 0 for bid, b in all_bots_dict.items())
             if env.is_valid(next_p) and not occupied_by_active: valid_moves.append(action_idx)
         if valid_moves: return random.choice(valid_moves), f"Rule:StuckRandom ({bot_state['stuckCounter']})"
         else: return -1, "Rule:StuckBlocked"
    elif bot_state['stuckCounter'] >= 5: return -1, f"Rule:Stuck ({bot_state['stuckCounter']})"
    current_path = bot_state.get('currentPath')
    if current_path:
        next_action = current_path[0]; intended_pos = env.perform_move_action(pos, next_action); is_pos_valid = env.is_valid(intended_pos); is_pos_occupied_by_other = any(other_id != bot_id and other_bot['pos'] == intended_pos for other_id, other_bot in all_bots_dict.items())
        if is_pos_valid and not is_pos_occupied_by_other:
            bot_state['currentPath'].pop(0); mode_str = f"Rule:Path ({len(bot_state['currentPath'])} left)";
            if not bot_state['currentPath']: bot_state['targetGoalId'] = None; mode_str = "Rule:Path End"
            bot_state['randomMoveCounter'] = 0; return next_action, mode_str
        else: bot_state['currentPath'] = None; bot_state['targetGoalId'] = None
    visible_goals = senses.get('_visibleGoals', []); target_goal_obj = None
    if bot_state['targetGoalId']:
        potential_target = next((g for g in visible_goals if g['id'] == bot_state['targetGoalId']), None)
        if potential_target: target_goal_obj = potential_target
        else: bot_state['targetGoalId'] = None
    if not target_goal_obj and visible_goals: target_goal_obj = visible_goals[0]; bot_state['targetGoalId'] = target_goal_obj['id']
    if target_goal_obj:
        path_to_goal = env.find_path(pos, target_goal_obj, all_bots_dict, bot_id)
        if path_to_goal:
            bot_state['currentPath'] = path_to_goal
            if bot_state['currentPath']:
                next_action = bot_state['currentPath'].pop(0); mode_str = f"Rule:NewPath ({len(bot_state['currentPath'])} left)"
                if not bot_state['currentPath']: bot_state['targetGoalId'] = None; mode_str="Rule:NewPath End"
                bot_state['randomMoveCounter'] = 0; return next_action, mode_str
            else: bot_state['targetGoalId'] = None; bot_state['currentPath'] = None
    valid_moves = []
    for action_idx in range(4):
        next_p = env.perform_move_action(pos, action_idx); occupied_by_active = any(bid != bot_id and b['pos'] == next_p and b.get('freezeTimer', 0) <= 0 for bid, b in all_bots_dict.items())
        if env.is_valid(next_p) and not occupied_by_active: valid_moves.append(action_idx)
    if not valid_moves: return -1, "Rule:Blocked"
    last_action = bot_state.get('lastAction', -1); reverse_action = -1
    if 0 <= last_action <= 3: reverse_map = {0: 3, 1: 2, 2: 1, 3: 0}; reverse_action = reverse_map.get(last_action)
    non_reverse_moves = [m for m in valid_moves if m != reverse_action]; chosen_move = -1
    if non_reverse_moves: chosen_move = random.choice(non_reverse_moves)
    elif valid_moves: chosen_move = random.choice(valid_moves)
    bot_state['currentPath'] = None; return chosen_move, f"Rule:Random ({bot_state['stuckCounter']})"

def _get_sensory_input_tensor_for_L0(bot_state, senses, config):
    """ Creates the base sensory tensor (excluding external signals like action/rules/EEG). """
    dim = config['SENSORY_INPUT_DIM']
    vis_range = config['VISIBILITY_RANGE']
    if dim <= 0: raise ValueError("SENSORY_INPUT_DIM must be positive.")
    features = []; bl = np
    def norm_capped(val, cap=vis_range): v = float(val) if val is not None else 0.0; c = float(cap); return 0.0 if c <= 0 else math.copysign(min(abs(v), c), v) / c
    
    features.append(norm_capped(senses.get('wall_distance_N', vis_range / 2))) 
    features.append(norm_capped(senses.get('wall_distance_S', vis_range / 2)))
    features.append(norm_capped(senses.get('wall_distance_W', vis_range / 2)))
    features.append(norm_capped(senses.get('wall_distance_E', vis_range / 2)))
    features.append(norm_capped(senses.get('nearest_goal_dist', vis_range + 1)))
    features.append(norm_capped(senses.get('nearest_goal_dx', 0)))
    features.append(norm_capped(senses.get('nearest_goal_dy', 0)))
    features.append(min(1.0, max(0.0, senses.get('num_visible_goals', 0) / 5.0))) 
    features.append(norm_capped(senses.get('nearest_opponent_dist', vis_range + 1)))
    features.append(norm_capped(senses.get('nearest_opponent_dx', 0)))
    features.append(norm_capped(senses.get('nearest_opponent_dy', 0)))
    features.append(float(senses.get('opponent_is_frozen', 0.0)))
    features.append(float(senses.get('opponent_type_HC', 0.0)))
    features.append(float(senses.get('opponent_type_LN', 0.0)))
    features.append(float(senses.get('opponent_type_PL', 0.0)))
    features.append(float(senses.get('self_is_frozen', 0.0))) 
    features.append(float(senses.get('is_hallucinating', 0.0))) 
    current_len = len(features)
    if current_len < dim: features.extend([0.0] * (dim - current_len))
    elif current_len > dim: print(f"Warning: Sensory feature vec ({current_len}) > SENSORY_INPUT_DIM ({dim}). Truncating."); features = features[:dim]
    
    try:
        np_features = np.array(features, dtype=np.float32)
        input_tensor = torch.from_numpy(np_features).to(device).unsqueeze(0).unsqueeze(0) 
        if input_tensor.shape != (1, 1, dim): raise ValueError(f"Shape mismatch: {input_tensor.shape}")
        return input_tensor
    except Exception as e:
        print(f"Error creating L0 input tensor: {e}"); traceback.print_exc()
        return torch.zeros((1, 1, dim), dtype=torch.float32, device=device)

def get_learning_action(bot_state: Dict[str, Any],
                        senses: Dict[str, Any], 
                        env: GridEnvironment,
                        all_bots_dict: Dict[str, Dict[str, Any]],
                        direct_player_action: Optional[int],
                        config: Dict[str, Any],
                        hns_system: Optional[HierarchicalSystemV5],
                        hns_device: torch.device,
                        hier_sensory_inputs: Dict[str, Tensor], 
                        hier_external_inputs: Dict[str, Tensor] 
                       ) -> Tuple[int, str, int, Optional[Dict[str, Any]]]:
    """ Determines the learning bot's action using HNS. Does NOT check freeze status. """
    bot_id = bot_state['id']
    chosen_action = -1
    mode_code = 5  # Default: Idle
    mode_str = "Idle"
    hns_diagnostics_output = None

    policy_head = bot_state.get('policy_head')
    if not hns_system or not HIERARCHICAL_NEURAL_LIB_AVAILABLE or not bot_state.get('memory_state') or not policy_head:
        mode_str = "Error (No HNS/State)"; mode_code = 5
        hns_diagnostics_output = None 
        bot_state['last_l0_reconstruction'] = None 
    else:
        try:
            new_retrieved_outputs_dict, next_bot_memory_states, current_anomalies, current_weight_changes, bu_norms, td_norms, ext_norms = hns_system.step(
                bot_state['memory_state'],
                bot_state['last_step_outputs_for_bot'],
                hier_sensory_inputs, 
                hier_external_inputs, 
                detach_next_states_memory=True
            )
            bot_state['memory_state'] = next_bot_memory_states
            bot_state['last_step_outputs_for_bot'] = {lname: {'retrieved': tens.detach().clone()} for lname, tens in new_retrieved_outputs_dict.items()}

            hns_diagnostics_output = {
                "retrieved_outputs": new_retrieved_outputs_dict, "anomalies": current_anomalies,
                "weight_changes": current_weight_changes, "bu_norms": bu_norms, "td_norms": td_norms, "ext_norms": ext_norms
            }
            l0_name = config['HIERARCHY_LEVEL_CONFIGS'][0]['name']
            bot_state['last_l0_reconstruction'] = new_retrieved_outputs_dict.get(l0_name) 

            current_anomalies_ema = bot_state.get('last_anomaly_proxy', {})
            for level_name, anomaly_tensor in current_anomalies.items():
                 current_anomalies_ema[level_name] = anomaly_tensor.item() * 0.1 + current_anomalies_ema.get(level_name, 0.0) * 0.9
            bot_state['last_anomaly_proxy'] = current_anomalies_ema

        except Exception as e:
            print(f"Error in HNS step for {bot_id}: {e}"); traceback.print_exc()
            mode_str = "Error (HNS Step)"; mode_code = 5
            hns_diagnostics_output = None; bot_state['last_l0_reconstruction'] = None
            chosen_action = random.choice(list(range(config['NUM_ACTIONS'])))
            return chosen_action, mode_str, mode_code, hns_diagnostics_output

    is_hallucinating = senses.get('is_hallucinating', False) 

    if is_hallucinating:
        policy_head.eval(); policy_head = policy_head.to(hns_device)
        output_level_name_for_policy = config['POLICY_HEAD_INPUT_LEVEL_NAME']
        policy_input_dim = get_level_dim_by_name(output_level_name_for_policy, config['HIERARCHY_LEVEL_CONFIGS'])
        policy_input_tensor = hns_diagnostics_output['retrieved_outputs'][output_level_name_for_policy] if hns_diagnostics_output and policy_input_dim else torch.zeros((1,1, policy_input_dim or 128), device=hns_device)
        with torch.no_grad(): action_logits = policy_head(policy_input_tensor.squeeze(0).squeeze(0))
        chosen_action = torch.argmax(action_logits, dim=-1).item()
        mode_str = f"Hallucinating (Predict {chosen_action})"
        mode_code = 6 
        
    elif bot_state.get('is_player_controlled', False):
        control_influence_percent = max(0.0, min(100.0, config['PLAYER_CONTROL_PERCENT']))
        player_action = -1; player_mode_str = "Player Idle"; player_mode_code = 5

        if direct_player_action is not None and 0 <= direct_player_action < config['NUM_ACTIONS']:
            player_action = direct_player_action; player_mode_str = f"Player Direct ({player_action})"; player_mode_code = 3; bot_state['target_coordinate'] = None
        elif bot_state.get('target_coordinate'):
            target = bot_state['target_coordinate']; current_pos = bot_state['pos']; dist = env._manhattan_distance(current_pos, target); player_mode_code = 4
            if dist == 0: player_action = -1; player_mode_str = "Player Target Reached"; bot_state['target_coordinate'] = None
            else:
                temp_action = -1
                if dist == 1:
                    opponent_at_target = next((b for bid, b in all_bots_dict.items() if bid != bot_id and b['pos'] == target and b['freezeTimer'] <= 0), None)
                    if opponent_at_target: temp_action = 4; player_mode_str = "Player Target Punch"
                    else:
                        goal_at_target = next((g for g in env.goals if g['id'] not in env.claimed_goals and g['x'] == target['x'] and g['y'] == target['y']), None)
                        if goal_at_target: temp_action = 5; player_mode_str = "Player Target Claim"
                if temp_action == -1: # If not punch or claim at dist 1, or dist > 1
                    # Check for Mental Attack possibility if target is within range and not self
                    if 1 <= dist <= config.get("MENTAL_ATTACK_RANGE", 3):
                        opponent_at_target_for_ma = next((b for bid,b in all_bots_dict.items() if b['pos']==target and bid != bot_id and b.get('mental_attack_timer',0) <=0 and b.get('freezeTimer',0) <=0), None)
                        if opponent_at_target_for_ma:
                            temp_action = 6 # Mental Attack
                            player_mode_str = "Player Target MAtt"

                if temp_action == -1: # Pathfind if no special action taken
                    path_to_target = env.find_path(current_pos, target, all_bots_dict, bot_id)
                    if path_to_target: temp_action = path_to_target[0]; player_mode_str = f"Player Target Move {temp_action}"
                    else: temp_action = -1; player_mode_str = "Player Target Blocked"
                player_action = temp_action
        
        if control_influence_percent < 100.0 and player_action != -1 and hns_diagnostics_output: 
            ai_action = -1; ai_mode_str = "AI Blend"
            policy_head.eval(); policy_head = policy_head.to(hns_device)
            output_level_name_for_policy = config['POLICY_HEAD_INPUT_LEVEL_NAME']
            policy_input_dim = get_level_dim_by_name(output_level_name_for_policy, config['HIERARCHY_LEVEL_CONFIGS'])
            policy_input_tensor = hns_diagnostics_output['retrieved_outputs'][output_level_name_for_policy] if policy_input_dim else torch.zeros((1,1, policy_input_dim or 128), device=hns_device)
            with torch.no_grad(): action_logits = policy_head(policy_input_tensor.squeeze(0).squeeze(0))
            ai_action = torch.argmax(action_logits, dim=-1).item()

            if random.uniform(0, 100) < control_influence_percent:
                chosen_action = player_action; mode_str = player_mode_str; mode_code = player_mode_code
            else:
                chosen_action = ai_action; mode_str = ai_mode_str; mode_code = 0
        else: 
            chosen_action = player_action; mode_str = player_mode_str; mode_code = player_mode_code
            
    else: 
        base_explore_rate = config['LEARNING_BOT_BASE_EXPLORATION_RATE']
        output_level_name = config['POLICY_HEAD_INPUT_LEVEL_NAME']
        anomaly_val = bot_state.get('last_anomaly_proxy', {}).get(output_level_name, 0.0)
        anomaly_factor = min(3.0, 1.0 + anomaly_val * 10.0)
        current_explore_thresh = min(98.0, base_explore_rate * anomaly_factor)
        is_exploring = random.uniform(0, 100) < current_explore_thresh

        if is_exploring:
            if random.uniform(0, 100) < config['LEARNING_BOT_RULE_EXPLORE_PERCENT']:
                mode_code = 2; hc_action, hc_mode = get_hardcoded_action(bot_state, senses, env, all_bots_dict)
                chosen_action = hc_action; mode_str = f"Explore Rule ({current_explore_thresh:.1f}%) -> {hc_mode}"
            else:
                mode_code = 1; chosen_action = random.choice(list(range(config['NUM_ACTIONS'])))
                mode_str = f"Explore Random ({current_explore_thresh:.1f}%)"
        else: 
            mode_code = 0
            policy_head.eval(); policy_head = policy_head.to(hns_device)
            output_level_name_for_policy = config['POLICY_HEAD_INPUT_LEVEL_NAME']
            policy_input_dim = get_level_dim_by_name(output_level_name_for_policy, config['HIERARCHY_LEVEL_CONFIGS'])
            policy_input_tensor = hns_diagnostics_output['retrieved_outputs'][output_level_name_for_policy] if hns_diagnostics_output and policy_input_dim else torch.zeros((1,1, policy_input_dim or 128), device=hns_device)
            with torch.no_grad(): action_logits = policy_head(policy_input_tensor.squeeze(0).squeeze(0))
            chosen_action = torch.argmax(action_logits, dim=-1).item()
            mode_str = f"Exploit (Predict {chosen_action})"

    if not isinstance(chosen_action, int) or chosen_action < -1 or chosen_action >= config['NUM_ACTIONS']:
        chosen_action = -1
        if mode_code not in [5, 6]: mode_str += " -> IdleFallback" 

    return chosen_action, mode_str, mode_code, hns_diagnostics_output


# ================================================================
# --- Simulation Setup & Control ---
# ================================================================
def create_learning_bot_instance(bot_id, start_pos, config):
    global hierarchical_neural_system, device
    if not hierarchical_neural_system or not HIERARCHICAL_NEURAL_LIB_AVAILABLE:
         raise RuntimeError(f"HNS not ready. Cannot create learning bot {bot_id}.")

    initial_hier_mem_states = hierarchical_neural_system.get_initial_states()
    initial_last_step_outputs = {
        level_config['name']: {'retrieved': torch.zeros((1, 1, level_config['dim']), device=device, dtype=torch.float32)}
        for level_config in config['HIERARCHY_LEVEL_CONFIGS']
    }

    policy_head_input_level_name = config['POLICY_HEAD_INPUT_LEVEL_NAME']
    policy_head_input_dim = get_level_dim_by_name(policy_head_input_level_name, config['HIERARCHY_LEVEL_CONFIGS'])
    if policy_head_input_dim is None: raise ValueError(f"Policy head input level '{policy_head_input_level_name}' not found or dim missing.")

    print(f"Creating Learning Bot {bot_id} (PolicyInDim={policy_head_input_dim}, Actions={config['NUM_ACTIONS']}) on {device}")
    try:
        policy_head = nn.Linear(policy_head_input_dim, config['NUM_ACTIONS']).to(device); nn.init.xavier_uniform_(policy_head.weight); nn.init.zeros_(policy_head.bias) if policy_head.bias is not None else None
    except Exception as e: print(f"FATAL ERROR: Failed policy head creation for {bot_id}: {e}"); traceback.print_exc(); raise

    return {
        'id': bot_id, 'type': 'Learning', 'pos': start_pos.copy(), 'steps': 0, 'goalsReachedThisRound': 0, 'goalsReachedTotal': 0,
        'freezeTimer': 0, 'mental_attack_timer': 0, 'lastAction': -1, 'mode': 'Init', 'senses': {},
        'memory_state': initial_hier_mem_states, 'last_step_outputs_for_bot': initial_last_step_outputs,
        'policy_head': policy_head, 'last_anomaly_proxy': {}, 'is_player_controlled': False, 'target_coordinate': None,
        'original_bot_id': bot_id, 'lastPos': {'x':-1,'y':-1}, 'stuckCounter': 0, 'lastMoveAttempt': -1,
        'currentPath': None, 'targetGoalId': None, 'randomMoveCounter': 0, 'last_av_data': None,
        'last_l0_reconstruction': None, 
        'is_hallucinating_state': False, 
        'last_eeg_features': None, 
    }

def create_hardcoded_bot_instance(bot_id, start_pos):
     return {
         'id': bot_id, 'type': 'Hardcoded', 'pos': start_pos.copy(), 'steps': 0, 'goalsReachedThisRound': 0, 'goalsReachedTotal': 0,
         'freezeTimer': 0, 'mental_attack_timer': 0, 'lastAction': -1, 'mode': 'Init', 'senses': {}, 'lastPos': {'x':-1,'y':-1},
         'stuckCounter': 0, 'lastMoveAttempt': -1, 'currentPath': None, 'targetGoalId': None, 'randomMoveCounter': 0,
         'is_hallucinating_state': False, 
     }

def setup_simulation(full_reset=False, new_environment=False):
    global environment, bots, round_number, stats, current_config, players, player_direct_actions, av_manager, hierarchical_neural_system, signal_processor, device
    print(f"--- Setting up Simulation (Full Reset: {full_reset}, New Env: {new_environment}) ---")
    if av_manager: stop_av_system(av_manager); av_manager = None

    eeg_config_changed = update_eeg_processor_instance() 
    hns_config_changed = configure_hns_external_inputs()
    hns_recreated = False
    if eeg_config_changed or hns_config_changed or full_reset:
        hns_recreated = update_hierarchical_neural_system_instance()
        if not hierarchical_neural_system: print("CRITICAL: HNS update failed. Aborting setup."); return False

    if full_reset:
        print("Performing full reset..."); round_number = 0; stats = {'hc_total_goals': 0, 'learning_total_goals': 0}
        print("Clearing existing bot states..."); bots.clear(); players.clear(); player_direct_actions.clear()
        environment = None; new_environment = True 
        if device.type == 'cuda': torch.cuda.empty_cache()
    else: 
        round_number += 1
        env_structure_keys = ['GRID_SIZE', 'NUM_HC_BOTS', 'NUM_LEARNING_BOTS', 'NUM_GOALS']
        
        env_config_mismatch = any(
            current_config[k] != getattr(environment, k.lower().replace('num_', 'num_'), None)
            for k in env_structure_keys if environment and hasattr(environment, k.lower().replace('num_', 'num_'))
        )
        
        env_bot_count_mismatch = (
            (environment and environment.num_hc_bots != current_config['NUM_HC_BOTS']) or
            (environment and environment.num_learning_bots != current_config['NUM_LEARNING_BOTS'])
        )
        
        env_structure_changed = env_config_mismatch or env_bot_count_mismatch

        if not environment or env_structure_changed:
            print("Env structure change or missing, forcing new env...")
            new_environment = True; environment = None
        elif new_environment or current_config.get("RANDOMIZE_ENV_PER_ROUND", False):
            print("Randomizing environment...")
            environment.randomize()
        else:
            environment.reset_round_state() 
        player_direct_actions.clear()


    if environment is None: 
         print("Recreating environment...")
         try:
             obstacle_range = (current_config['OBSTACLES_FACTOR_MIN'], current_config['OBSTACLES_FACTOR_MAX'])
             dist_factors = {
                 'goal_dist': current_config.get('MIN_GOAL_START_DISTANCE_FACTOR'),
                 'bot_dist': current_config.get('MIN_BOT_START_DISTANCE_FACTOR'),
                 'bot_goal_dist': current_config.get('MIN_BOT_GOAL_DISTANCE_FACTOR')
             }
             environment = GridEnvironment(current_config['GRID_SIZE'], current_config['NUM_GOALS'], obstacle_range, current_config['NUM_HC_BOTS'], current_config['NUM_LEARNING_BOTS'], dist_factors)
             
             if environment.num_hc_bots != current_config['NUM_HC_BOTS'] or \
                environment.num_learning_bots != current_config['NUM_LEARNING_BOTS']:
                 print(f"Env adjusted bot counts: HC={environment.num_hc_bots}, Lrn={environment.num_learning_bots}. Updating config.")
                 current_config['NUM_HC_BOTS'] = environment.num_hc_bots
                 current_config['NUM_LEARNING_BOTS'] = environment.num_learning_bots
             full_reset = True 
         except Exception as e: print(f"FATAL: Environment creation failed: {e}"); traceback.print_exc(); return False

    new_bots = {}; bot_starts = environment.start_positions if environment else []
    required_bots = current_config['NUM_HC_BOTS'] + current_config['NUM_LEARNING_BOTS']
    
    if len(bot_starts) != required_bots and environment:
         print(f"Warning: Mismatch between env starts ({len(bot_starts)}) and updated config counts ({required_bots}). Re-randomizing env once.")
         environment.num_hc_bots = current_config['NUM_HC_BOTS'] 
         environment.num_learning_bots = current_config['NUM_LEARNING_BOTS']
         environment.randomize()
         bot_starts = environment.start_positions
         if len(bot_starts) != required_bots:
              print(f"FATAL MISMATCH: Env starts ({len(bot_starts)}) still != counts ({required_bots}) after re-randomize. Aborting."); return False


    try:
        for start_pos_data in bot_starts:
            bot_id = start_pos_data['id']; bot_type = start_pos_data['type']; start_pos = {'x': start_pos_data['x'], 'y': start_pos_data['y']}
            controlling_sid = next((sid for sid, p_data in players.items() if p_data['original_bot_id'] == bot_id), None)
            
            if bot_id in bots and not full_reset and not hns_recreated:
                existing_bot = bots[bot_id]
                existing_bot.update({
                    'pos': start_pos.copy(), 'steps': 0, 'goalsReachedThisRound': 0, 'freezeTimer': 0, 'mental_attack_timer': 0,
                    'lastAction': -1, 'mode': 'Reset', 'senses': {}, 'lastPos': {'x':-1,'y':-1}, 'stuckCounter': 0,
                    'lastMoveAttempt': -1, 'currentPath': None, 'targetGoalId': None, 'randomMoveCounter': 0,
                    'last_av_data': None, 'is_hallucinating_state': False, 'last_l0_reconstruction': None,
                    'last_eeg_features': None 
                })
                if bot_type == 'Learning':
                     existing_bot['last_anomaly_proxy'] = {}; existing_bot['target_coordinate'] = None
                     if hierarchical_neural_system: 
                         existing_bot['memory_state'] = hierarchical_neural_system.get_initial_states()
                         existing_bot['last_step_outputs_for_bot'] = {
                             level_config['name']: {'retrieved': torch.zeros((1, 1, level_config['dim']), device=device, dtype=torch.float32)}
                             for level_config in current_config['HIERARCHY_LEVEL_CONFIGS'] }
                     existing_bot['is_player_controlled'] = bool(controlling_sid)
                new_bots[bot_id] = existing_bot
            else: 
                if bot_id in bots: del bots[bot_id] 
                if bot_type == 'Hardcoded': new_bots[bot_id] = create_hardcoded_bot_instance(bot_id, start_pos)
                elif bot_type == 'Learning':
                    if hierarchical_neural_system:
                         new_bots[bot_id] = create_learning_bot_instance(bot_id, start_pos, current_config)
                         if controlling_sid: new_bots[bot_id]['is_player_controlled'] = True; players[controlling_sid]['player_bot_id'] = bot_id
                    else: print(f"Warning: Cannot create new learning bot {bot_id}, HNS not ready.")
        bots = new_bots
        for sid, player_data in list(players.items()): 
            if player_data['original_bot_id'] not in bots: print(f"Removing player {sid}, bot {player_data['original_bot_id']} no longer exists."); del players[sid]
    except Exception as e: print(f"Error: Bot creation/reset failed: {e}"); traceback.print_exc(); return False

    av_state_changed = update_av_manager_instance()

    print(f"Setup complete for Round {round_number}. Active Bots: {list(bots.keys())}")
    socketio.emit('config_update', current_config) 
    return True


# --- Simulation Step ---
def simulation_step():
    global player_direct_actions, av_manager, hierarchical_neural_system, signal_processor, device, current_config
    if not environment or not bots: return False

    round_over = False; max_steps_reached_for_all = True
    bot_ids_this_step = list(bots.keys()); current_direct_actions = player_direct_actions.copy(); player_direct_actions.clear()
    live_av_data = {}; learning_bot_ids = sorted([bid for bid, b in bots.items() if b['type'] == 'Learning'])
    bot_id_to_av_idx = {bot_id: idx for idx, bot_id in enumerate(learning_bot_ids)}

    eeg_settings = current_config.get("EEG_SETTINGS", {"enabled": False})
    current_eeg_features_tensor = None 
    processed_eeg_this_step = False
    if signal_processor and eeg_settings.get("enabled", False):
        n_channels = len(eeg_settings["channel_names"])
        samples_per_step = max(1, int(current_config['SIMULATION_SPEED_MS'] / 1000.0 * eeg_settings['sfreq']))
        
        if eeg_settings.get("simulate_eeg"):
            sim_eeg_chunk = np.random.randn(n_channels, samples_per_step) * 5 
            signal_processor.append_data(sim_eeg_chunk)
        
        if signal_processor.should_process():
            eeg_window = signal_processor.get_processing_window()
            if eeg_window is not None:
                try:
                    processed_window = signal_processor.preprocess(eeg_window)
                    nf_metrics, _ = signal_processor.extract_features(processed_window, nf_protocol_config={})
                    feature_vector = flatten_nf_metrics(nf_metrics, signal_processor, eeg_settings.get("flattening_order", ["band_power", "coherence"]))
                    expected_dim = current_config.get('EEG_SETTINGS',{}).get('feature_dim', 0)
                    if len(feature_vector) == expected_dim and expected_dim > 0 :
                        current_eeg_features_tensor = torch.tensor(feature_vector, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                        processed_eeg_this_step = True
                    elif expected_dim > 0 : print(f"EEG Feature mismatch! Expected {expected_dim}, Got {len(feature_vector)}")
                except Exception as e: print(f"Error processing EEG step: {e}")

    for bot_id in bot_ids_this_step:
        if bot_id not in bots: continue
        bot_state = bots[bot_id]
        if bot_state['steps'] >= current_config['MAX_STEPS_PER_ROUND']: continue
        else: max_steps_reached_for_all = False

        action_attempted = -1; mode_str = "Idle"; mode_code = 5; hns_step_diagnostics = None
        
        is_frozen = bot_state['freezeTimer'] > 0
        is_attacked = bot_state.get('mental_attack_timer', 0) > 0
        bot_state['is_hallucinating_state'] = False 

        hier_sensory_inputs = None
        l0_name = current_config['HIERARCHY_LEVEL_CONFIGS'][0]['name']
        if bot_state['type'] == 'Learning' and is_attacked and current_config.get("MENTAL_ATTACK_USES_RECONSTRUCTION", False):
            l0_reconstruction = bot_state.get('last_l0_reconstruction')
            sensory_input_dim = current_config['SENSORY_INPUT_DIM']
            if l0_reconstruction is not None and l0_reconstruction.shape[-1] == sensory_input_dim:
                 sensory_tensor_L0 = l0_reconstruction.detach().clone()
            else: 
                 bot_state['senses'] = environment.get_sensory_data(bot_state, bots, current_config['VISIBILITY_RANGE']) 
                 sensory_tensor_L0 = _get_sensory_input_tensor_for_L0(bot_state, bot_state['senses'], current_config)
            bot_state['is_hallucinating_state'] = True 
            hier_sensory_inputs = {l0_name: sensory_tensor_L0}
        else: 
            bot_state['senses'] = environment.get_sensory_data(bot_state, bots, current_config['VISIBILITY_RANGE'])
            if bot_state['type'] == 'Learning':
                 sensory_tensor_L0 = _get_sensory_input_tensor_for_L0(bot_state, bot_state['senses'], current_config)
                 hier_sensory_inputs = {l0_name: sensory_tensor_L0}
            if is_attacked: 
                bot_state['is_hallucinating_state'] = True


        hier_external_inputs = {}
        last_action = bot_state.get('lastAction', -1)
        motor_signal_name = "motor_efference_copy" 
        last_action_one_hot = torch.zeros(1, 1, current_config['NUM_ACTIONS'], device=device)
        if 0 <= last_action < current_config['NUM_ACTIONS']: last_action_one_hot[0, 0, last_action] = 1.0
        hier_external_inputs[motor_signal_name] = last_action_one_hot

        if bot_state['type'] == 'Learning' and current_config.get("USE_RULES_AS_HNS_INPUT", False):
             rule_signal_name = "hardcoded_rule_guidance"
             suggested_action, _ = get_hardcoded_action(bot_state, bot_state['senses'], environment, bots)
             rule_guidance_one_hot = torch.zeros(1, 1, current_config['NUM_ACTIONS'], device=device)
             if 0 <= suggested_action < current_config['NUM_ACTIONS']: rule_guidance_one_hot[0, 0, suggested_action] = 1.0
             hier_external_inputs[rule_signal_name] = rule_guidance_one_hot

        eeg_signal_name = eeg_settings.get("hns_signal_name")
        if eeg_signal_name and eeg_settings.get("enabled", False) and bot_state['type'] == 'Learning':
            if processed_eeg_this_step and current_eeg_features_tensor is not None: 
                 bot_state['last_eeg_features'] = current_eeg_features_tensor 
            last_eeg_tensor = bot_state.get('last_eeg_features')
            if last_eeg_tensor is not None: 
                 hier_external_inputs[eeg_signal_name] = last_eeg_tensor

        action_intended = -1
        try:
            if bot_state['type'] == 'Learning':
                direct_action = None
                if bot_state.get('is_player_controlled', False):
                    controlling_sid = next((sid for sid, p_data in players.items() if p_data['player_bot_id'] == bot_id), None)
                    if controlling_sid and controlling_sid in current_direct_actions: direct_action = current_direct_actions[controlling_sid]

                action_intended, mode_str, mode_code, hns_step_diagnostics = get_learning_action(
                    bot_state, bot_state['senses'], environment, bots, direct_action, current_config,
                    hierarchical_neural_system, device,
                    hier_sensory_inputs=hier_sensory_inputs, 
                    hier_external_inputs=hier_external_inputs 
                )

            elif bot_state['type'] == 'Hardcoded':
                action_intended, mode_str = get_hardcoded_action(bot_state, bot_state['senses'], environment, bots)
                mode_code = 6 if bot_state['is_hallucinating_state'] else 2 

        except Exception as e:
             print(f"CRITICAL Error determining action for {bot_id}: {e}"); traceback.print_exc()
             action_intended = -1; mode_str = "Error Action"; mode_code = 5

        if is_frozen:
            action_attempted = -1 
            if bot_state['type'] == 'Learning': mode_str = "Frozen (Thinking)"
            else: mode_str = "Frozen"
            mode_code = 5 
        else:
            action_attempted = action_intended 

        bot_state['mode'] = mode_str
        next_pos = bot_state['pos'].copy()
        action_executed_str = "Idle"

        if action_attempted != -1:
            if 0 <= action_attempted <= 3: 
                intended_pos = environment.perform_move_action(bot_state['pos'], action_attempted)
                occupied = any(bid != bot_id and b['pos'] == intended_pos for bid, b in bots.items())
                if environment.is_valid(intended_pos) and not occupied:
                    next_pos = intended_pos; action_executed_str = f"Move {action_attempted}"
                else: bot_state['mode'] += " (Blocked)"; action_executed_str = f"Move {action_attempted} Blocked"
            elif action_attempted == 4: 
                target_bot = next((ob for ob_id, ob in bots.items() if ob_id != bot_id and environment._manhattan_distance(bot_state['pos'], ob['pos']) == 1 and ob['freezeTimer'] <= 0 and ob.get('mental_attack_timer', 0) <=0), None)
                if target_bot: target_bot['freezeTimer'] = current_config['FREEZE_DURATION']; bot_state['mode'] += f" (Hit {target_bot['id']})"; action_executed_str = f"Punch Hit {target_bot['id']}"
                else: bot_state['mode'] += " (Punch Miss)"; action_executed_str = "Punch Miss"
            elif action_attempted == 5: 
                adj_goal = environment.get_adjacent_unclaimed_goal(bot_state['pos'])
                if adj_goal and environment.claim_goal(adj_goal['id'], bot_id):
                     bot_state['goalsReachedThisRound'] += 1; bot_state['goalsReachedTotal'] += 1
                     if bot_state['type'] == 'Hardcoded': stats['hc_total_goals'] += 1
                     else: stats['learning_total_goals'] += 1
                     bot_state['mode'] += f" (Claimed {adj_goal['id']})"; action_executed_str = f"Claimed {adj_goal['id']}"
                     if environment.are_all_goals_claimed(): round_over = True; print(f"--- Round {round_number} Over: All goals claimed! ---")
                else: bot_state['mode'] += " (Claim Fail)"; action_executed_str = "Claim Fail"
            elif action_attempted == 6: 
                bot_state['mode'] += " (MAtt)"
                potential_targets = []
                for ob_id, ob_state in bots.items():
                    if ob_id == bot_id or ob_state['freezeTimer'] > 0: continue 
                    dist = environment._manhattan_distance(bot_state['pos'], ob_state['pos'])
                    if 1 <= dist <= current_config['MENTAL_ATTACK_RANGE']: potential_targets.append(ob_state)
                if potential_targets:
                    potential_targets.sort(key=lambda t: environment._manhattan_distance(bot_state['pos'], t['pos']))
                    target_bot_to_attack = potential_targets[0]
                    target_bot_to_attack['mental_attack_timer'] = current_config['MENTAL_ATTACK_DURATION']
                    bot_state['mode'] += f" (Hit {target_bot_to_attack['id']})"; action_executed_str = f"MAtt Hit {target_bot_to_attack['id']}"
                else: bot_state['mode'] += " (NoTgt)"; action_executed_str = "MAtt NoTgt"

        bot_state['pos'] = next_pos; bot_state['steps'] += 1
        if bot_state['freezeTimer'] > 0: bot_state['freezeTimer'] -= 1
        if bot_state.get('mental_attack_timer', 0) > 0: bot_state['mental_attack_timer'] -=1
        bot_state['lastAction'] = action_attempted 

        if bot_state['type'] == 'Learning' and hns_step_diagnostics:
             try:
                 anomaly_l0_tensor = hns_step_diagnostics['anomalies'].get(current_config['HIERARCHY_LEVEL_CONFIGS'][0]['name'], None)
                 anomaly_l2_tensor = hns_step_diagnostics['anomalies'].get(current_config['POLICY_HEAD_INPUT_LEVEL_NAME'], None)
                 wc_l0_tensor = hns_step_diagnostics['weight_changes'].get(current_config['HIERARCHY_LEVEL_CONFIGS'][0]['name'], None)

                 anomaly_l0 = anomaly_l0_tensor.item() if anomaly_l0_tensor is not None else 0.0
                 anomaly_l2 = anomaly_l2_tensor.item() if anomaly_l2_tensor is not None else 0.0
                 wc_l0 = wc_l0_tensor.item() if wc_l0_tensor is not None else 0.0

                 chat_msg = f"Action: {action_executed_str} (Mode: {bot_state['mode']}). "
                 chat_parts = []
                 if anomaly_l2 > 0.1: chat_parts.append(f"High Exec Surprise ({anomaly_l2:.3f})")
                 if anomaly_l0 > 0.05: chat_parts.append(f"Sensory Err ({anomaly_l0:.3f})")
                 if wc_l0 > 0.01: chat_parts.append(f"Mem Update ({wc_l0:.4f})")
                 if is_frozen: chat_parts.append("Frozen!")
                 if bot_state['is_hallucinating_state']: chat_parts.append("Hallucinating!") 

                 if chat_parts: chat_msg += "State: " + ", ".join(chat_parts) + "."
                 
                 socketio.emit('bot_chat', {'bot_id': bot_id, 'message': chat_msg})
             except Exception as chat_e: print(f"Error generating chat for {bot_id}: {chat_e}")

        if bot_state['type'] == 'Learning' and hns_step_diagnostics and not is_frozen: 
            av_idx = bot_id_to_av_idx.get(bot_id)
            if current_config.get("ENABLE_AV", False) and av_manager and av_idx is not None and \
               (av_manager.enable_audio_output or av_manager.enable_visual_output):
                av_data_source_level = current_config.get("AV_DATA_SOURCE_LEVEL_NAME", current_config.get("POLICY_HEAD_INPUT_LEVEL_NAME"))
                default_dim = get_level_dim_by_name(av_data_source_level, current_config['HIERARCHY_LEVEL_CONFIGS']) or 1
                
                av_retrieved_tensor = hns_step_diagnostics['retrieved_outputs'].get(av_data_source_level, None)
                av_anomalies_tensor = hns_step_diagnostics['anomalies'].get(av_data_source_level, None)
                av_wc_tensor = hns_step_diagnostics['weight_changes'].get(av_data_source_level, None)

                av_retrieved = av_retrieved_tensor if av_retrieved_tensor is not None else torch.zeros((1,1,default_dim), device=device)
                av_anomalies = av_anomalies_tensor if av_anomalies_tensor is not None else torch.tensor(0.0, device=device)
                av_wc = av_wc_tensor if av_wc_tensor is not None else torch.tensor(0.0, device=device)
                
                live_av_data[av_idx] = {
                    'anomaly_score': av_anomalies, 
                    'retrieved_memory_vector': av_retrieved, 
                    'weight_change_metric': av_wc, 
                    'input_stream_vector': av_retrieved, 
                    'mode_code': mode_code, 
                    'is_player_controlled': bot_state.get('is_player_controlled', False) 
                }
    
    if current_config.get("ENABLE_AV", False) and av_manager and live_av_data:
        update_av_system(av_manager, live_av_data)

    if not round_over and max_steps_reached_for_all: round_over = True; print(f"--- Round {round_number} Over: Max steps reached! ---")
    return not round_over


# --- Simulation Loop --- 
def simulation_loop():
    global simulation_running, round_number, simulation_loop_task
    print("Simulation loop started."); loop_count = 0; emit_interval_steps = 2; last_emit_time = time.monotonic(); min_emit_interval_time = 0.04
    while simulation_running:
        loop_start_time = time.monotonic()
        try:
            continue_round = simulation_step()
            if not simulation_running: break
            if not continue_round:
                 new_env_next = current_config.get("RANDOMIZE_ENV_PER_ROUND", False)
                 if setup_simulation(full_reset=False, new_environment=new_env_next):
                     loop_count = 0; last_emit_time = time.monotonic(); emit_state(); print(f"Starting Round {round_number}...")
                 else:
                     print("Error: Failed next round setup. Stopping."); simulation_running = False;
                     socketio.emit('simulation_stopped', {'message': 'Error setting up next round.'}); break
            else:
                 loop_count += 1; current_time = time.monotonic()
                 if loop_count % emit_interval_steps == 0 or (current_time - last_emit_time) > min_emit_interval_time:
                     emit_state(); last_emit_time = current_time
            elapsed_time = time.monotonic() - loop_start_time; target_delay = current_config['SIMULATION_SPEED_MS'] / 1000.0
            delay = max(0.001, target_delay - elapsed_time); eventlet.sleep(delay)
        except Exception as e:
            print(f"CRITICAL Error in simulation loop: {e}"); traceback.print_exc();
            simulation_running = False; socketio.emit('simulation_stopped', {'message': f'Runtime Error: {e}'}); break
    print("Simulation loop finished."); emit_state(); simulation_loop_task = None


# ================================================================
# --- Flask Routes & SocketIO Events ---
# ================================================================
@app.route('/')
def index():
    try: return render_template('index.html')
    except Exception as e: print(f"Error rendering template: {e}"); traceback.print_exc(); return "Error loading page.", 500

@app.route('/visuals')
def visuals_page():
    return """<!DOCTYPE html><html><head><title>Web Visualizer</title><script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.min.js"></script><style>body{font-family:sans-serif;background-color:#222;color:#eee;}#data{white-space:pre-wrap;font-family:monospace;}</style></head><body><h1>Web Visualizer Placeholder</h1><p>Connects to SocketIO namespace '/visuals'. Data will appear below.</p><div id="status">Connecting...</div><div id="data"></div><script>const socket=io('/visuals');socket.on('connect',()=>{document.getElementById('status').textContent='Connected to /visuals';});socket.on('disconnect',()=>{document.getElementById('status').textContent='Disconnected from /visuals';});socket.on('visual_update',(data)=>{document.getElementById('data').textContent=JSON.stringify(data,null,2);});</script></body></html>"""

class VisualsNamespace(Namespace):
    def on_connect(self): print(f"Web Visualizer client connected: {request.sid}")
    def on_disconnect(self): print(f"Web Visualizer client disconnected: {request.sid}")
socketio.on_namespace(VisualsNamespace('/visuals'))

@socketio.on('connect')
def handle_connect():
    sid = request.sid; print(f"Client connected: {sid}")
    try:
        if environment is None or not bots:
             print("First connection, ensuring initial setup...")
             if not setup_simulation(full_reset=True, new_environment=True): emit('status_update', {'message': 'Error: Server setup failed.'}, room=sid); return
        state = get_game_state(); state['isRunning'] = simulation_running
        emit('initial_state', state, room=sid); print(f"Initial state sent to {sid} (Running: {simulation_running})")
    except Exception as e: print(f"Error sending initial data to {sid}: {e}"); traceback.print_exc()

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid; print(f"Client disconnected: {sid}")
    if sid in players:
        player_data = players.pop(sid); original_bot_id = player_data['original_bot_id']
        print(f"Player {sid} released control of bot {original_bot_id}")
        if original_bot_id in bots: bots[original_bot_id]['is_player_controlled'] = False; bots[original_bot_id]['target_coordinate'] = None; bots[original_bot_id]['mode'] = "AI Control"; socketio.emit('player_left', {'player_id': original_bot_id}); emit_state()
    if sid in player_direct_actions: del player_direct_actions[sid]

def get_game_state():
    serializable_bots = {}
    for bot_id, bot_state in bots.items():
        s_bot = {
            'id': bot_state['id'], 'type': bot_state['type'], 'pos': bot_state['pos'],
            'freezeTimer': bot_state['freezeTimer'], 'mental_attack_timer': bot_state.get('mental_attack_timer', 0),
            'mode': bot_state['mode'], 'goals_round': bot_state.get('goalsReachedThisRound', 0),
            'is_player': bot_state.get('is_player_controlled', False), 'target_coord': bot_state.get('target_coordinate'),
            'is_hallucinating': bot_state.get('is_hallucinating_state', False) 
        }
        if bot_state['type'] == 'Learning':
            output_level_name = current_config.get("POLICY_HEAD_INPUT_LEVEL_NAME", "L2_Executive")
            anomaly_val = bot_state.get('last_anomaly_proxy', {}).get(output_level_name, 0.0)
            s_bot['anomaly'] = round(anomaly_val, 5)
        serializable_bots[bot_id] = s_bot
    return {'environment': environment.get_state() if environment else None, 'bots': serializable_bots, 'round': round_number, 'stats': stats, 'config': current_config}

def emit_state():
    try: state = get_game_state(); state['isRunning'] = simulation_running; socketio.emit('update_state', state)
    except Exception as e: print(f"Error emitting state: {e}"); traceback.print_exc()

@socketio.on('join_game')
def handle_join_game(data=None):
    sid = request.sid
    if sid in players: emit('join_ack', {'success': False, 'message': 'Already controlling.'}, room=sid); return
    if not environment or not bots: emit('join_ack', {'success': False, 'message': 'Sim not ready.'}, room=sid); return
    target_bot_id = data.get('target_bot_id') if data else None; available_bot_id = None
    if target_bot_id:
         if target_bot_id in bots and bots[target_bot_id]['type'] == 'Learning' and not bots[target_bot_id].get('is_player_controlled', False): available_bot_id = target_bot_id
         else: emit('join_ack', {'success': False, 'message': f"Bot {target_bot_id} unavailable."}, room=sid); return
    else: available_bot_id = next((bid for bid, b in bots.items() if b['type'] == 'Learning' and not b.get('is_player_controlled', False)), None)
    if available_bot_id:
        original_id = bots[available_bot_id]['original_bot_id']; print(f"Player {sid} taking control of {available_bot_id}")
        bots[available_bot_id]['is_player_controlled'] = True; bots[available_bot_id]['target_coordinate'] = None; bots[available_bot_id]['mode'] = "Player Control"
        players[sid] = {'player_bot_id': available_bot_id, 'original_bot_id': original_id}; emit('join_ack', {'success': True, 'player_id': available_bot_id, 'original_bot_id': original_id}, room=sid); socketio.emit('player_joined', {'player_id': available_bot_id}); emit_state()
    else: emit('join_ack', {'success': False, 'message': "No available Learning Bots."}, room=sid); print(f"Player {sid} failed join: No bots.")

@socketio.on('rejoin_game')
def handle_rejoin_game(data):
    sid = request.sid
    if sid in players: emit('rejoin_ack', {'success': False, 'message': 'Already controlling.'}, room=sid); return
    original_bot_id = data.get('originalBotId')
    if not original_bot_id: emit('rejoin_ack', {'success': False, 'message': 'No original bot ID.'}, room=sid); return
    print(f"Player {sid} attempting rejoin for {original_bot_id}")
    if original_bot_id in bots and bots[original_bot_id]['type'] == 'Learning':
        already_controlled = any(other_sid != sid and p_data['original_bot_id'] == original_bot_id for other_sid, p_data in players.items())
        if not already_controlled:
            bots[original_bot_id]['is_player_controlled'] = True; bots[original_bot_id]['target_coordinate'] = None; bots[original_bot_id]['mode'] = "Player Control (Rejoin)"; players[sid] = {'player_bot_id': original_bot_id, 'original_bot_id': original_bot_id}
            emit('rejoin_ack', {'success': True, 'player_id': original_bot_id, 'original_bot_id': original_bot_id}, room=sid); socketio.emit('player_joined', {'player_id': original_bot_id}); emit_state(); print(f"Player {sid} rejoined {original_bot_id}")
        else: emit('rejoin_ack', {'success': False, 'message': f'Bot {original_bot_id} already controlled.'}, room=sid)
    else: emit('rejoin_ack', {'success': False, 'message': f'Bot {original_bot_id} not available/learning.'}, room=sid)

@socketio.on('leave_game')
def handle_leave_game(data=None):
    sid = request.sid
    if sid in players:
        player_data = players.pop(sid); original_bot_id = player_data['original_bot_id']; print(f"Player {sid} leaving {original_bot_id}")
        if original_bot_id in bots: bots[original_bot_id]['is_player_controlled'] = False; bots[original_bot_id]['target_coordinate'] = None; bots[original_bot_id]['mode'] = "AI Control"; socketio.emit('player_left', {'player_id': original_bot_id}); emit_state()
        emit('leave_ack', {'success': True}, room=sid);
        if sid in player_direct_actions: del player_direct_actions[sid]
    else: emit('leave_ack', {'success': False, 'message': 'Not controlling.'}, room=sid)

@socketio.on('player_action')
def handle_player_action(data):
    sid = request.sid
    if sid in players:
        player_bot_id = players[sid]['player_bot_id']
        if player_bot_id in bots and bots[player_bot_id].get('is_player_controlled', False):
            action = data.get('action')
            try:
                action_int = int(action)
                if 0 <= action_int < current_config['NUM_ACTIONS']: player_direct_actions[sid] = action_int
                else: print(f"Warning: Invalid action {action} from {sid} vs NUM_ACTIONS={current_config['NUM_ACTIONS']}")
            except (ValueError, TypeError): print(f"Warning: Non-int action '{action}' from {sid}")

@socketio.on('update_player_target')
def handle_update_player_target(data):
    sid = request.sid
    if sid in players:
        player_bot_id = players[sid]['player_bot_id']
        if player_bot_id in bots and bots[player_bot_id].get('is_player_controlled', False):
            target = data.get('target'); grid_size = environment.size if environment else current_config['GRID_SIZE']
            if target is None: bots[player_bot_id]['target_coordinate'] = None
            elif isinstance(target, dict) and 'x' in target and 'y' in target:
                try: tx, ty = int(target['x']), int(target['y']); bots[player_bot_id]['target_coordinate'] = {'x': tx, 'y': ty} if 0 <= tx < grid_size and 0 <= ty < grid_size else None
                except (ValueError, TypeError): bots[player_bot_id]['target_coordinate'] = None
            else: bots[player_bot_id]['target_coordinate'] = None

@socketio.on('start_simulation')
def handle_start_simulation(data=None):
    global simulation_running, simulation_loop_task
    if simulation_running: emit('status_update', {'message': 'Sim already running.'}, room=request.sid); return
    print("Start simulation request.");
    if environment is None or not bots: 
         if not setup_simulation(full_reset=False, new_environment=True): 
              socketio.emit('simulation_stopped', {'message': 'Initial setup for start failed.'}); return
    
    if environment is None or not bots:
        socketio.emit('simulation_stopped', {'message': 'Environment or bots still missing after setup attempt.'})
        return

    simulation_running = True
    if simulation_loop_task is None or (hasattr(simulation_loop_task, 'dead') and simulation_loop_task.dead):
         simulation_loop_task = socketio.start_background_task(simulation_loop)
    socketio.emit('simulation_started'); emit_state()

@socketio.on('stop_simulation')
def handle_stop_simulation(data=None):
    global simulation_running
    if not simulation_running: emit('status_update', {'message': 'Sim already stopped.'}, room=request.sid); return
    print("Stop simulation request."); simulation_running = False; socketio.emit('simulation_stopped', {'message': 'Simulation stopped.'})

@socketio.on('reset_round')
def handle_reset_round(data=None):
    global simulation_running; was_running = simulation_running; print("Reset round request.")
    if simulation_running: handle_stop_simulation(); eventlet.sleep(0.1) 
    new_env = current_config.get("RANDOMIZE_ENV_PER_ROUND", False)
    if setup_simulation(full_reset=False, new_environment=new_env): 
        emit_state(); status_msg = 'New Round Ready.' + (' Press Start.' if not was_running else ''); socketio.emit('status_update', {'message': status_msg}); socketio.emit('simulation_stopped', {'message': 'New Round Ready.'})
    else: socketio.emit('status_update', {'message': 'Error resetting round.'})

@socketio.on('reset_full')
def handle_reset_full(data=None):
    global simulation_running; was_running = simulation_running; print("Full reset request.")
    if simulation_running: handle_stop_simulation(); eventlet.sleep(0.1)
    if setup_simulation(full_reset=True, new_environment=True): 
        emit_state(); status_msg = 'Full Reset Complete.' + (' Press Start.' if not was_running else ''); socketio.emit('status_update', {'message': status_msg}); socketio.emit('simulation_stopped', {'message': 'Full Reset Complete.'})
    else: socketio.emit('status_update', {'message': 'Error during full reset.'})

@socketio.on('update_config')
def handle_update_config(data):
    global current_config, av_manager, hierarchical_neural_system, signal_processor
    if simulation_running: emit('config_update_ack', {'success': False, 'message': 'Stop simulation first.'}, room=request.sid); return

    try:
        new_config_data = data.get('config', {})
        needs_full_reset = False; needs_round_reset = False; changed_keys = []
        temp_config = copy.deepcopy(current_config)
        # print("Received config update request:", {k:v for k,v in new_config_data.items() if k != 'HIERARCHY_LEVEL_CONFIGS'}) # Avoid printing large hierarchy

        reset_all_keys = ['GRID_SIZE', 'NUM_HC_BOTS', 'NUM_LEARNING_BOTS', 'NUM_GOALS',
                          'ENABLE_AV', 'ENABLE_AV_OUTPUT', 'VISUALIZATION_MODE',
                          'SENSORY_INPUT_DIM', 'POLICY_HEAD_INPUT_LEVEL_NAME', 'AV_DATA_SOURCE_LEVEL_NAME',
                          'NUM_ACTIONS', 'USE_RULES_AS_HNS_INPUT'] 
        reset_round_keys = ['MAX_STEPS_PER_ROUND', 'VISIBILITY_RANGE', 'OBSTACLES_FACTOR_MIN',
                            'OBSTACLES_FACTOR_MAX', 'MIN_GOAL_START_DISTANCE_FACTOR',
                            'MIN_BOT_START_DISTANCE_FACTOR', 'MIN_BOT_GOAL_DISTANCE_FACTOR',
                            'MENTAL_ATTACK_RANGE', 'MENTAL_ATTACK_DURATION']
        special_keys = ['HIERARCHY_LEVEL_CONFIGS', 'EEG_SETTINGS']

        if 'EEG_SETTINGS' in new_config_data:
            if str(new_config_data['EEG_SETTINGS']) != str(current_config.get('EEG_SETTINGS', {})):
                 print("EEG_SETTINGS changed.")
                 temp_config['EEG_SETTINGS'] = copy.deepcopy(new_config_data['EEG_SETTINGS'])
                 changed_keys.append('EEG_SETTINGS')
                 needs_full_reset = True 

        if 'HIERARCHY_LEVEL_CONFIGS' in new_config_data:
            if str(new_config_data['HIERARCHY_LEVEL_CONFIGS']) != str(current_config.get('HIERARCHY_LEVEL_CONFIGS', [])):
                print("HIERARCHY_LEVEL_CONFIGS changed.")
                temp_config['HIERARCHY_LEVEL_CONFIGS'] = copy.deepcopy(new_config_data['HIERARCHY_LEVEL_CONFIGS'])
                changed_keys.append('HIERARCHY_LEVEL_CONFIGS'); needs_full_reset = True

        for key, value in new_config_data.items():
            if key in special_keys: continue 
            if key in DEFAULT_CONFIG: 
                try:
                    default_type = type(DEFAULT_CONFIG[key]); current_value = temp_config.get(key)
                    if value is None: continue 
                    
                    if default_type is bool: converted_value = str(value).lower() in ['true', '1', 'yes', 'on']
                    elif default_type is int: converted_value = int(round(float(value))) 
                    elif default_type is float: converted_value = float(value)
                    else: converted_value = default_type(value) 

                    if key == "VISUALIZATION_MODE" and converted_value not in ['vispy', 'matplotlib', 'web', 'none']: converted_value = 'none'
                    if key == "GRID_SIZE": converted_value = max(10, min(200, converted_value))
                    if key == "SENSORY_INPUT_DIM": converted_value = max(10, min(256, converted_value))
                    if key == "NUM_ACTIONS": converted_value = max(4, min(10, converted_value)) 
                    if key == "MENTAL_ATTACK_RANGE": converted_value = max(1, min(temp_config.get("VISIBILITY_RANGE", 8), converted_value)) 
                    if key == "MENTAL_ATTACK_DURATION": converted_value = max(1, min(500, converted_value))
                    if key == "SIMULATION_SPEED_MS": converted_value = max(1, min(2000, converted_value))
                    if "FACTOR" in key.upper() and isinstance(converted_value, float): converted_value = max(0.0, min(1.0, converted_value))
                    if "RATE" in key.upper() and isinstance(converted_value, float): converted_value = max(0.0, min(100.0, converted_value))
                    if key == "PLAYER_CONTROL_PERCENT" and isinstance(converted_value, (float, int)): converted_value = max(0.0, min(100.0, float(converted_value)))


                    is_different = abs(converted_value - current_value) > 1e-9 if isinstance(converted_value, float) and isinstance(current_value, float) else str(current_value) != str(converted_value)
                    if is_different:
                        print(f"Applying config change: {key}: {current_value} -> {converted_value}")
                        temp_config[key] = converted_value; changed_keys.append(key)
                        if key in reset_all_keys: needs_full_reset = True
                        elif key in reset_round_keys: needs_round_reset = True
                except (ValueError, TypeError) as e: print(f"Warning: Invalid type/value for '{key}': '{value}'. Skipping. Err: {e}"); continue

        if changed_keys:
             current_config = temp_config; print(f"Config updated. Changed: {changed_keys}")
             if needs_full_reset: needs_round_reset = True 

             if 'EEG_SETTINGS' in changed_keys and current_config.get("EEG_SETTINGS",{}).get("enabled"):
                 eeg_config_changed_dim = update_eeg_processor_instance() 
                 if eeg_config_changed_dim: 
                     needs_full_reset = True
             elif 'EEG_SETTINGS' in changed_keys and not current_config.get("EEG_SETTINGS",{}).get("enabled"):
                 update_eeg_processor_instance() 

             hns_relevant_keys = ['HIERARCHY_LEVEL_CONFIGS', 'SENSORY_INPUT_DIM', 'NUM_ACTIONS', 'USE_RULES_AS_HNS_INPUT'] 
             if any(k in hns_relevant_keys for k in changed_keys) or needs_full_reset:
                 hns_recreated = update_hierarchical_neural_system_instance()
                 if not hierarchical_neural_system:
                      emit('config_update_ack', {'success': False, 'message': 'Error: Failed HNS update.'}, room=request.sid); return
                 if hns_recreated: needs_full_reset = True 

             av_params = ['ENABLE_AV', 'ENABLE_AV_OUTPUT', 'VISUALIZATION_MODE', 
                          'NUM_LEARNING_BOTS', 
                          'AV_DATA_SOURCE_LEVEL_NAME', 'HIERARCHY_LEVEL_CONFIGS'] 
             if any(k in av_params for k in changed_keys) or needs_full_reset :
                  av_state_changed = update_av_manager_instance()
                  if av_state_changed : needs_full_reset = True 

             emit('config_update_ack', {'success': True, 'needs_full_reset': needs_full_reset, 'needs_round_reset': needs_round_reset, 'updated_config': current_config}, room=request.sid)
             socketio.emit('config_update', current_config) 
        else:
             print("No effective config changes detected.")
             emit('config_update_ack', {'success': True, 'needs_full_reset': False, 'needs_round_reset': False, 'updated_config': current_config}, room=request.sid)
    except Exception as e: print(f"Error updating config: {e}"); traceback.print_exc(); emit('config_update_ack', {'success': False, 'message': f'Server error: {e}'}, room=request.sid)


def start_simulation_if_configured():
    if current_config.get('AUTOSTART_SIMULATION', False):
        print("\n=== AUTO-STARTING SIMULATION ===")
        global simulation_running, simulation_loop_task
        if not simulation_running:
            if environment is None or not bots: 
                if not setup_simulation(full_reset=True, new_environment=True):
                     print("CRITICAL: Auto-start setup failed. Sim not started.")
                     return
            
            if environment is None or not bots:
                 print("CRITICAL: Environment or bots still missing after auto-start setup attempt. Sim not started.")
                 return

            simulation_running = True
            simulation_loop_task = socketio.start_background_task(simulation_loop)
            print("Simulation auto-started successfully")

# ================================================================
# --- Server Start ---
# ================================================================
if __name__ == '__main__':
    if sys.platform == "darwin" or sys.platform == "win32":
        print(f"Platform is {sys.platform}, setting multiprocessing start method to 'spawn'.")
        try: mp.set_start_method('spawn', force=True)
        except RuntimeError as e: print(f"Warning: Could not force 'spawn': {e}")

    print("Initializing simulation state...")
    if not HIERARCHICAL_NEURAL_LIB_AVAILABLE: print("CRITICAL: Hierarchical NN Lib failed. Exiting."); sys.exit(1)
    
    if not setup_simulation(full_reset=True, new_environment=True):
        print("CRITICAL: Initial simulation setup failed. Exiting.")
        if av_manager: stop_av_system(av_manager)
        sys.exit(1)
    else: print("Initial setup successful.")

    start_simulation_if_configured()

    port = int(os.environ.get('PORT', 5001)); host = '0.0.0.0'
    print(f"Attempting to start server on http://{host}:{port}")
    try:
        print("Starting Flask-SocketIO server with eventlet...")
        import eventlet.wsgi
        eventlet.wsgi.server(eventlet.listen((host, port)), app)
    except OSError as e:
         if "Address already in use" in str(e): print(f"Error: Port {port} is already in use.")
         else: print(f"Error: Failed server start OS error: {e}"); traceback.print_exc()
    except Exception as e: print(f"Error: Unexpected server startup error: {e}"); traceback.print_exc()
    finally:
        print("Server shutting down...");
        if simulation_running: simulation_running = False
        if simulation_loop_task and hasattr(simulation_loop_task, 'kill'):
            try: simulation_loop_task.kill()
            except Exception as ke: print(f"Error killing sim loop task: {ke}")
        if av_manager: stop_av_system(av_manager)
        print("Server stopped.")

