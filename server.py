# Filename: server.py
# coding: utf-8
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

# --- Eventlet Monkey Patching (IMPORTANT: Must be done early) ---
# Consider conditional patching if VisPy/Matplotlib have issues with it
# For now, keep it enabled as SocketIO relies on it.
import eventlet
# Check if we need to avoid patching 'thread' if using certain AV modes?
# Needs testing, but patching is usually required for SocketIO async modes.
eventlet.monkey_patch()
print("Eventlet monkey patching applied.")

# --- Flask & SocketIO Setup ---
from flask import Flask, render_template, request, jsonify, Response # Added Response
from flask_socketio import SocketIO, emit, Namespace # Added Namespace

app = Flask(__name__, template_folder='.') # Look for index.html in the same directory
app.config['SECRET_KEY'] = os.urandom(24)
# Use eventlet for async mode
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

print("Flask and SocketIO initialized.")

# --- PyTorch Setup ---
import torch
from torch import nn, Tensor, is_tensor, cat, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, Parameter, ParameterList

# --- Import Custom Libraries ---
try:
    from neural_memory_lib import NeuralMemoryManager, NeuralMemState, mem_state_detach
    print("Neural Memory Library imported successfully.")
    NEURAL_LIB_AVAILABLE = True
except ImportError as e:
    print(f"FATAL ERROR: Failed to import neural_memory_lib.py: {e}")
    traceback.print_exc()
    NEURAL_LIB_AVAILABLE = False
    NeuralMemoryManager = None # Define as None if import fails
    NeuralMemState = namedtuple('DummyNeuralMemState', ['seq_index', 'weights', 'optim_state'])
    mem_state_detach = lambda state: state

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
    # Import AV functions/class from the new file
    from audiovisualization import setup_av_system, update_av_system, stop_av_system, AVManager
    print("Audiovisualization Library imported successfully.")
    AV_LIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Failed to import audiovisualization.py: {e}. AV features disabled.")
    AV_LIB_AVAILABLE = False
    # Define dummy functions so calls don't break server logic
    setup_av_system = lambda *args, **kwargs: None
    update_av_system = lambda *args, **kwargs: None
    stop_av_system = lambda *args, **kwargs: None
    AVManager = None # Define AVManager as None

# --- Determine Device ---
try:
    if torch.cuda.is_available():
        # Check if multiprocessing context needs to be set for CUDA
        if sys.platform != 'win32': # 'fork' context (default on Linux/Mac) can have issues with CUDA
            if mp.get_start_method(allow_none=True) is None or mp.get_start_method(allow_none=True) == 'fork':
                print("Warning: Setting multiprocessing start method to 'spawn' for CUDA compatibility.")
                # mp.set_start_method('spawn', force=True) # Use spawn for CUDA safety
                # NOTE: 'spawn' is generally safer but might be slower.
                # If 'spawn' causes issues elsewhere, consider 'forkserver'.
                # Forcing can be problematic if libraries expect 'fork'. Test carefully.
                # Let's try without forcing first, rely on user setting CUDA_VISIBLE_DEVICES if needed.
                pass

        device = torch.device("cuda")
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        try:
            _ = torch.tensor([1.0], device=device)
            torch.cuda.empty_cache()
            print("CUDA device test successful and memory cache cleared.")
        except Exception as e:
            print(f"Warning: CUDA device error during test/clear cache: {e}. Falling back to CPU.")
            traceback.print_exc()
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
except Exception as e:
    print(f"Error during PyTorch device setup: {e}. Falling back to CPU.")
    device = torch.device("cpu")

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
    "NUM_ACTIONS": 6, # 0:Up, 1:Left, 2:Right, 3:Down, 4:Punch, 5:ClaimGoal
    "RANDOMIZE_ENV_PER_ROUND": True, # Generate new obstacles/starts each round?

    # Titans-Inspired Learning Bot Params (Library)
    "LEARNING_BOT_DIM": 128,
    "LEARNING_BOT_MEM_DEPTH": 2,
    "LEARNING_BOT_LR": 0.001,
    "LEARNING_BOT_WEIGHT_DECAY": 0.01,
    "LEARNING_BOT_MOMENTUM": 0.9,
    "LEARNING_BOT_MAX_GRAD_NORM": 1.0,

    # Learning Bot Behavior (Outside Library)
    "LEARNING_BOT_BASE_EXPLORATION_RATE": 15.0, # Percentage 0-100
    "LEARNING_BOT_RULE_EXPLORE_PERCENT": 60.0, # Percentage 0-100
    "PLAYER_CONTROL_PERCENT": 100.0, # Player control influence (0-100)

    # Env Generation Params
    "MIN_GOAL_START_DISTANCE_FACTOR": 0.15,
    "MIN_BOT_START_DISTANCE_FACTOR": 0.25,
    "MIN_BOT_GOAL_DISTANCE_FACTOR": 0.15,

    # Audiovisualization Options
    "ENABLE_AV": True, # Default to True, but requires libs
    "ENABLE_AV_OUTPUT": True, # Default to True, controls VisPy/SoundDevice/etc.
    "VISUALIZATION_MODE": "web", # Options: 'vispy', 'matplotlib', 'web', 'none'
    
    "AUTOSTART_SIMULATION": True,  # Set to False to disable auto-start
}
current_config = copy.deepcopy(DEFAULT_CONFIG)

# --- Global State ---
bots = {} # bot_id -> bot_state_dict
players = {} # sid -> {'player_bot_id': player_bot_id, 'original_bot_id': original_learning_bot_id}
environment = None
neural_memory_manager = None # Instance of NeuralMemoryManager
av_manager = None # Instance of AVManager (or None if disabled/failed)
simulation_running = False
simulation_loop_task = None
round_number = 0
stats = {'hc_total_goals': 0, 'learning_total_goals': 0}
player_direct_actions = {} # sid -> action_code (for mobile buttons)

# --- Function to Update NN Manager ---
def update_neural_memory_manager_instance():
    """ Creates or updates the NeuralMemory manager based on current_config """
    global neural_memory_manager
    if not NEURAL_LIB_AVAILABLE:
        print("Error: Cannot update NN Manager, library not available.")
        neural_memory_manager = None
        return False

    print("Updating Neural Memory Manager...")
    try:
        # Check if manager exists and if relevant params changed
        needs_new_manager = True
        if neural_memory_manager:
             current_dim = getattr(neural_memory_manager, 'dim', -1)
             current_lr = neural_memory_manager.optimizer_config.get('lr', -1)
             current_wd = neural_memory_manager.optimizer_config.get('weight_decay', -1)
             current_beta1 = neural_memory_manager.optimizer_config.get('betas', (-1,))[0]
             current_grad_norm = getattr(neural_memory_manager, 'max_grad_norm', -1)
             current_depth = 0
             if hasattr(neural_memory_manager, 'memory_model_template') and hasattr(neural_memory_manager.memory_model_template, 'net'):
                 current_depth = sum(1 for m in neural_memory_manager.memory_model_template.net if isinstance(m, nn.Linear))

             if (current_dim == current_config['LEARNING_BOT_DIM'] and
                 abs(current_lr - current_config['LEARNING_BOT_LR']) < 1e-9 and
                 abs(current_wd - current_config['LEARNING_BOT_WEIGHT_DECAY']) < 1e-9 and
                 abs(current_beta1 - current_config['LEARNING_BOT_MOMENTUM']) < 1e-9 and
                 abs(current_grad_norm - current_config['LEARNING_BOT_MAX_GRAD_NORM']) < 1e-9 and
                 current_depth == current_config['LEARNING_BOT_MEM_DEPTH'] and
                 neural_memory_manager.target_device == device):
                  needs_new_manager = False
             else:
                  print("NNM parameters changed, recreating...")
                  if neural_memory_manager.target_device != device:
                       print(f"  Device changed from {neural_memory_manager.target_device} to {device}")

        if needs_new_manager:
            print("Creating new NN Manager instance...")
            if neural_memory_manager: del neural_memory_manager; torch.cuda.empty_cache() if device.type == 'cuda' else None

            neural_memory_manager = NeuralMemoryManager(
                dim=current_config['LEARNING_BOT_DIM'],
                mem_model_depth=current_config['LEARNING_BOT_MEM_DEPTH'],
                learning_rate=current_config['LEARNING_BOT_LR'],
                weight_decay=current_config['LEARNING_BOT_WEIGHT_DECAY'],
                momentum_beta=current_config['LEARNING_BOT_MOMENTUM'],
                max_grad_norm=current_config['LEARNING_BOT_MAX_GRAD_NORM'],
                target_device=device
            )
            print(f"New Neural Memory Manager ready on device: {device}")
            return True # Indicates manager was recreated
        else:
            print("Existing NN Manager is compatible, reusing.")
            return False # Indicates manager was reused

    except Exception as e:
         print(f"FATAL: Failed to create/update Neural Memory Manager: {e}")
         traceback.print_exc()
         neural_memory_manager = None
         return False

# --- Function to Update AV Manager ---
def update_av_manager_instance():
    """ Creates or stops/recreates the AV manager based on config """
    global av_manager
    enable_av = current_config.get("ENABLE_AV", False)
    enable_av_output = current_config.get("ENABLE_AV_OUTPUT", False)
    visualization_mode = current_config.get("VISUALIZATION_MODE", "none")

    if not AV_LIB_AVAILABLE and enable_av:
        print("Warning: AV Library not available, cannot enable AV Manager.")
        enable_av = False # Force disable if lib missing

    state_changed = False
    if enable_av:
        num_av_bots = current_config.get('NUM_LEARNING_BOTS', 0)
        av_dim = current_config.get('LEARNING_BOT_DIM', 0)

        needs_new_av_manager = True
        if av_manager and isinstance(av_manager, AVManager):
            if (av_manager.num_bots == num_av_bots and
                av_manager.dim == av_dim and
                av_manager.enable_audio_output == (enable_av_output and SOUNDDEVICE_AVAILABLE) and # Check actual audio capability
                av_manager.visualization_mode == (visualization_mode if enable_av_output else 'none') and # Check actual visual mode
                str(av_manager.device) == str(device)):
                needs_new_av_manager = False
                print(f"Existing AV Manager compatible (Mode: {av_manager.visualization_mode}, Audio: {av_manager.enable_audio_output}).")
            else:
                print("AV parameters, device, or mode changed, recreating AV Manager...")
                stop_av_system(av_manager)
                av_manager = None

        if needs_new_av_manager:
            print(f"Creating new AV Manager instance (Mode: {visualization_mode})...")
            if av_manager: stop_av_system(av_manager) # Ensure stop again

            # Pass socketio instance if web mode is selected
            sio_instance = socketio if visualization_mode == 'web' else None

            av_manager = setup_av_system(
                num_bots=num_av_bots,
                dim=av_dim,
                device=device,
                enable_output=enable_av_output,
                visualization_mode=visualization_mode,
                socketio_instance=sio_instance # Pass socketio instance
            )
            if av_manager:
                 print("New AV Manager created successfully.")
                 state_changed = True
            else:
                 print("Warning: Failed to create new AV Manager instance.")
                 # Keep av_manager as None
    else:
        # AV should be disabled
        if av_manager:
            print("Disabling and stopping AV Manager...")
            stop_av_system(av_manager)
            av_manager = None
            state_changed = True

    return state_changed

# ================================================================
# --- Simulation Environment (No changes needed from previous version) ---
# ================================================================
class GridEnvironment:
    def __init__(self, size, num_goals, obstacles_factor_range, num_hc_bots, num_learning_bots, config_factors):
        self.size = max(10, int(size))
        self.num_goals = max(0, int(num_goals))
        self.min_obstacles_factor, self.max_obstacles_factor = obstacles_factor_range
        self.num_hc_bots = max(0, int(num_hc_bots))
        self.num_learning_bots = max(0, int(num_learning_bots))
        self.config_factors = {
            'goal_dist': config_factors.get('MIN_GOAL_START_DISTANCE_FACTOR', 0.15),
            'bot_dist': config_factors.get('MIN_BOT_START_DISTANCE_FACTOR', 0.25),
            'bot_goal_dist': config_factors.get('MIN_BOT_GOAL_DISTANCE_FACTOR', 0.15)
        }
        self.obstacles = set()
        self.goals = []
        self.claimed_goals = set()
        self.start_positions = []
        self._initial_goals = []

        try:
            self.randomize() # Initial randomization
        except Exception as e:
            print(f"FATAL ERROR during environment initialization: {e}")
            traceback.print_exc()
            # Provide minimal safe defaults
            self.size=10; self.num_goals=0; self.num_hc_bots=0; self.num_learning_bots=0
            self.goals=[]; self.obstacles=set(); self.start_positions=[]; self._initial_goals = []


    def _manhattan_distance(self, pos1, pos2):
        if not pos1 or not pos2 or 'x' not in pos1 or 'y' not in pos1 or 'x' not in pos2 or 'y' not in pos2: return float('inf')
        return abs(pos1['x'] - pos2['x']) + abs(pos1['y'] - pos2['y'])

    def randomize(self):
        """Generates a new layout for obstacles, goals, and bot start positions."""
        self.obstacles.clear()
        self.goals = []
        self.claimed_goals.clear()
        self.start_positions = []
        total_bots = self.num_hc_bots + self.num_learning_bots
        total_cells = self.size * self.size
        required_items = total_bots + self.num_goals
        print(f"Randomizing Env: Size={self.size}x{self.size}, Goals={self.num_goals}, HC={self.num_hc_bots}, Lrn={self.num_learning_bots}")

        if total_cells <= 0: raise ValueError("Grid size must be positive.")
        if required_items == 0: print("No goals or bots to place."); # Still place obstacles below

        occupied = set()
        max_placement_attempts = max(required_items * 100, total_cells * 20) # Increased attempts
        attempts = 0

        def is_valid_placement(pos_tuple, occupied_set, check_dists={}):
            if not (0 <= pos_tuple[0] < self.size and 0 <= pos_tuple[1] < self.size): return False
            if pos_tuple in occupied_set: return False
            pos_dict = {'x': pos_tuple[0], 'y': pos_tuple[1]}
            # Check distances (more efficiently)
            if 'goal_min_dist' in check_dists:
                min_d = check_dists['goal_min_dist']
                if any(self._manhattan_distance(pos_dict, g) < min_d for g in self.goals): return False
            if 'bot_min_dist' in check_dists:
                min_d = check_dists['bot_min_dist']
                if any(self._manhattan_distance(pos_dict, sp) < min_d for sp in self.start_positions): return False
            if 'bot_goal_min_dist' in check_dists:
                min_d = check_dists['bot_goal_min_dist']
                if any(self._manhattan_distance(pos_dict, g) < min_d for g in self.goals): return False
            return True

        # --- Place Goals ---
        min_goal_dist = max(2, int(self.size * self.config_factors['goal_dist']))
        goal_id_counter = 0
        attempts = 0
        while len(self.goals) < self.num_goals and attempts < max_placement_attempts:
            attempts += 1
            pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if is_valid_placement(pos, occupied, {'goal_min_dist': min_goal_dist}):
                goal = {'x': pos[0], 'y': pos[1], 'id': f'G{goal_id_counter}'}
                self.goals.append(goal); occupied.add(pos); goal_id_counter += 1
        if len(self.goals) < self.num_goals: print(f"Warning: Placed only {len(self.goals)}/{self.num_goals} goals after {attempts} attempts.")

        # --- Place Bots ---
        min_bot_dist = max(3, int(self.size * self.config_factors['bot_dist']))
        min_bot_goal_dist = max(3, int(self.size * self.config_factors['bot_goal_dist']))
        attempts = 0
        placed_bots = 0
        # Create bot positions first, then assign types/IDs
        temp_bot_positions = []
        while len(temp_bot_positions) < total_bots and attempts < max_placement_attempts:
            attempts += 1
            pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            # Need to check against already placed temp bots too
            temp_occupied = occupied.union(set(temp_bot_positions))
            if is_valid_placement(pos, temp_occupied, {'bot_min_dist': min_bot_dist, 'bot_goal_min_dist': min_bot_goal_dist}):
                temp_bot_positions.append(pos)

        # Assign IDs and types sequentially to the successfully placed positions
        hc_placed = 0; ln_placed = 0
        for idx, pos in enumerate(temp_bot_positions):
             bot_type = 'Hardcoded' if idx < self.num_hc_bots else 'Learning'
             if bot_type == 'Hardcoded': bot_num = hc_placed; hc_placed += 1
             else: bot_num = ln_placed; ln_placed += 1
             bot_id = f'{bot_type[0]}{bot_num}'
             self.start_positions.append({'x': pos[0], 'y': pos[1], 'type': bot_type, 'id': bot_id})
             occupied.add(pos) # Add to main occupied set

        if len(self.start_positions) < total_bots:
             print(f"CRITICAL Warning: Placed only {len(self.start_positions)}/{total_bots} bots. Adjusting counts.")
             self.num_hc_bots = hc_placed
             self.num_learning_bots = ln_placed


        # --- Place Obstacles ---
        num_obstacles_to_place = random.randint(
            int(total_cells * self.min_obstacles_factor),
            int(total_cells * self.max_obstacles_factor)
        ) if total_cells > 0 else 0
        attempts = 0
        placed_obstacles = 0
        while placed_obstacles < num_obstacles_to_place and attempts < max_placement_attempts:
             attempts += 1
             pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
             if is_valid_placement(pos, occupied): # Check against goals and bots
                 self.obstacles.add(pos); occupied.add(pos); placed_obstacles += 1

        self._initial_goals = [{'x': g['x'], 'y': g['y'], 'id': g['id']} for g in self.goals]
        print(f"Environment randomized. Placed {len(self.goals)} goals, {len(self.start_positions)} bots, {len(self.obstacles)} obstacles.")

    def reset_round_state(self):
        """Resets goal claims, keeping the same obstacle/start layout."""
        self.claimed_goals.clear()
        # Use the stored initial goals to reset
        self.goals = [{'x': g['x'], 'y': g['y'], 'id': g['id']} for g in self._initial_goals]
        print(f"Round state reset. {len(self.goals)} goals active.")

    def is_valid(self, pos):
        """Checks if a position is within bounds and not an obstacle."""
        if not pos or 'x' not in pos or 'y' not in pos: return False
        x, y = pos['x'], pos['y']
        return 0 <= x < self.size and 0 <= y < self.size and (x, y) not in self.obstacles

    def find_path(self, start_pos, goal_pos, all_bots_dict=None, moving_bot_id=None):
        """ Finds a path using BFS, avoiding obstacles and other bots (optional). """
        if not self.is_valid(start_pos) or goal_pos is None: return None # Goal might be invalid if claimed
        # Check if goal is valid before proceeding
        if not (0 <= goal_pos.get('x', -1) < self.size and 0 <= goal_pos.get('y', -1) < self.size):
             return None # Goal is outside bounds

        start_tuple = (start_pos['x'], start_pos['y'])
        goal_tuple = (goal_pos['x'], goal_pos['y'])
        if start_tuple == goal_tuple: return []

        current_obstacles = self.obstacles.copy()
        if all_bots_dict:
            for bot_id, bot_state in all_bots_dict.items():
                if bot_id == moving_bot_id: continue # Don't block self
                current_obstacles.add((bot_state['pos']['x'], bot_state['pos']['y']))

        queue = deque([(start_tuple, [])])
        visited = {start_tuple}
        # Up, Left, Right, Down actions
        deltas = [(0, -1, 0), (-1, 0, 1), (1, 0, 2), (0, 1, 3)]

        max_path_len = self.size * self.size # Limit search depth reasonably

        while queue:
            current_pos_tuple, path = queue.popleft()

            if len(path) >= max_path_len: continue # Limit search depth

            for dx, dy, action in deltas: # Corrected order dx, dy, action
                next_x, next_y = current_pos_tuple[0] + dx, current_pos_tuple[1] + dy
                next_pos_tuple = (next_x, next_y)

                if next_pos_tuple == goal_tuple: # Found goal
                    return path + [action]

                if (0 <= next_x < self.size and 0 <= next_y < self.size and
                    next_pos_tuple not in current_obstacles and next_pos_tuple not in visited):
                    visited.add(next_pos_tuple)
                    new_path = path + [action]
                    queue.append((next_pos_tuple, new_path))
        return None # No path found


    def get_sensory_data(self, acting_bot, all_bots_dict, visibility_range):
        """ Gathers sensory information for a bot. """
        bot_pos = acting_bot['pos']
        vis_range = max(1, int(visibility_range))
        senses = {
            'wall_distance_N': bot_pos['y'],
            'wall_distance_S': self.size - 1 - bot_pos['y'],
            'wall_distance_W': bot_pos['x'],
            'wall_distance_E': self.size - 1 - bot_pos['x'],
            'nearest_goal_dist': vis_range + 1,
            'nearest_goal_dx': 0,
            'nearest_goal_dy': 0,
            'num_visible_goals': 0,
            'nearest_opponent_dist': vis_range + 1,
            'nearest_opponent_dx': 0,
            'nearest_opponent_dy': 0,
            'opponent_is_frozen': 0.0, # Float 0.0 or 1.0
            'opponent_type_HC': 0.0,
            'opponent_type_LN': 0.0,
            'opponent_type_PL': 0.0,
            'self_is_frozen': 1.0 if acting_bot['freezeTimer'] > 0 else 0.0,
            '_visibleGoals': [], # Internal list for HC bot logic
            '_nearestOpponent': None # Internal dict for HC bot logic
        }

        min_goal_dist = vis_range + 1
        nearest_goal_obj = None
        for goal in self.goals:
             if goal['id'] not in self.claimed_goals:
                 dist = self._manhattan_distance(bot_pos, goal)
                 if dist <= vis_range:
                     senses['num_visible_goals'] += 1
                     senses['_visibleGoals'].append({'x': goal['x'], 'y': goal['y'], 'id': goal['id'], 'dist': dist})
                     if dist < min_goal_dist:
                         min_goal_dist = dist
                         nearest_goal_obj = goal

        senses['nearest_goal_dist'] = min_goal_dist
        if nearest_goal_obj:
             senses['nearest_goal_dx'] = nearest_goal_obj['x'] - bot_pos['x']
             senses['nearest_goal_dy'] = nearest_goal_obj['y'] - bot_pos['y']

        min_opp_dist = vis_range + 1
        nearest_opponent_obj = None
        for opp_id, opponent_bot in all_bots_dict.items():
             if opp_id == acting_bot['id']: continue
             dist = self._manhattan_distance(bot_pos, opponent_bot['pos'])
             if dist <= vis_range:
                 if dist < min_opp_dist:
                     min_opp_dist = dist
                     nearest_opponent_obj = opponent_bot

        senses['nearest_opponent_dist'] = min_opp_dist
        if nearest_opponent_obj:
            senses['_nearestOpponent'] = nearest_opponent_obj # Store full obj for HC logic
            senses['nearest_opponent_dx'] = nearest_opponent_obj['pos']['x'] - bot_pos['x']
            senses['nearest_opponent_dy'] = nearest_opponent_obj['pos']['y'] - bot_pos['y']
            senses['opponent_is_frozen'] = 1.0 if nearest_opponent_obj['freezeTimer'] > 0 else 0.0
            is_player = nearest_opponent_obj.get('is_player_controlled', False)
            if is_player: senses['opponent_type_PL'] = 1.0
            elif nearest_opponent_obj['type'] == 'Hardcoded': senses['opponent_type_HC'] = 1.0
            elif nearest_opponent_obj['type'] == 'Learning': senses['opponent_type_LN'] = 1.0


        # Sort visible goals by distance for HC logic
        senses['_visibleGoals'].sort(key=lambda g: g['dist'])
        return senses


    def perform_move_action(self, bot_pos, action_index):
        """ Calculates the resulting position from a move action. """
        next_pos = bot_pos.copy()
        # 0:Up, 1:Left, 2:Right, 3:Down
        delta = [(0, -1), (-1, 0), (1, 0), (0, 1)] # Corrected deltas (dx, dy)
        if 0 <= action_index <= 3:
            dx, dy = delta[action_index]
            next_pos['x'] += dx
            next_pos['y'] += dy
        return next_pos

    def get_adjacent_unclaimed_goal(self, bot_pos):
        """ Finds an adjacent, unclaimed goal. """
        for goal in self.goals:
            if goal['id'] not in self.claimed_goals:
                if self._manhattan_distance(bot_pos, goal) == 1:
                    return goal
        return None

    def claim_goal(self, goal_id, bot_id):
        """ Marks a goal as claimed if it exists and is not claimed. """
        if goal_id in self.claimed_goals: return False
        goal_exists = any(g['id'] == goal_id for g in self.goals)
        if goal_exists:
            self.claimed_goals.add(goal_id)
            return True
        return False

    def are_all_goals_claimed(self):
         """ Checks if all initially placed goals are claimed. """
         return len(self._initial_goals) > 0 and len(self.claimed_goals) >= len(self._initial_goals)

    def get_state(self):
        """ Gets the current serializable state of the environment. """
        active_goals = [g for g in self.goals if g['id'] not in self.claimed_goals]
        return {
            'size': self.size,
            'goals': active_goals,
            'obstacles': list(self.obstacles), # Convert set to list for JSON
            'claimedGoals': list(self.claimed_goals) # Convert set to list
        }

# ================================================================
# --- Bot Logic (No changes needed from previous version) ---
# ================================================================

# --- Hardcoded Bot Logic ---
def get_hardcoded_action(bot_state, senses, env, all_bots_dict):
    bot_id, pos = bot_state['id'], bot_state['pos']
    bot_state.setdefault('stuckCounter', 0)
    bot_state.setdefault('currentPath', None)
    bot_state.setdefault('lastAction', -1)
    bot_state.setdefault('targetGoalId', None)
    bot_state.setdefault('lastPos', {'x': -1, 'y': -1})
    bot_state.setdefault('randomMoveCounter', 0)

    if pos == bot_state['lastPos']: bot_state['stuckCounter'] += 1
    else: bot_state['stuckCounter'] = 0; bot_state['lastPos'] = pos.copy(); bot_state['randomMoveCounter'] = 0

    if bot_state['freezeTimer'] > 0:
        bot_state['stuckCounter'] = 0; bot_state['currentPath'] = None; bot_state['targetGoalId'] = None; bot_state['randomMoveCounter'] = 0
        return -1, "Frozen"

    adjacent_goal = env.get_adjacent_unclaimed_goal(pos)
    if adjacent_goal:
        bot_state['stuckCounter'] = 0; bot_state['currentPath'] = None; bot_state['targetGoalId'] = None; bot_state['randomMoveCounter'] = 0
        return 5, f"Claim {adjacent_goal['id']}"

    nearest_opponent = senses.get('_nearestOpponent')
    if nearest_opponent and senses.get('nearest_opponent_dist', 99) == 1 and not senses.get('opponent_is_frozen'):
        bot_state['stuckCounter'] = 0; bot_state['currentPath'] = None; bot_state['targetGoalId'] = None; bot_state['randomMoveCounter'] = 0
        return 4, f"Punch {nearest_opponent['id']}"

    if bot_state['stuckCounter'] >= 5 and bot_state['randomMoveCounter'] < 3:
         bot_state['randomMoveCounter'] += 1
         bot_state['currentPath'] = None; bot_state['targetGoalId'] = None
         valid_moves = []
         for action_idx in range(4):
             next_p = env.perform_move_action(pos, action_idx)
             occupied_by_active = any(bid != bot_id and b['pos'] == next_p and b.get('freezeTimer', 0) <= 0 for bid, b in all_bots_dict.items())
             if env.is_valid(next_p) and not occupied_by_active: valid_moves.append(action_idx)
         if valid_moves: return random.choice(valid_moves), f"StuckRandom ({bot_state['stuckCounter']})"
         else: return -1, "StuckBlocked"
    elif bot_state['stuckCounter'] >= 5:
         return -1, f"Stuck ({bot_state['stuckCounter']})"

    current_path = bot_state.get('currentPath')
    if current_path:
        next_action = current_path[0]
        intended_pos = env.perform_move_action(pos, next_action)
        is_pos_valid = env.is_valid(intended_pos)
        is_pos_occupied_by_other = any(other_id != bot_id and other_bot['pos'] == intended_pos for other_id, other_bot in all_bots_dict.items())

        if is_pos_valid and not is_pos_occupied_by_other:
            bot_state['currentPath'].pop(0)
            mode_str = f"Path ({len(bot_state['currentPath'])} left)"
            if not bot_state['currentPath']: bot_state['targetGoalId'] = None; mode_str = "Path End"
            bot_state['randomMoveCounter'] = 0
            return next_action, mode_str
        else:
            bot_state['currentPath'] = None; bot_state['targetGoalId'] = None

    visible_goals = senses.get('_visibleGoals', [])
    target_goal_obj = None
    if bot_state['targetGoalId']:
        potential_target = next((g for g in visible_goals if g['id'] == bot_state['targetGoalId']), None)
        if potential_target: target_goal_obj = potential_target
        else: bot_state['targetGoalId'] = None

    if not target_goal_obj and visible_goals:
        target_goal_obj = visible_goals[0]; bot_state['targetGoalId'] = target_goal_obj['id']

    if target_goal_obj:
        path_to_goal = env.find_path(pos, target_goal_obj, all_bots_dict, bot_id)
        if path_to_goal:
            bot_state['currentPath'] = path_to_goal
            if bot_state['currentPath']:
                next_action = bot_state['currentPath'].pop(0)
                mode_str = f"NewPath ({len(bot_state['currentPath'])} left)"
                if not bot_state['currentPath']: bot_state['targetGoalId'] = None; mode_str="NewPath End"
                bot_state['randomMoveCounter'] = 0
                return next_action, mode_str
            else:
                 bot_state['targetGoalId'] = None; bot_state['currentPath'] = None

    valid_moves = []
    for action_idx in range(4):
        next_p = env.perform_move_action(pos, action_idx)
        occupied_by_active = any(bid != bot_id and b['pos'] == next_p and b.get('freezeTimer', 0) <= 0 for bid, b in all_bots_dict.items())
        if env.is_valid(next_p) and not occupied_by_active: valid_moves.append(action_idx)

    if not valid_moves:
        return -1, "Blocked"

    last_action = bot_state.get('lastAction', -1)
    reverse_action = -1
    if 0 <= last_action <= 3: reverse_map = {0: 3, 1: 2, 2: 1, 3: 0}; reverse_action = reverse_map.get(last_action)

    non_reverse_moves = [m for m in valid_moves if m != reverse_action]
    chosen_move = -1

    if non_reverse_moves: chosen_move = random.choice(non_reverse_moves)
    elif valid_moves: chosen_move = random.choice(valid_moves)

    bot_state['currentPath'] = None
    return chosen_move, f"Random ({bot_state['stuckCounter']})"

# --- Learning Bot Input Encoding ---
def _get_input_tensor_for_bot(bot_state, senses, config):
    """ Encodes bot's senses and last action into a tensor for the NeuralMemory """
    dim = config['LEARNING_BOT_DIM']
    vis_range = config['VISIBILITY_RANGE']
    num_actions = config['NUM_ACTIONS']

    if dim <= 0: raise ValueError("LEARNING_BOT_DIM must be positive.")

    features = []
    bl = np # Use numpy for feature construction

    def norm_capped(val, cap=vis_range):
        v = float(val) if val is not None else 0.0
        c = float(cap); return 0.0 if c <= 0 else math.copysign(min(abs(v), c), v) / c

    features.append(norm_capped(senses.get('wall_distance_N')))
    features.append(norm_capped(senses.get('wall_distance_S')))
    features.append(norm_capped(senses.get('wall_distance_W')))
    features.append(norm_capped(senses.get('wall_distance_E')))
    features.append(norm_capped(senses.get('nearest_goal_dist')))
    features.append(norm_capped(senses.get('nearest_goal_dx')))
    features.append(norm_capped(senses.get('nearest_goal_dy')))
    features.append(min(1.0, max(0.0, senses.get('num_visible_goals', 0) / 5.0)))
    features.append(norm_capped(senses.get('nearest_opponent_dist')))
    features.append(norm_capped(senses.get('nearest_opponent_dx')))
    features.append(norm_capped(senses.get('nearest_opponent_dy')))
    features.append(float(senses.get('opponent_is_frozen', 0.0)))
    features.append(float(senses.get('opponent_type_HC', 0.0)))
    features.append(float(senses.get('opponent_type_LN', 0.0)))
    features.append(float(senses.get('opponent_type_PL', 0.0)))
    features.append(float(senses.get('self_is_frozen', 0.0)))

    last_action = bot_state.get('lastAction', -1)
    action_enc = bl.zeros(num_actions)
    if 0 <= last_action < num_actions: action_enc[last_action] = 1.0
    features.extend(action_enc.tolist())

    current_len = len(features)
    if current_len < dim: features.extend([0.0] * (dim - current_len))
    elif current_len > dim: print(f"Warning: Feature vector length ({current_len}) > DIM ({dim}). Truncating."); features = features[:dim]

    try:
        np_features = np.array(features, dtype=np.float32)
        input_tensor = torch.from_numpy(np_features).to(device).unsqueeze(0).unsqueeze(0) # Shape [1, 1, dim]
        if input_tensor.shape != (1, 1, dim): raise ValueError(f"Shape mismatch: {input_tensor.shape}")
        return input_tensor
    except Exception as e:
        print(f"Error creating input tensor: {e}"); traceback.print_exc()
        return torch.zeros((1, 1, dim), dtype=torch.float32, device=device)

# --- Learning Bot Action Selection ---
def get_learning_action(bot_state, senses, env, all_bots_dict, direct_player_action, config):
    """ Determines the action for a learning bot, handling AI, player target, and direct player actions """
    bot_id = bot_state['id']
    chosen_action = -1; mode_code = 5; mode_str = "Idle"

    if bot_state['freezeTimer'] > 0:
        bot_state['target_coordinate'] = None; mode_code = 5; mode_str = "Frozen"
        return -1, mode_str, mode_code

    if bot_state.get('is_player_controlled', False):
        control_influence_percent = max(0.0, min(100.0, config['PLAYER_CONTROL_PERCENT']))
        player_action = -1; player_mode_str = "Player Idle"; player_mode_code = 5

        if direct_player_action is not None and 0 <= direct_player_action < config['NUM_ACTIONS']:
            player_action = direct_player_action; player_mode_str = f"Player Direct ({player_action})"; player_mode_code = 3
            bot_state['target_coordinate'] = None
        elif bot_state.get('target_coordinate'):
            target = bot_state['target_coordinate']; current_pos = bot_state['pos']
            dist = env._manhattan_distance(current_pos, target); player_mode_code = 4

            if dist == 0: player_action = -1; player_mode_str = "Player Target Reached"; bot_state['target_coordinate'] = None
            else:
                temp_action = -1
                if dist == 1:
                    opponent_at_target = next((b for bid, b in all_bots_dict.items() if bid != bot_id and b['pos'] == target and b['freezeTimer'] <= 0), None)
                    if opponent_at_target: temp_action = 4; player_mode_str = f"Player Target Punch"
                    else:
                        goal_at_target = next((g for g in env.goals if g['id'] not in env.claimed_goals and g['x'] == target['x'] and g['y'] == target['y']), None)
                        if goal_at_target: temp_action = 5; player_mode_str = f"Player Target Claim"

                if temp_action == -1:
                    path_to_target = env.find_path(current_pos, target, all_bots_dict, bot_id)
                    if path_to_target: temp_action = path_to_target[0]; player_mode_str = f"Player Target Move {temp_action}"
                    else: temp_action = -1; player_mode_str = "Player Target Blocked"
                player_action = temp_action

        if control_influence_percent < 100.0 and player_action != -1:
            ai_action = -1; ai_mode_str = "AI Blend"
            if neural_memory_manager and NEURAL_LIB_AVAILABLE:
                memory_state = bot_state.get('memory_state'); policy_head = bot_state.get('policy_head')
                if memory_state and policy_head:
                    try:
                        input_tensor = _get_input_tensor_for_bot(bot_state, senses, config)
                        retrieved, _, _, _, _ = neural_memory_manager.forward_step(input_tensor, memory_state, detach_next_state=True)
                        policy_head.eval(); policy_head = policy_head.to(device)
                        with torch.no_grad(): action_logits = policy_head(retrieved.to(device).squeeze(0))
                        ai_action = torch.argmax(action_logits, dim=-1).item()
                    except Exception as e: print(f"Error in AI blend pred: {e}"); ai_action = random.choice(list(range(config['NUM_ACTIONS'])))
                else: ai_action = random.choice(list(range(config['NUM_ACTIONS'])))
            else: ai_action = random.choice(list(range(config['NUM_ACTIONS'])))

            if random.uniform(0, 100) < control_influence_percent: chosen_action = player_action; mode_str = player_mode_str; mode_code = player_mode_code
            else: chosen_action = ai_action; mode_str = ai_mode_str; mode_code = 0
        else: chosen_action = player_action; mode_str = player_mode_str; mode_code = player_mode_code

    else: # AI Control Logic
        if not neural_memory_manager or not NEURAL_LIB_AVAILABLE: mode_str = "Error (No NNM)"; mode_code = 5; return -1, mode_str, mode_code
        memory_state = bot_state.get('memory_state'); policy_head = bot_state.get('policy_head')
        if not memory_state or not policy_head: mode_str = "Error (Components)"; mode_code = 5; return -1, mode_str, mode_code

        base_explore_rate_percent = config['LEARNING_BOT_BASE_EXPLORATION_RATE']
        anomaly_factor = min(3.0, 1.0 + bot_state.get('last_anomaly_proxy', 0.0) * 10.0)
        current_exploration_threshold_percent = min(98.0, base_explore_rate_percent * anomaly_factor)
        is_exploring = random.uniform(0, 100) < current_exploration_threshold_percent

        if is_exploring:
            rule_explore_percent_chance = config['LEARNING_BOT_RULE_EXPLORE_PERCENT']
            if random.uniform(0, 100) < rule_explore_percent_chance:
                mode_code = 2; hc_action, hc_mode = get_hardcoded_action(bot_state, senses, env, all_bots_dict)
                chosen_action = hc_action; mode_str = f"Explore Rule ({current_exploration_threshold_percent:.1f}%) -> {hc_mode}"
            else:
                mode_code = 1; chosen_action = random.choice(list(range(config['NUM_ACTIONS'])))
                mode_str = f"Explore Random ({current_exploration_threshold_percent:.1f}%)"
        else: # Exploitation
            mode_code = 0
            try:
                input_tensor = _get_input_tensor_for_bot(bot_state, senses, config)
                retrieved, _, _, _, _ = neural_memory_manager.forward_step(input_tensor, memory_state, detach_next_state=True)
                policy_head.eval(); policy_head = policy_head.to(device)
                retrieved_on_device = retrieved.to(device)
                with torch.no_grad(): action_logits = policy_head(retrieved_on_device.squeeze(0))
                chosen_action = torch.argmax(action_logits, dim=-1).item()
                mode_str = f"Exploit (Predict {chosen_action})"
            except Exception as e:
                print(f"Error: Exploitation failed for AI bot {bot_id}: {e}"); traceback.print_exc()
                mode_str = "Error (Exploitation)"; mode_code = 5
                chosen_action = random.choice(list(range(config['NUM_ACTIONS'])))

    if not isinstance(chosen_action, int) or chosen_action < -1 or chosen_action >= config['NUM_ACTIONS']:
        chosen_action = -1
        if mode_code != 5: mode_str += " -> IdleFallback"

    return chosen_action, mode_str, mode_code


# ================================================================
# --- Simulation Setup & Control ---
# ================================================================

def create_learning_bot_instance(bot_id, start_pos, config):
    """ Creates the state dictionary for a new learning bot using the manager """
    global neural_memory_manager
    if not neural_memory_manager or not NEURAL_LIB_AVAILABLE:
         raise RuntimeError(f"NN Manager not ready. Cannot create learning bot {bot_id}.")

    nnm_dim = neural_memory_manager.dim
    print(f"Creating Learning Bot {bot_id} (DIM={nnm_dim}) on {device}")
    initial_mem_state = neural_memory_manager.get_initial_state()

    try:
        policy_head = nn.Linear(nnm_dim, config['NUM_ACTIONS']).to(device)
        nn.init.xavier_uniform_(policy_head.weight)
        if policy_head.bias is not None: nn.init.zeros_(policy_head.bias)
    except Exception as e:
        print(f"FATAL ERROR: Failed to create policy head for {bot_id} on {device}: {e}")
        traceback.print_exc(); raise

    return {
        'id': bot_id, 'type': 'Learning', 'pos': start_pos.copy(),
        'steps': 0, 'goalsReachedThisRound': 0, 'goalsReachedTotal': 0,
        'freezeTimer': 0, 'lastAction': -1, 'mode': 'Init', 'senses': {},
        'memory_state': initial_mem_state, # NeuralMemState tuple
        'policy_head': policy_head,        # nn.Module instance
        'last_anomaly_proxy': 0.0,         # EMA of anomaly score
        'is_player_controlled': False, 'target_coordinate': None,
        'original_bot_id': bot_id,
        'lastPos': {'x':-1,'y':-1}, 'stuckCounter': 0, 'lastMoveAttempt': -1,
        'currentPath': None, 'targetGoalId': None, 'randomMoveCounter': 0,
        'last_av_data': None
    }

def create_hardcoded_bot_instance(bot_id, start_pos):
     """ Creates the state dictionary for a new hardcoded bot """
     return {
         'id': bot_id, 'type': 'Hardcoded', 'pos': start_pos.copy(),
         'steps': 0, 'goalsReachedThisRound': 0, 'goalsReachedTotal': 0,
         'freezeTimer': 0, 'lastAction': -1, 'mode': 'Init', 'senses': {},
         'lastPos': {'x':-1,'y':-1}, 'stuckCounter': 0, 'lastMoveAttempt': -1,
         'currentPath': None, 'targetGoalId': None, 'randomMoveCounter': 0
     }

def setup_simulation(full_reset=False, new_environment=False):
    """ Sets up or resets the simulation environment and bots. """
    global environment, bots, round_number, stats, current_config, players, player_direct_actions, av_manager
    print(f"--- Setting up Simulation (Full Reset: {full_reset}, New Env: {new_environment}) ---")

    # Stop AV system *before* potentially changing bot counts or NNM
    if av_manager:
        print("Stopping existing AV system before setup...")
        stop_av_system(av_manager)
        av_manager = None

    if full_reset:
        print("Performing full reset...")
        round_number = 0; stats = {'hc_total_goals': 0, 'learning_total_goals': 0}
        nnm_recreated_or_moved = update_neural_memory_manager_instance()
        if not neural_memory_manager:
            print("CRITICAL: NN Manager update failed during full reset. Aborting setup.")
            return False
        print("Clearing existing bot states..."); bots.clear(); players.clear(); player_direct_actions.clear()
        environment = None; new_environment = True
        if device.type == 'cuda': torch.cuda.empty_cache()
    else: # Round reset
        round_number += 1
        if environment:
             env_structure_keys = ['GRID_SIZE', 'NUM_HC_BOTS', 'NUM_LEARNING_BOTS', 'NUM_GOALS']
             env_changed_structurally = any(
                 current_config[k] != getattr(environment, k.lower().replace('num_','num_'), None)
                 for k in env_structure_keys if hasattr(environment, k.lower().replace('num_','num_'))
             )
             if env_changed_structurally:
                  print("Environment structure change detected, forcing full reset logic...")
                  full_reset = True; new_environment = True; environment = None
             elif new_environment or current_config.get("RANDOMIZE_ENV_PER_ROUND", False):
                  print("Randomizing environment for new round..."); environment.randomize()
             else: environment.reset_round_state()
        else: print("No environment found, forcing full reset..."); full_reset = True; new_environment = True
        player_direct_actions.clear()

    # --- Recreate Environment if needed ---
    if environment is None:
         print("Recreating environment...")
         try:
             obstacle_range = (current_config['OBSTACLES_FACTOR_MIN'], current_config['OBSTACLES_FACTOR_MAX'])
             dist_factors = {k: current_config.get(k) for k in ['MIN_GOAL_START_DISTANCE_FACTOR', 'MIN_BOT_START_DISTANCE_FACTOR', 'MIN_BOT_GOAL_DISTANCE_FACTOR']}
             environment = GridEnvironment(current_config['GRID_SIZE'], current_config['NUM_GOALS'], obstacle_range, current_config['NUM_HC_BOTS'], current_config['NUM_LEARNING_BOTS'], dist_factors)
             if environment.num_hc_bots != current_config['NUM_HC_BOTS'] or environment.num_learning_bots != current_config['NUM_LEARNING_BOTS']:
                 print(f"Environment adjusted bot counts: HC={environment.num_hc_bots}, Lrn={environment.num_learning_bots}. Updating config.")
                 current_config['NUM_HC_BOTS'] = environment.num_hc_bots
                 current_config['NUM_LEARNING_BOTS'] = environment.num_learning_bots
             full_reset = True # Treat as full reset for bot handling
         except Exception as e: print(f"FATAL: Environment creation failed: {e}"); traceback.print_exc(); return False

    # --- Create/Reset Bot States ---
    new_bots = {}
    bot_starts = environment.start_positions if environment else []
    required_bots = environment.num_hc_bots + environment.num_learning_bots
    if len(bot_starts) != required_bots:
         print(f"FATAL MISMATCH: Env start positions ({len(bot_starts)}) != env bot counts ({required_bots}). Setup failed.")
         return False

    try:
        for start_pos_data in bot_starts:
            bot_id = start_pos_data['id']; bot_type = start_pos_data['type']
            start_pos = {'x': start_pos_data['x'], 'y': start_pos_data['y']}
            controlling_sid = next((sid for sid, p_data in players.items() if p_data['original_bot_id'] == bot_id), None)

            if bot_id in bots and not full_reset: # Reset existing bot
                existing_bot = bots[bot_id]
                existing_bot.update({
                    'pos': start_pos.copy(), 'steps': 0, 'goalsReachedThisRound': 0,
                    'freezeTimer': 0, 'lastAction': -1, 'mode': 'Reset', 'senses': {},
                    'lastPos': {'x':-1,'y':-1}, 'stuckCounter': 0, 'lastMoveAttempt': -1,
                    'currentPath': None, 'targetGoalId': None, 'randomMoveCounter': 0,
                    'last_av_data': None
                })
                if bot_type == 'Learning':
                     existing_bot['last_anomaly_proxy'] = 0.0
                     existing_bot['target_coordinate'] = None
                     if neural_memory_manager: existing_bot['memory_state'] = neural_memory_manager.get_initial_state()
                     else: print(f"Warning: NNM missing during reset for {bot_id}")
                     existing_bot['is_player_controlled'] = bool(controlling_sid)
                new_bots[bot_id] = existing_bot
            else: # Create new bot instance
                if bot_type == 'Hardcoded': new_bots[bot_id] = create_hardcoded_bot_instance(bot_id, start_pos)
                elif bot_type == 'Learning':
                    if neural_memory_manager:
                         new_bots[bot_id] = create_learning_bot_instance(bot_id, start_pos, current_config)
                         if controlling_sid: new_bots[bot_id]['is_player_controlled'] = True; players[controlling_sid]['player_bot_id'] = bot_id
                    else: print(f"Warning: Cannot create new learning bot {bot_id}, NNM not ready.")

        bots = new_bots
        # Cleanup players whose original bots no longer exist
        for sid, player_data in list(players.items()):
            if player_data['original_bot_id'] not in bots:
                print(f"Removing player SID {sid} as their original bot {player_data['original_bot_id']} no longer exists.")
                del players[sid]

    except Exception as e: print(f"Error: Bot creation/reset failed: {e}"); traceback.print_exc(); return False

    # --- Setup AV system *after* bots are created/reset ---
    num_learning_bots_actual = sum(1 for b in bots.values() if b['type'] == 'Learning')
    if current_config.get("ENABLE_AV", False) and AV_LIB_AVAILABLE:
        print("Setting up AV system...")
        av_dim = current_config.get('LEARNING_BOT_DIM', 0)
        vis_mode = current_config.get("VISUALIZATION_MODE", "none")
        enable_output = current_config.get("ENABLE_AV_OUTPUT", False)
        sio_instance = socketio if vis_mode == 'web' else None

        av_manager = setup_av_system(
            num_bots=num_learning_bots_actual,
            dim=av_dim,
            device=device,
            enable_output=enable_output,
            visualization_mode=vis_mode,
            socketio_instance=sio_instance
        )
        if not av_manager:
            print("Warning: AV setup failed but simulation continues.")
        elif not av_manager.is_setup and enable_output:
             print(f"Warning: AV Manager created but failed to setup output (Mode: {vis_mode}). Check library installs/permissions.")
        elif av_manager.is_setup:
             print(f"AV Manager setup successful (Mode: {vis_mode}, Audio: {av_manager.enable_audio_output}).")

    print(f"Setup complete for Round {round_number}. Active Bots: {list(bots.keys())}")
    socketio.emit('config_update', current_config) # Send potentially updated config
    return True

# --- Simulation Step ---
def simulation_step():
    """ Performs one step of the simulation for all active bots """
    global player_direct_actions, av_manager
    if not environment or not bots: return False

    round_over = False
    max_steps_reached_for_all = True
    bot_ids_this_step = list(bots.keys())
    current_direct_actions = player_direct_actions.copy()
    player_direct_actions.clear()
    live_av_data = {} # Collect data for AV system {av_idx: {data}}

    learning_bot_ids = sorted([bid for bid, b in bots.items() if b['type'] == 'Learning'])
    bot_id_to_av_idx = {bot_id: idx for idx, bot_id in enumerate(learning_bot_ids)}

    # --- Bot Action Phase ---
    for bot_id in bot_ids_this_step:
        if bot_id not in bots: continue
        bot_state = bots[bot_id]

        if bot_state['steps'] >= current_config['MAX_STEPS_PER_ROUND']: continue
        else: max_steps_reached_for_all = False

        action_attempted = -1; mode_str = "Idle"; mode_code = 5
        next_pos = bot_state['pos'].copy()

        try:
            bot_state['senses'] = environment.get_sensory_data(bot_state, bots, current_config['VISIBILITY_RANGE'])

            if bot_state['freezeTimer'] > 0: action_attempted = -1; mode_str = "Frozen"; mode_code = 5
            elif bot_state['type'] == 'Hardcoded': action_attempted, mode_str = get_hardcoded_action(bot_state, bot_state['senses'], environment, bots); mode_code = 2
            elif bot_state['type'] == 'Learning':
                direct_action = None
                if bot_state.get('is_player_controlled', False):
                    controlling_sid = next((sid for sid, p_data in players.items() if p_data['player_bot_id'] == bot_id), None)
                    if controlling_sid and controlling_sid in current_direct_actions: direct_action = current_direct_actions[controlling_sid]
                action_attempted, mode_str, mode_code = get_learning_action(bot_state, bot_state['senses'], environment, bots, direct_action, current_config)

            bot_state['mode'] = mode_str

            # --- Execute Action ---
            if action_attempted != -1 and bot_state['freezeTimer'] <= 0:
                if 0 <= action_attempted <= 3: # Move
                    intended_pos = environment.perform_move_action(bot_state['pos'], action_attempted)
                    occupied = any(bid != bot_id and b['pos'] == intended_pos for bid, b in bots.items())
                    if environment.is_valid(intended_pos) and not occupied: next_pos = intended_pos
                    else: bot_state['mode'] += " (Blocked)"
                elif action_attempted == 4: # Punch
                    target_bot = next((ob for ob_id, ob in bots.items() if ob_id != bot_id and environment._manhattan_distance(bot_state['pos'], ob['pos']) == 1 and ob['freezeTimer'] <= 0), None)
                    if target_bot: target_bot['freezeTimer'] = current_config['FREEZE_DURATION']; bot_state['mode'] += f" (Hit {target_bot['id']})"
                    else: bot_state['mode'] += " (Punch Miss)"
                elif action_attempted == 5: # Claim
                    adj_goal = environment.get_adjacent_unclaimed_goal(bot_state['pos'])
                    if adj_goal and environment.claim_goal(adj_goal['id'], bot_id):
                         bot_state['goalsReachedThisRound'] += 1; bot_state['goalsReachedTotal'] += 1
                         if bot_state['type'] == 'Hardcoded': stats['hc_total_goals'] += 1
                         else: stats['learning_total_goals'] += 1
                         bot_state['mode'] += f" (Claimed {adj_goal['id']})"
                         if environment.are_all_goals_claimed(): round_over = True; print(f"--- Round {round_number} Over: All goals claimed! ---")
                    else: bot_state['mode'] += " (Claim Failed)"

            bot_state['pos'] = next_pos
            bot_state['steps'] += 1
            if bot_state['freezeTimer'] > 0: bot_state['freezeTimer'] -= 1

            # --- Update Neural Memory & Collect AV Data (Learning Bots Only) ---
            if bot_state['type'] == 'Learning' and neural_memory_manager:
                try:
                    bot_state_for_update = bot_state.copy(); bot_state_for_update['lastAction'] = action_attempted
                    input_tensor = _get_input_tensor_for_bot(bot_state_for_update, bot_state['senses'], current_config)
                    input_tensor_on_device = input_tensor.to(neural_memory_manager.target_device)

                    retrieved, next_mem_state, anomaly, weight_diff, _ = neural_memory_manager.forward_step(
                        input_tensor_on_device, bot_state['memory_state'], detach_next_state=True
                    )
                    bot_state['memory_state'] = next_mem_state
                    current_anomaly_val = anomaly.item()
                    bot_state['last_anomaly_proxy'] = current_anomaly_val * 0.1 + bot_state.get('last_anomaly_proxy', 0.0) * 0.9

                    # Store data needed for AV (references to GPU tensors if using CUDA)
                    av_idx = bot_id_to_av_idx.get(bot_id)
                    # Check if AV is enabled AND manager exists AND has output enabled
                    if current_config.get("ENABLE_AV", False) and av_manager and av_idx is not None and \
                       (av_manager.enable_audio_output or av_manager.enable_visual_output):
                        av_data_packet = {
                            'anomaly_score': anomaly, # Tensor
                            'retrieved_memory_vector': retrieved, # Tensor [1,1,dim]
                            'weight_change_metric': weight_diff, # Tensor
                            'input_stream_vector': input_tensor_on_device, # Tensor [1,1,dim]
                            'mode_code': mode_code, # int
                            'is_player_controlled': bot_state.get('is_player_controlled', False) # bool
                        }
                        live_av_data[av_idx] = av_data_packet

                except Exception as e:
                    print(f"Error: NNM update failed for bot {bot_id} (Action: {action_attempted}): {e}"); traceback.print_exc()

            bot_state['lastAction'] = action_attempted

        except Exception as e:
            print(f"CRITICAL Error processing bot {bot_id}: {e}"); traceback.print_exc()
            global simulation_running; simulation_running = False
            socketio.emit('simulation_stopped', {'message': f'Error processing bot {bot_id}.'})
            return False # Stop simulation

    # --- Update AV System (After processing all bots) ---
    if current_config.get("ENABLE_AV", False) and av_manager and live_av_data:
        update_av_system(av_manager, live_av_data) # Pass dict {av_idx: data_packet}

    # Check for round end conditions
    if not round_over and max_steps_reached_for_all:
        round_over = True; print(f"--- Round {round_number} Over: Max steps reached! ---")

    return not round_over # True to continue round, False to end


# --- Simulation Loop ---
def simulation_loop():
    """ Main simulation loop managed by eventlet background task. """
    global simulation_running, round_number, simulation_loop_task
    print("Simulation loop started.")
    loop_count = 0
    emit_interval_steps = 2
    last_emit_time = time.monotonic()
    min_emit_interval_time = 0.04 # ~25 FPS target

    while simulation_running:
        loop_start_time = time.monotonic()
        try:
            continue_round = simulation_step()

            if not simulation_running: break

            if not continue_round: # Round ended
                 new_env_next = current_config.get("RANDOMIZE_ENV_PER_ROUND", False)
                 if setup_simulation(full_reset=False, new_environment=new_env_next):
                     loop_count = 0; last_emit_time = time.monotonic()
                     emit_state(); print(f"Starting Round {round_number}...")
                 else:
                     print("Error: Failed next round setup. Stopping."); simulation_running = False
                     socketio.emit('simulation_stopped', {'message': 'Error setting up next round.'}); break
            else: # Round continues
                 loop_count += 1; current_time = time.monotonic()
                 if loop_count % emit_interval_steps == 0 or (current_time - last_emit_time) > min_emit_interval_time:
                      emit_state(); last_emit_time = current_time

            elapsed_time = time.monotonic() - loop_start_time
            target_delay = current_config['SIMULATION_SPEED_MS'] / 1000.0
            delay = max(0.001, target_delay - elapsed_time)
            eventlet.sleep(delay) # Non-blocking sleep

        except Exception as e:
            print(f"CRITICAL Error in simulation loop: {e}"); traceback.print_exc()
            simulation_running = False
            socketio.emit('simulation_stopped', {'message': f'Runtime Error: {e}'}); break

    print("Simulation loop finished.")
    emit_state() # Emit final state
    simulation_loop_task = None


# ================================================================
# --- Flask Routes & SocketIO Events ---
# ================================================================
@app.route('/')
def index():
    try: return render_template('index.html')
    except Exception as e: print(f"Error rendering template: {e}"); traceback.print_exc(); return "Error loading page.", 500

# --- Add Route for Web Visualizer (Example) ---
@app.route('/visuals')
def visuals_page():
    # This would render an HTML page containing the web visualization client (e.g., using Plotly.js or Three.js)
    # For now, just return a placeholder
    return """
    <!DOCTYPE html><html><head><title>Web Visualizer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.min.js"></script>
    <style>body { font-family: sans-serif; background-color: #222; color: #eee; } #data { white-space: pre-wrap; font-family: monospace; }</style>
    </head><body><h1>Web Visualizer Placeholder</h1>
    <p>Connects to SocketIO namespace '/visuals'. Data will appear below.</p>
    <div id="status">Connecting...</div><div id="data"></div>
    <script>
        const statusEl = document.getElementById('status');
        const dataEl = document.getElementById('data');
        const socket = io('/visuals'); // Connect to the specific namespace
        socket.on('connect', () => { statusEl.textContent = 'Connected to /visuals'; });
        socket.on('disconnect', () => { statusEl.textContent = 'Disconnected from /visuals'; });
        socket.on('visual_update', (data) => {
            // In a real implementation, use this data to update Plotly/Three.js/etc.
            dataEl.textContent = JSON.stringify(data, null, 2);
        });
    </script></body></html>
    """

# --- SocketIO Namespace for Web Visualizer ---
class VisualsNamespace(Namespace):
    def on_connect(self):
        print(f"Web Visualizer client connected: {request.sid}")
        # Optionally send initial setup data if needed
        # emit('initial_visual_config', {'num_bots': current_config.get('NUM_LEARNING_BOTS', 0)})

    def on_disconnect(self):
        print(f"Web Visualizer client disconnected: {request.sid}")

# Register the namespace
socketio.on_namespace(VisualsNamespace('/visuals'))


# --- Main SocketIO Events ---
@socketio.on('connect')
def handle_connect():
    sid = request.sid; print(f"Client connected: {sid}")
    try:
        if environment is None or not bots:
             print("First connection or missing state, ensuring initial setup...")
             if not setup_simulation(full_reset=True, new_environment=True):
                  emit('status_update', {'message': 'Error: Server setup failed.'}, room=sid); return
        state = get_game_state()
        state['isRunning'] = simulation_running
        emit('initial_state', state, room=sid)
        print(f"Initial state sent to {sid} (Running: {simulation_running})")
    except Exception as e: print(f"Error sending initial data to {sid}: {e}"); traceback.print_exc()

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid; print(f"Client disconnected: {sid}")
    if sid in players:
        player_data = players.pop(sid)
        original_bot_id = player_data['original_bot_id']
        print(f"Player {sid} released control of bot {original_bot_id}")
        if original_bot_id in bots:
            bots[original_bot_id]['is_player_controlled'] = False
            bots[original_bot_id]['target_coordinate'] = None
            bots[original_bot_id]['mode'] = "AI Control"
            socketio.emit('player_left', {'player_id': original_bot_id})
            emit_state()
    if sid in player_direct_actions: del player_direct_actions[sid]

# --- get_game_state helper ---
def get_game_state():
    """ Creates a JSON-serializable representation of the game state """
    serializable_bots = {}
    for bot_id, bot_state in bots.items():
        s_bot = {
            'id': bot_state['id'], 'type': bot_state['type'], 'pos': bot_state['pos'],
            'freezeTimer': bot_state['freezeTimer'], 'mode': bot_state['mode'],
            'goals_round': bot_state.get('goalsReachedThisRound', 0),
            'is_player': bot_state.get('is_player_controlled', False),
            'target_coord': bot_state.get('target_coordinate')
        }
        if bot_state['type'] == 'Learning':
            s_bot['anomaly'] = round(bot_state.get('last_anomaly_proxy', 0.0), 5)
        serializable_bots[bot_id] = s_bot

    return {
        'environment': environment.get_state() if environment else None,
        'bots': serializable_bots, 'round': round_number, 'stats': stats,
        'config': current_config # Send full config
    }

# --- emit_state helper ---
def emit_state():
    """ Safely gets and emits the current game state """
    try:
        state = get_game_state()
        state['isRunning'] = simulation_running
        socketio.emit('update_state', state)
    except Exception as e:
        print(f"Error emitting state: {e}"); traceback.print_exc()


# --- Player/Control SocketIO handlers (No changes needed from previous version) ---
@socketio.on('join_game')
def handle_join_game(data=None):
    sid = request.sid
    if sid in players: emit('join_ack', {'success': False, 'message': 'Already controlling a bot.'}, room=sid); return
    if not environment or not bots: emit('join_ack', {'success': False, 'message': 'Simulation not ready.'}, room=sid); return

    target_bot_id = data.get('target_bot_id') if data else None
    available_bot_id = None

    if target_bot_id:
         if target_bot_id in bots and bots[target_bot_id]['type'] == 'Learning' and not bots[target_bot_id].get('is_player_controlled', False):
             available_bot_id = target_bot_id
         else:
             message = f"Bot {target_bot_id} is not available or not a learning bot."
             emit('join_ack', {'success': False, 'message': message}, room=sid); return
    else:
         available_bot_id = next((bid for bid, b in bots.items() if b['type'] == 'Learning' and not b.get('is_player_controlled', False)), None)

    if available_bot_id:
        original_id = bots[available_bot_id]['original_bot_id']
        print(f"Player {sid} taking control of Learning Bot {available_bot_id} (Original: {original_id})")
        bots[available_bot_id]['is_player_controlled'] = True
        bots[available_bot_id]['target_coordinate'] = None
        bots[available_bot_id]['mode'] = "Player Control"
        players[sid] = {'player_bot_id': available_bot_id, 'original_bot_id': original_id}
        emit('join_ack', {'success': True, 'player_id': available_bot_id, 'original_bot_id': original_id}, room=sid)
        socketio.emit('player_joined', {'player_id': available_bot_id})
        emit_state()
    else:
        message = "No available Learning Bots to control."
        emit('join_ack', {'success': False, 'message': message}, room=sid)
        print(f"Player {sid} failed to join: {message}")

@socketio.on('rejoin_game')
def handle_rejoin_game(data):
    sid = request.sid
    if sid in players: emit('rejoin_ack', {'success': False, 'message': 'Already controlling.'}, room=sid); return
    original_bot_id = data.get('originalBotId')
    if not original_bot_id: emit('rejoin_ack', {'success': False, 'message': 'No original bot ID provided.'}, room=sid); return

    print(f"Player {sid} attempting rejoin for original bot: {original_bot_id}")
    if original_bot_id in bots and bots[original_bot_id]['type'] == 'Learning':
        already_controlled = any(other_sid != sid and p_data['original_bot_id'] == original_bot_id for other_sid, p_data in players.items())
        if not already_controlled:
            bots[original_bot_id]['is_player_controlled'] = True
            bots[original_bot_id]['target_coordinate'] = None
            bots[original_bot_id]['mode'] = "Player Control (Rejoin)"
            players[sid] = {'player_bot_id': original_bot_id, 'original_bot_id': original_bot_id}
            emit('rejoin_ack', {'success': True, 'player_id': original_bot_id, 'original_bot_id': original_bot_id}, room=sid)
            socketio.emit('player_joined', {'player_id': original_bot_id})
            emit_state()
            print(f"Player {sid} rejoined control of {original_bot_id}")
        else:
            emit('rejoin_ack', {'success': False, 'message': f'Bot {original_bot_id} already controlled by another player.'}, room=sid)
    else:
        emit('rejoin_ack', {'success': False, 'message': f'Original bot {original_bot_id} not available or not a learning bot.'}, room=sid)

@socketio.on('leave_game')
def handle_leave_game(data=None):
    sid = request.sid
    if sid in players:
        player_data = players.pop(sid)
        original_bot_id = player_data['original_bot_id']
        print(f"Player {sid} leaving control of bot {original_bot_id}")
        if original_bot_id in bots:
            bots[original_bot_id]['is_player_controlled'] = False
            bots[original_bot_id]['target_coordinate'] = None
            bots[original_bot_id]['mode'] = "AI Control"
            socketio.emit('player_left', {'player_id': original_bot_id})
            emit_state()
        emit('leave_ack', {'success': True}, room=sid)
        if sid in player_direct_actions: del player_direct_actions[sid]
    else: emit('leave_ack', {'success': False, 'message': 'Not controlling a bot.'}, room=sid)

@socketio.on('player_action')
def handle_player_action(data):
    sid = request.sid
    if sid in players:
        player_bot_id = players[sid]['player_bot_id']
        if player_bot_id in bots and bots[player_bot_id].get('is_player_controlled', False):
            action = data.get('action')
            try:
                action_int = int(action)
                if 0 <= action_int < current_config['NUM_ACTIONS']:
                    player_direct_actions[sid] = action_int
                else: print(f"Warning: Invalid action value {action} from {sid}")
            except (ValueError, TypeError): print(f"Warning: Non-integer action '{action}' from {sid}")

@socketio.on('update_player_target')
def handle_update_player_target(data):
    sid = request.sid
    if sid in players:
        player_bot_id = players[sid]['player_bot_id']
        if player_bot_id in bots and bots[player_bot_id].get('is_player_controlled', False):
            target = data.get('target')
            if target is None: bots[player_bot_id]['target_coordinate'] = None
            elif isinstance(target, dict) and 'x' in target and 'y' in target:
                try:
                    tx, ty = int(target['x']), int(target['y'])
                    grid_size = environment.size if environment else current_config['GRID_SIZE']
                    if 0 <= tx < grid_size and 0 <= ty < grid_size: bots[player_bot_id]['target_coordinate'] = {'x': tx, 'y': ty}
                    else: bots[player_bot_id]['target_coordinate'] = None
                except (ValueError, TypeError): bots[player_bot_id]['target_coordinate'] = None
            else: bots[player_bot_id]['target_coordinate'] = None

@socketio.on('start_simulation')
def handle_start_simulation(data=None):
    global simulation_running, simulation_loop_task
    if simulation_running: emit('status_update', {'message': 'Simulation already running.'}, room=request.sid); return
    print("Start simulation request received.")
    if environment is None or not bots:
         print("Initial setup required before starting.")
         if not setup_simulation(full_reset=False, new_environment=True):
              socketio.emit('simulation_stopped', {'message': 'Initialization failed on start.'}); return
    simulation_running = True
    if simulation_loop_task is None or (hasattr(simulation_loop_task, 'dead') and simulation_loop_task.dead):
         print("Starting background simulation loop task.")
         simulation_loop_task = socketio.start_background_task(simulation_loop)
    socketio.emit('simulation_started'); emit_state()

@socketio.on('stop_simulation')
def handle_stop_simulation(data=None):
    global simulation_running
    if not simulation_running: emit('status_update', {'message': 'Simulation already stopped.'}, room=request.sid); return
    print("Stop simulation request received.")
    simulation_running = False
    socketio.emit('simulation_stopped', {'message': 'Simulation stopped.'})

@socketio.on('reset_round')
def handle_reset_round(data=None):
    global simulation_running
    was_running = simulation_running; print("Reset round request received.")
    if simulation_running: handle_stop_simulation(); eventlet.sleep(0.1)
    new_env = current_config.get("RANDOMIZE_ENV_PER_ROUND", False)
    if setup_simulation(full_reset=False, new_environment=new_env):
        emit_state(); status_msg = 'New Round Ready.' + (' Press Start to run.' if not was_running else '')
        socketio.emit('status_update', {'message': status_msg})
    else: socketio.emit('status_update', {'message': 'Error resetting round.'})
    socketio.emit('simulation_stopped', {'message': 'New Round Ready.'})

@socketio.on('reset_full')
def handle_reset_full(data=None):
    global simulation_running
    was_running = simulation_running; print("Full reset request received.")
    if simulation_running: handle_stop_simulation(); eventlet.sleep(0.1)
    if setup_simulation(full_reset=True, new_environment=True):
        emit_state(); status_msg = 'Full Reset Complete.' + (' Press Start to run.' if not was_running else '')
        socketio.emit('status_update', {'message': status_msg})
    else: socketio.emit('status_update', {'message': 'Error during full reset.'})
    socketio.emit('simulation_stopped', {'message': 'Full Reset Complete.'})


@socketio.on('update_config')
def handle_update_config(data):
    global current_config, av_manager
    if simulation_running: emit('config_update_ack', {'success': False, 'message': 'Stop simulation before changing parameters.'}, room=request.sid); return

    try:
        new_config_data = data.get('config', {})
        needs_full_reset = False; needs_round_reset = False
        changed_keys = []; temp_config = copy.deepcopy(current_config)
        print("Received config update request:", new_config_data)

        # Define which keys trigger which reset type (Added VISUALIZATION_MODE)
        reset_all_keys = ['GRID_SIZE', 'NUM_HC_BOTS', 'NUM_LEARNING_BOTS', 'NUM_GOALS', 'LEARNING_BOT_DIM', 'LEARNING_BOT_MEM_DEPTH', 'LEARNING_BOT_LR', 'LEARNING_BOT_WEIGHT_DECAY', 'LEARNING_BOT_MOMENTUM', 'LEARNING_BOT_MAX_GRAD_NORM', 'ENABLE_AV', 'ENABLE_AV_OUTPUT', 'VISUALIZATION_MODE']
        reset_round_keys = ['MAX_STEPS_PER_ROUND', 'VISIBILITY_RANGE', 'OBSTACLES_FACTOR_MIN', 'OBSTACLES_FACTOR_MAX', 'MIN_GOAL_START_DISTANCE_FACTOR', 'MIN_BOT_START_DISTANCE_FACTOR', 'MIN_BOT_GOAL_DISTANCE_FACTOR']
        immediate_update_keys = ['SIMULATION_SPEED_MS', 'FREEZE_DURATION', 'LEARNING_BOT_BASE_EXPLORATION_RATE', 'LEARNING_BOT_RULE_EXPLORE_PERCENT', 'PLAYER_CONTROL_PERCENT', 'RANDOMIZE_ENV_PER_ROUND']

        for key, value in new_config_data.items():
            if key in DEFAULT_CONFIG:
                try:
                    default_type = type(DEFAULT_CONFIG[key]); current_value = temp_config.get(key)
                    if value is None: continue
                    # Type Conversion and Validation
                    if default_type is bool: converted_value = str(value).lower() in ['true', '1', 'yes', 'on']
                    elif default_type is int: converted_value = int(round(float(value)))
                    elif default_type is float: converted_value = float(value)
                    else: converted_value = default_type(value) # String or other (like VISUALIZATION_MODE)

                    # Apply constraints (Add VISUALIZATION_MODE validation)
                    if key == "VISUALIZATION_MODE" and converted_value not in ['vispy', 'matplotlib', 'web', 'none']:
                        print(f"Warning: Invalid VISUALIZATION_MODE '{converted_value}'. Using 'none'.")
                        converted_value = 'none'
                    # ... (keep other constraints) ...
                    if key == "GRID_SIZE": converted_value = max(10, min(200, converted_value))
                    if key == "NUM_HC_BOTS": converted_value = max(0, min(100, converted_value))
                    if key == "NUM_LEARNING_BOTS": converted_value = max(0, min(100, converted_value))
                    if key == "NUM_GOALS": converted_value = max(0, min(500, converted_value))
                    if key == "LEARNING_BOT_DIM": converted_value = max(32, min(4096, (converted_value // 16) * 16 if converted_value > 32 else 32))
                    if key == "LEARNING_BOT_MEM_DEPTH": converted_value = max(1, min(12, converted_value))
                    if key == "SIMULATION_SPEED_MS": converted_value = max(1, min(2000, converted_value))
                    if key == "PLAYER_CONTROL_PERCENT": converted_value = max(0.0, min(100.0, converted_value))
                    if key == "LEARNING_BOT_BASE_EXPLORATION_RATE": converted_value = max(0.0, min(100.0, converted_value))
                    if key == "LEARNING_BOT_RULE_EXPLORE_PERCENT": converted_value = max(0.0, min(100.0, converted_value))


                    is_different = abs(converted_value - current_value) > 1e-9 if isinstance(converted_value, float) else current_value != converted_value
                    if is_different:
                        print(f"Applying config change: {key}: {current_value} -> {converted_value}")
                        temp_config[key] = converted_value; changed_keys.append(key)
                        if key in reset_all_keys: needs_full_reset = True
                        elif key in reset_round_keys: needs_round_reset = True
                except (ValueError, TypeError) as e: print(f"Warning: Invalid type/value for '{key}': '{value}'. Skipping. Err: {e}"); continue

        if changed_keys:
             current_config = temp_config; print(f"Config updated. Changed: {changed_keys}")
             if needs_full_reset: needs_round_reset = True

             # Update NN manager if relevant params changed
             nn_params = ['LEARNING_BOT_DIM', 'LEARNING_BOT_MEM_DEPTH', 'LEARNING_BOT_LR', 'LEARNING_BOT_WEIGHT_DECAY', 'LEARNING_BOT_MOMENTUM', 'LEARNING_BOT_MAX_GRAD_NORM']
             if any(k in nn_params for k in changed_keys):
                 nnm_recreated = update_neural_memory_manager_instance()
                 if not neural_memory_manager: emit('config_update_ack', {'success': False, 'message': 'Error: Failed NN manager update.'}, room=request.sid); return
                 if nnm_recreated: needs_full_reset = True; needs_round_reset = True

             # Update AV manager based on relevant params
             av_params = ['ENABLE_AV', 'ENABLE_AV_OUTPUT', 'VISUALIZATION_MODE', 'NUM_LEARNING_BOTS', 'LEARNING_BOT_DIM']
             if any(k in av_params for k in changed_keys):
                  av_state_changed = update_av_manager_instance()
                  # If AV state changed (created/stopped/recreated), force full reset
                  if av_state_changed: needs_full_reset = True; needs_round_reset = True

             emit('config_update_ack', {'success': True, 'needs_full_reset': needs_full_reset, 'needs_round_reset': needs_round_reset, 'updated_config': current_config}, room=request.sid)
             socketio.emit('config_update', current_config) # Broadcast new config
        else:
             print("No effective config changes detected.")
             emit('config_update_ack', {'success': True, 'needs_full_reset': False, 'needs_round_reset': False, 'updated_config': current_config}, room=request.sid)

    except Exception as e: print(f"Error updating config: {e}"); traceback.print_exc(); emit('config_update_ack', {'success': False, 'message': f'Server error: {e}'}, room=request.sid)

def start_simulation_if_configured():
    """Helper function to auto-start simulation if configured"""
    if current_config.get('AUTOSTART_SIMULATION', False):
        print("\n=== AUTO-STARTING SIMULATION ===")
        global simulation_running
        if not simulation_running:
            simulation_running = True
            global simulation_loop_task
            simulation_loop_task = socketio.start_background_task(simulation_loop)
            print("Simulation auto-started successfully")

# ================================================================
# --- Server Start ---
# ================================================================
if __name__ == '__main__':
    # IMPORTANT: Set start method for multiprocessing BEFORE any other mp usage
    # Especially important on MacOS and Windows for GUI/CUDA compatibility
    # 'spawn' is generally safest but might be slower. 'forkserver' is another option.
    # Default 'fork' on Linux/Mac can cause issues with CUDA and some GUI libs.
    if sys.platform == "darwin" or sys.platform == "win32":
        print(f"Platform is {sys.platform}, setting multiprocessing start method to 'spawn'.")
        mp.set_start_method('spawn', force=True)
    elif mp.get_start_method(allow_none=True) is None or mp.get_start_method(allow_none=True) == 'fork':
         # Keep fork on Linux unless CUDA issues arise, then switch to spawn/forkserver
         print("Using default 'fork' multiprocessing start method (Linux).")
         # Consider adding a check/warning if CUDA is used with 'fork'

    print("Initializing simulation state...")
    if not NEURAL_LIB_AVAILABLE: print("CRITICAL: Neural Memory Library failed to load. Cannot start."); sys.exit(1)
    if not update_neural_memory_manager_instance(): print("CRITICAL: Neural Memory Manager failed to initialize. Check PyTorch/CUDA setup. Exiting."); sys.exit(1)

    # Initial AV setup is handled within setup_simulation based on config

    if not setup_simulation(full_reset=True, new_environment=True):
        print("CRITICAL: Initial simulation setup failed. Check environment parameters. Exiting.")
        if av_manager: stop_av_system(av_manager)
        sys.exit(1)
    else: print("Initial setup successful.")
    
    # Auto-start the simulation if configured
    start_simulation_if_configured()

    port = int(os.environ.get('PORT', 5001)); host = '0.0.0.0'
    print(f"Attempting to start server on http://{host}:{port}")
    print(f"Web visualizer placeholder available at http://{host}:{port}/visuals")
    try:
        print("Starting Flask-SocketIO server with eventlet...")
        # socketio.run(app, host=host, port=port, debug=False, use_reloader=False)
        # Use eventlet.wsgi.server directly for potentially better control/compatibility
        import eventlet.wsgi
        eventlet.wsgi.server(eventlet.listen((host, port)), app)

    except OSError as e:
         if "Address already in use" in str(e): print(f"Error: Port {port} is already in use.")
         else: print(f"Error: Failed to start server OS error: {e}"); traceback.print_exc()
    except Exception as e: print(f"Error: Unexpected error during server startup: {e}"); traceback.print_exc()
    finally:
        print("Server shutting down...");
        if simulation_running:
            simulation_running = False
            if simulation_loop_task:
                try: simulation_loop_task.kill() # Attempt to kill greenlet
                except: pass
        if av_manager:
            stop_av_system(av_manager)
        print("Server stopped.")
