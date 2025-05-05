# Filename: audiovisualization.py
# coding: utf-8

import numpy as np
import torch
import threading
import multiprocessing as mp # Use multiprocessing for GUI isolation
import time
import queue
import traceback
import math
from scipy.signal import chirp # For more interesting sounds
import sys # For checking platform

print("Audiovisualization Library Loading...")

# --- Attempt to import optional libraries ---
VISPY_AVAILABLE = False
SOUNDDEVICE_AVAILABLE = False
SKLEARN_AVAILABLE = False # For PCA/t-SNE
MATPLOTLIB_AVAILABLE = False # For Matplotlib alternative

try:
    # VisPy needs to be imported *before* checking platform sometimes
    import vispy.app
    import vispy.scene
    from vispy.scene import visuals
    VISPY_AVAILABLE = True
    print("VisPy library imported successfully.")
except ImportError:
    print("Warning: VisPy library not found. VisPy visualization disabled. (Install with: pip install vispy)")
except Exception as e:
    print(f"Warning: VisPy library failed to import ({e}). VisPy visualization disabled.")


try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    print("SoundDevice library imported successfully.")
except ImportError:
    print("Warning: SoundDevice library not found. Audio output disabled. (Install with: pip install sounddevice)")
except Exception as e:
    # SoundDevice can sometimes raise other errors on import if system libs are missing
    print(f"Warning: SoundDevice library failed to import ({e}). Audio output disabled.")


try:
    from sklearn.decomposition import PCA
    # from sklearn.manifold import TSNE # t-SNE is much slower, PCA is better for real-time
    SKLEARN_AVAILABLE = True
    print("Scikit-learn library imported successfully (for PCA).")
except ImportError:
     print("Warning: Scikit-learn library not found. Dimensionality reduction (PCA) for visuals disabled. (Install with: pip install scikit-learn)")

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib library imported successfully (for alternative visualization).")
except ImportError:
    print("Warning: Matplotlib library not found. Matplotlib visualization disabled. (Install with: pip install matplotlib)")


# --- Constants ---
SAMPLE_RATE = 44100  # Audio sample rate
AUDIO_BLOCK_SIZE = 1024 # Samples per audio callback buffer
MAX_BOTS_FOR_AUDIO = 16 # Limit simultaneous distinct audio streams for performance/clarity
VISUAL_UPDATE_RATE_HZ = 20 # Target FPS for VisPy/Matplotlib window (Lowered slightly)
AUDIO_UPDATE_INTERVAL_S = 0.05 # How often to update audio parameters
QUEUE_TIMEOUT_S = 0.1 # Timeout for getting data from queue

# --- Visualization Process Target Functions ---

def _vispy_process_target(data_queue: mp.Queue, stop_event: mp.Event, num_bots, dim, pca_available):
    """ Target function for the VisPy visualization process. """
    if not VISPY_AVAILABLE:
        print("VisPy Process: VisPy not available, exiting.")
        return

    print("VisPy Process: Starting...")
    canvas = None
    view = None
    bot_visuals = None
    bot_positions = np.zeros((num_bots, 3), dtype=np.float32) # x, y, z
    bot_colors = np.ones((num_bots, 4), dtype=np.float32) # r, g, b, a
    bot_sizes = np.full(num_bots, 10, dtype=np.float32)
    pca = None
    if pca_available and dim > 2:
        try:
            pca = PCA(n_components=2)
            print("VisPy Process: PCA initialized.")
        except Exception as e:
            print(f"VisPy Process: Warning - PCA init failed: {e}")
            pca = None
    elif dim == 2:
        print("VisPy Process: Using raw 2D vectors for positioning.")
    else:
        print("VisPy Process: PCA not available or dim <= 2. Using basic positioning.")

    def process_data_for_visuals(av_data_batch):
        """ Updates internal visual state (positions, colors, sizes) from batch data. """
        nonlocal bot_positions, bot_colors, bot_sizes # Modify outer scope variables
        if not av_data_batch: return

        all_retrieved_vectors = []
        indices_in_batch = []

        # --- Collect data for PCA (if applicable) ---
        if pca or dim == 2:
            for av_idx, data in av_data_batch.items():
                if 0 <= av_idx < num_bots and data.get('retrieved_memory_vector') is not None:
                    # Data should already be numpy array from the main process
                    vec = data['retrieved_memory_vector'] # Expecting numpy array
                    if isinstance(vec, np.ndarray) and vec.shape == (dim,):
                        all_retrieved_vectors.append(vec)
                        indices_in_batch.append(av_idx)

            # --- Apply PCA or use raw vectors ---
            if indices_in_batch:
                vectors_np = np.array(all_retrieved_vectors)
                transformed_positions = np.zeros((len(vectors_np), 2)) # Default
                if pca:
                    try:
                        if len(vectors_np) >= 2: # Need at least 2 samples for PCA
                             transformed_positions = pca.fit_transform(vectors_np)
                        # else: Keep zeros
                    except Exception as pca_err:
                         print(f"VisPy Process: Warning - PCA failed: {pca_err}. Using zeros.")
                         # Keep zeros
                elif dim == 2:
                     transformed_positions = vectors_np

                # --- Update bot positions based on PCA/raw vectors ---
                for i, av_idx in enumerate(indices_in_batch):
                    # Scale positions for better visualization range (e.g., -5 to 5)
                    bot_positions[av_idx, 0] = transformed_positions[i, 0] * 5
                    bot_positions[av_idx, 1] = transformed_positions[i, 1] * 5
                    anomaly = data.get('anomaly_score', 0.0) # Expecting float
                    bot_positions[av_idx, 2] = np.clip(anomaly * 10, -5, 5) # Z based on anomaly

        # --- Update Colors and Sizes for all bots in the batch ---
        for av_idx, data in av_data_batch.items():
            if 0 <= av_idx < num_bots:
                anomaly = data.get('anomaly_score', 0.0)
                weight_change = data.get('weight_change_metric', 0.0)
                mode_code = data.get('mode_code', 5)
                is_player = data.get('is_player_controlled', False)

                # Color based on mode, player status, and anomaly
                alpha = np.clip(0.5 + anomaly * 5.0, 0.5, 1.0)
                color = [0.5, 0.5, 0.5, alpha * 0.8] # Default Grey
                if is_player: color = [0.1, np.clip(0.5 + anomaly * 5, 0.5, 1.0), 0.1, alpha] # Greenish
                elif mode_code == 0: color = [np.clip(0.5 + anomaly * 5, 0.5, 1.0), 0.1, 0.1, alpha] # Exploit (Reddish)
                elif mode_code == 1 or mode_code == 2: color = [0.1, 0.1, np.clip(0.5 + anomaly * 5, 0.5, 1.0), alpha] # Explore (Bluish)
                elif mode_code == 3 or mode_code == 4: color = [np.clip(0.5 + anomaly * 5, 0.5, 1.0), np.clip(0.5 + anomaly * 5, 0.5, 1.0), 0.1, alpha] # Player Target/Direct (Yellowish)

                bot_colors[av_idx] = color

                # Size based on weight change
                base_size = 8
                size_boost = np.clip(weight_change * 50.0, 0, 15)
                bot_sizes[av_idx] = base_size + size_boost

        # If no PCA/2D, use a default layout
        if not pca and dim != 2:
             sqrt_num = math.ceil(math.sqrt(num_bots))
             spacing = 10.0 / (sqrt_num + 1)
             for av_idx, data in av_data_batch.items():
                  if 0 <= av_idx < num_bots:
                       row = av_idx // sqrt_num
                       col = av_idx % sqrt_num
                       bot_positions[av_idx, 0] = (col - sqrt_num / 2 + 0.5) * spacing
                       bot_positions[av_idx, 1] = (row - sqrt_num / 2 + 0.5) * spacing
                       anomaly = data.get('anomaly_score', 0.0)
                       bot_positions[av_idx, 2] = np.clip(anomaly * 10, -5, 5)

    def update_vispy_callback(event):
        """ Called periodically by VisPy timer to update visuals from queue data. """
        nonlocal bot_visuals # Access outer scope
        try:
            latest_data = None
            # Drain the queue, keeping only the most recent batch
            while not data_queue.empty():
                try:
                    latest_data = data_queue.get_nowait()
                except queue.Empty:
                    break
                except (EOFError, BrokenPipeError):
                    print("VisPy Process: Data queue broken, stopping.")
                    vispy.app.quit()
                    return

            if latest_data:
                process_data_for_visuals(latest_data)

            # Update VisPy visual elements if they exist
            if bot_visuals:
                bot_visuals.set_data(pos=bot_positions, size=bot_sizes, face_color=bot_colors, edge_color=None)
                # canvas.update() # Might not be needed

            # Check if the main process requested stop
            if stop_event.is_set():
                print("VisPy Process: Stop event received, quitting.")
                vispy.app.quit()

        except Exception as e:
            print(f"VisPy Process: Error during VisPy update: {e}")
            # traceback.print_exc() # Can be noisy

    try:
        # Create canvas and view *inside the process*
        canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, title=f'Bot Memory Visualization ({num_bots} bots)')
        view = canvas.central_widget.add_view()
        view.camera = 'turntable' # Use a 3D camera

        # Create Markers visual for bots
        bot_visuals = visuals.Markers(parent=view.scene)
        bot_visuals.set_data(pos=bot_positions, size=bot_sizes, face_color=bot_colors, edge_color=None)

        # Add axes for context
        visuals.XYZAxis(parent=view.scene)

        # Setup a timer to process data from the queue
        timer = vispy.app.Timer(interval=1.0/VISUAL_UPDATE_RATE_HZ, connect=update_vispy_callback, start=True)

        print("VisPy Process: Setup complete. Running event loop...")
        vispy.app.run() # Start the event loop - THIS WILL BLOCK THIS PROCESS

    except Exception as e:
        print(f"VisPy Process: CRITICAL ERROR: {e}")
        traceback.print_exc()
    finally:
        if canvas:
            try:
                canvas.close()
            except Exception as ce:
                print(f"VisPy Process: Error closing canvas: {ce}")
        print("VisPy Process: Stopped.")

# --- Matplotlib Process Target (Placeholder/Example) ---
def _matplotlib_process_target(data_queue: mp.Queue, stop_event: mp.Event, num_bots, dim, pca_available):
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib Process: Matplotlib not available, exiting.")
        return

    print("Matplotlib Process: Starting...")
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    scatter = None # Store the scatter plot object

    bot_positions = np.zeros((num_bots, 3), dtype=np.float32)
    bot_colors = np.array([[0.5, 0.5, 0.5, 0.8]] * num_bots, dtype=np.float32) # Default Grey
    bot_sizes = np.full(num_bots, 30, dtype=np.float32) # Matplotlib sizes are different scale
    pca = None
    if pca_available and dim > 2: pca = PCA(n_components=2)

    def process_data_for_matplotlib(av_data_batch):
        # Similar logic to VisPy's process_data_for_visuals, but update numpy arrays used by Matplotlib
        nonlocal bot_positions, bot_colors, bot_sizes
        # ... (Adapt the data processing logic from VisPy's function here) ...
        # Ensure the arrays bot_positions, bot_colors, bot_sizes are updated
        # Example snippet (needs full adaptation):
        indices_in_batch = list(av_data_batch.keys())
        if not indices_in_batch: return

        # --- PCA/Positioning (Simplified Example) ---
        if pca or dim == 2:
            vectors = [d.get('retrieved_memory_vector') for d in av_data_batch.values() if d.get('retrieved_memory_vector') is not None]
            if vectors:
                vectors_np = np.array(vectors)
                transformed = np.zeros((len(vectors_np), 2))
                if pca and len(vectors_np) >= 2: transformed = pca.fit_transform(vectors_np)
                elif dim == 2: transformed = vectors_np
                for i, av_idx in enumerate(indices_in_batch):
                     if 0 <= av_idx < num_bots:
                         bot_positions[av_idx, 0] = transformed[i, 0] * 5
                         bot_positions[av_idx, 1] = transformed[i, 1] * 5
        # --- Color/Size (Simplified Example) ---
        for av_idx, data in av_data_batch.items():
             if 0 <= av_idx < num_bots:
                 anomaly = data.get('anomaly_score', 0.0)
                 weight_change = data.get('weight_change_metric', 0.0)
                 # ... (set color based on mode/player) ...
                 bot_colors[av_idx] = [np.clip(0.5 + anomaly * 5, 0.5, 1.0), 0.1, 0.1, np.clip(0.5 + anomaly * 5.0, 0.5, 1.0)] # Example: Reddish based on anomaly
                 bot_sizes[av_idx] = 20 + np.clip(weight_change * 100.0, 0, 60)
                 bot_positions[av_idx, 2] = np.clip(anomaly * 10, -5, 5) # Z based on anomaly

    def update_plot(frame):
        nonlocal scatter # Modify outer scope
        try:
            latest_data = None
            while not data_queue.empty():
                try:
                    latest_data = data_queue.get_nowait()
                except queue.Empty: break
                except (EOFError, BrokenPipeError): print("Matplotlib Process: Queue broken."); plt.close(fig); return None

            if latest_data:
                process_data_for_matplotlib(latest_data)

            if scatter is None: # First frame
                # Initial plot setup
                scatter = ax.scatter(bot_positions[:, 0], bot_positions[:, 1], bot_positions[:, 2],
                                     c=bot_colors, s=bot_sizes, depthshade=True)
                ax.set_xlabel("PCA 1 / X")
                ax.set_ylabel("PCA 2 / Y")
                ax.set_zlabel("Anomaly / Z")
                ax.set_title("Bot Memory State (Matplotlib)")
                ax.set_xlim(-6, 6)
                ax.set_ylim(-6, 6)
                ax.set_zlim(-6, 6)
            else:
                # Update existing scatter plot data (more efficient)
                # Note: Updating colors and sizes might require recreating scatter or using specific methods
                scatter._offsets3d = (bot_positions[:, 0], bot_positions[:, 1], bot_positions[:, 2])
                # scatter.set_sizes(bot_sizes) # Update sizes
                # scatter.set_facecolor(bot_colors) # Update colors (facecolor)
                # scatter.set_edgecolor(None) # Optional: remove edge colors

            if stop_event.is_set():
                print("Matplotlib Process: Stop event received, closing plot.")
                plt.close(fig)
                return None # Stop animation

            return scatter, # Return tuple of artists to update

        except Exception as e:
            print(f"Matplotlib Process: Error during update: {e}")
            # traceback.print_exc()
            return scatter,

    try:
        # Use FuncAnimation for periodic updates
        ani = animation.FuncAnimation(fig, update_plot,
                                      interval=1000.0/VISUAL_UPDATE_RATE_HZ, # Interval in ms
                                      blit=False, # Blitting can be complex with 3D/changing colors/sizes
                                      cache_frame_data=False)
        plt.show(block=True) # block=True runs the event loop

    except Exception as e:
        print(f"Matplotlib Process: CRITICAL ERROR: {e}")
        traceback.print_exc()
    finally:
        print("Matplotlib Process: Stopped.")


# --- AVManager Class ---
class AVManager:
    def __init__(self, num_bots, dim, device, enable_output, visualization_mode='vispy'):
        self.num_bots = num_bots
        self.dim = dim
        self.device = torch.device(device) # Ensure it's a device object
        self.visualization_mode = visualization_mode if enable_output else 'none'
        self.enable_audio_output = enable_output and SOUNDDEVICE_AVAILABLE
        self.enable_visual_output = enable_output and (
            (visualization_mode == 'vispy' and VISPY_AVAILABLE) or
            (visualization_mode == 'matplotlib' and MATPLOTLIB_AVAILABLE) or
            (visualization_mode == 'web') # Web mode doesn't need specific backend libs here
        )

        self.is_setup = False
        self.audio_thread = None # SoundDevice can run in a thread
        self.visual_process = None # VisPy/Matplotlib run in separate process
        self.visual_data_queue = None # Queue for sending data to visual process
        self.stop_event = mp.Event() if self.enable_visual_output else threading.Event() # Use mp.Event for processes

        # --- Visual State ---
        self.pca = None
        self.pca_available = SKLEARN_AVAILABLE
        if self.pca_available and self.dim > 2:
            try:
                # Initialize PCA here for potential use in data prep
                self.pca = PCA(n_components=2)
                print("AVManager: PCA initialized for data preparation.")
            except Exception as e:
                print(f"AVManager: Warning - PCA init failed: {e}")
                self.pca = None
                self.pca_available = False # Mark as unavailable if init fails
        elif self.dim == 2:
             print("AVManager: Using raw 2D vectors for visual positioning.")
        else:
             print("AVManager: PCA not available or dim <= 2. Using basic positioning.")


        # --- Audio State ---
        self.audio_stream = None
        self.audio_params = {} # bot_av_idx -> {freq, amp, decay, phase, waveform}
        self.last_audio_update_time = 0
        self.active_audio_indices = set() # Track which bots are making sound
        self.last_audio_frame = 0 # Moved here

        # --- Web Visualization State ---
        self.socketio_instance = None # To be set if using web mode
        self.socketio_namespace = '/visuals' # Example namespace

        print(f"AVManager initialized: Bots={num_bots}, Dim={dim}, Device={device}, Audio={self.enable_audio_output}, Visuals={self.visualization_mode if self.enable_visual_output else 'none'}")

    def set_socketio_instance(self, sio):
        """ Set the SocketIO instance if using web visualization mode. """
        if self.visualization_mode == 'web':
            self.socketio_instance = sio
            print("AVManager: SocketIO instance set for web visualization.")

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """ SoundDevice callback to generate audio. (Runs in SoundDevice's thread) """
        if status: print(f"Audio Callback Status: {status}", flush=True)
        try:
            t = (self.last_audio_frame + np.arange(frames)) / SAMPLE_RATE
            outdata.fill(0)

            # Use a copy of keys/indices to avoid issues if the main thread modifies dict/set
            active_indices_copy = list(self.active_audio_indices)
            params_copy = self.audio_params.copy()

            num_active_audio = 0
            for av_idx in active_indices_copy:
                 if av_idx in params_copy:
                    params = params_copy[av_idx]
                    freq = params['freq']
                    amp = params['amp'] * params['decay']
                    phase = params['phase']
                    waveform_type = params.get('waveform', 'sine')

                    # Generate waveform
                    signal = np.zeros(frames, dtype=np.float32)
                    if waveform_type == 'sine': signal = np.sin(2 * np.pi * freq * t + phase)
                    elif waveform_type == 'square': signal = np.sign(np.sin(2 * np.pi * freq * t + phase))
                    elif waveform_type == 'chirp':
                         chirp_end_freq = freq * 1.2
                         # Need to calculate chirp phase correctly across blocks - simplified here
                         signal = chirp(t, f0=freq, f1=chirp_end_freq, t1=frames/SAMPLE_RATE, method='linear', phi=phase * 180 / np.pi) # Approx phase
                    else: signal = np.sin(2 * np.pi * freq * t + phase)

                    outdata[:, 0] += (signal * amp).astype(np.float32)
                    num_active_audio += 1

                    # Update state for the *original* dict for next block (needs thread safety if accessed elsewhere, but updates are infrequent)
                    # This direct update might be okay as _process_data_for_audio runs less often
                    if av_idx in self.audio_params:
                        self.audio_params[av_idx]['decay'] *= 0.99
                        new_phase = (phase + 2 * np.pi * freq * (frames / SAMPLE_RATE)) % (2 * np.pi)
                        self.audio_params[av_idx]['phase'] = new_phase
                        if self.audio_params[av_idx]['decay'] < 0.01:
                            # Schedule removal in the main thread? Or handle potential key errors in _process_data_for_audio
                            # For simplicity, let's try removing directly here, but be aware of potential race conditions
                            self.active_audio_indices.discard(av_idx)
                            # del self.audio_params[av_idx] # Safer to let _process_data handle cleanup

            if num_active_audio > 0:
                 max_val = np.max(np.abs(outdata))
                 if max_val > 1.0: outdata /= max_val

            outdata[:] = np.ascontiguousarray(outdata, dtype=np.float32)
            self.last_audio_frame += frames

        except Exception as e:
            print(f"Error in audio callback: {e}")
            outdata.fill(0)

    def _process_data_for_audio(self, av_data_batch):
        """ Updates internal audio parameters based on batch data. (Runs in main server thread/loop) """
        if not self.enable_audio_output: return

        now = time.monotonic()
        if now - self.last_audio_update_time < AUDIO_UPDATE_INTERVAL_S: return
        self.last_audio_update_time = now

        num_can_add = MAX_BOTS_FOR_AUDIO - len(self.active_audio_indices)
        added_count = 0

        # Clean up sounds that have decayed in the callback (check if key still exists)
        indices_to_remove = {idx for idx in self.active_audio_indices if idx not in self.audio_params or self.audio_params[idx]['decay'] < 0.01}
        for idx in indices_to_remove:
            self.active_audio_indices.discard(idx)
            if idx in self.audio_params:
                del self.audio_params[idx]

        # Sort bots by anomaly score
        sorted_indices = sorted(av_data_batch.keys(), key=lambda idx: av_data_batch[idx].get('anomaly_score', torch.tensor(0.0)).item(), reverse=True)

        for av_idx in sorted_indices:
            if 0 <= av_idx < self.num_bots:
                # Data comes in as tensors, convert to CPU float/bool
                data = av_data_batch[av_idx]
                anomaly = data.get('anomaly_score', torch.tensor(0.0)).item()
                weight_change = data.get('weight_change_metric', torch.tensor(0.0)).item()
                mode_code = data.get('mode_code', 5) # Already int
                is_player = data.get('is_player_controlled', False) # Already bool

                should_trigger = anomaly > 0.1 or weight_change > 0.05

                if av_idx in self.active_audio_indices and av_idx in self.audio_params:
                     # Update existing sound parameters
                     params = self.audio_params[av_idx]
                     params['freq'] = 220 + np.clip(anomaly * 2000, 0, 2000)
                     params['amp'] = np.clip(0.1 + weight_change * 5, 0.1, 0.8)
                     params['decay'] = 1.0 # Reset decay
                     if is_player: params['waveform'] = 'square'
                     elif mode_code == 0: params['waveform'] = 'sine'
                     elif mode_code == 1 or mode_code == 2: params['waveform'] = 'chirp'
                     else: params['waveform'] = 'sine'

                elif should_trigger and added_count < num_can_add:
                     # Add new sound
                     new_params = {
                         'freq': 220 + np.clip(anomaly * 2000, 0, 2000),
                         'amp': np.clip(0.1 + weight_change * 5, 0.1, 0.8),
                         'decay': 1.0,
                         'phase': np.random.rand() * 2 * np.pi,
                         'waveform': 'sine'
                     }
                     if is_player: new_params['waveform'] = 'square'
                     elif mode_code == 0: new_params['waveform'] = 'sine'
                     elif mode_code == 1 or mode_code == 2: new_params['waveform'] = 'chirp'

                     self.audio_params[av_idx] = new_params
                     self.active_audio_indices.add(av_idx)
                     added_count += 1

    def _prepare_data_for_visual_queue(self, av_data_batch):
        """ Converts tensor data to numpy/primitives suitable for inter-process queue. """
        prepared_batch = {}
        for av_idx, data in av_data_batch.items():
            prep_data = {}
            # Convert tensors to numpy arrays on CPU
            retrieved = data.get('retrieved_memory_vector')
            if retrieved is not None:
                prep_data['retrieved_memory_vector'] = retrieved.squeeze().detach().cpu().numpy()

            anomaly = data.get('anomaly_score')
            if anomaly is not None:
                prep_data['anomaly_score'] = anomaly.item() # Float

            weight_change = data.get('weight_change_metric')
            if weight_change is not None:
                prep_data['weight_change_metric'] = weight_change.item() # Float

            # Copy other primitive types
            prep_data['mode_code'] = data.get('mode_code', 5)
            prep_data['is_player_controlled'] = data.get('is_player_controlled', False)
            # Add input stream vector if needed by visualizer (optional)
            # input_vec = data.get('input_stream_vector')
            # if input_vec is not None:
            #     prep_data['input_stream_vector'] = input_vec.squeeze().detach().cpu().numpy()

            prepared_batch[av_idx] = prep_data
        return prepared_batch

    def _prepare_data_for_web(self, av_data_batch):
        """ Converts data to JSON-serializable format for SocketIO emission. """
        json_batch = {}
        # Similar to _prepare_data_for_visual_queue, but maybe less data needed
        # Example: Only send position, color, size components
        # This part depends heavily on the specific web visualization needs
        # For now, let's reuse the queue prep logic
        prepared_data = self._prepare_data_for_visual_queue(av_data_batch)

        # Further convert numpy arrays to lists if needed for JSON
        for av_idx, data in prepared_data.items():
            json_data = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    json_data[key] = value.tolist() # Convert numpy arrays to lists
                else:
                    json_data[key] = value # Keep primitives
            json_batch[av_idx] = json_data
        return json_batch


    def setup(self):
        """ Initializes and starts the output systems based on mode. """
        if not self.enable_audio_output and not self.enable_visual_output:
            print("AVManager: All outputs disabled, setup skipped.")
            self.is_setup = False
            return

        print(f"AVManager: Setting up outputs (Audio: {self.enable_audio_output}, Visual: {self.visualization_mode})...")
        try:
            # --- Start Audio Thread ---
            if self.enable_audio_output:
                print(f"AVManager: Setting up SoundDevice stream (Rate: {SAMPLE_RATE}, Blocksize: {AUDIO_BLOCK_SIZE})...")
                self.last_audio_frame = 0
                self.audio_stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=AUDIO_BLOCK_SIZE,
                    channels=1, # Mono output
                    callback=self._audio_callback,
                    dtype='float32'
                )
                # Start stream in a separate thread to avoid potential blocking issues?
                # sd.OutputStream usually manages its own thread via the callback.
                self.audio_stream.start()
                print("AVManager: SoundDevice stream started.")
            else: print("AVManager: Audio output disabled.")

            # --- Start Visualization Process ---
            if self.enable_visual_output:
                if self.visualization_mode == 'vispy' and VISPY_AVAILABLE:
                    print("AVManager: Starting VisPy visualization process...")
                    self.visual_data_queue = mp.Queue()
                    self.visual_process = mp.Process(
                        target=_vispy_process_target,
                        args=(self.visual_data_queue, self.stop_event, self.num_bots, self.dim, self.pca_available),
                        daemon=True # Make daemon so it exits if main process exits
                    )
                    self.visual_process.start()
                elif self.visualization_mode == 'matplotlib' and MATPLOTLIB_AVAILABLE:
                    print("AVManager: Starting Matplotlib visualization process...")
                    self.visual_data_queue = mp.Queue()
                    self.visual_process = mp.Process(
                        target=_matplotlib_process_target,
                         args=(self.visual_data_queue, self.stop_event, self.num_bots, self.dim, self.pca_available),
                        daemon=True
                    )
                    self.visual_process.start()
                elif self.visualization_mode == 'web':
                    print("AVManager: Web visualization mode enabled. Waiting for SocketIO instance.")
                    # Setup happens when set_socketio_instance is called
                else:
                    print(f"AVManager: Visual output mode '{self.visualization_mode}' disabled or library not available.")
                    self.enable_visual_output = False # Ensure flag is false if not started

            self.is_setup = True
            print("AVManager: Setup complete.")

        except Exception as e:
            print(f"AVManager: ERROR during setup: {e}")
            traceback.print_exc()
            self.is_setup = False
            self.stop() # Attempt cleanup

    def update(self, av_data_batch: dict):
        """ Receives data from the server and processes/queues it. """
        if not self.is_setup or not av_data_batch:
            return

        # --- Process data directly for audio ---
        if self.enable_audio_output:
             try: self._process_data_for_audio(av_data_batch)
             except Exception as e: print(f"Error processing audio data: {e}")

        # --- Queue/Emit data for Visuals ---
        if self.enable_visual_output:
            if self.visualization_mode in ['vispy', 'matplotlib'] and self.visual_process and self.visual_process.is_alive():
                if self.visual_data_queue:
                    try:
                        # Prepare data (convert tensors to numpy/primitives)
                        prepared_data = self._prepare_data_for_visual_queue(av_data_batch)
                        # Put data in the queue for the visual process
                        self.visual_data_queue.put(prepared_data, block=False)
                    except queue.Full:
                        # print("Warning: Visual data queue full, dropping frame.") # Can be noisy
                        pass
                    except Exception as e:
                        print(f"Error preparing/queuing visual data: {e}")
            elif self.visualization_mode == 'web' and self.socketio_instance:
                 try:
                     # Prepare data for JSON serialization
                     json_data = self._prepare_data_for_web(av_data_batch)
                     # Emit data over SocketIO
                     self.socketio_instance.emit('visual_update', json_data, namespace=self.socketio_namespace)
                 except Exception as e:
                     print(f"Error preparing/emitting web visual data: {e}")


    def stop(self):
        """ Stops output threads/processes and cleans up resources. """
        print("AVManager: Stopping...")
        self.stop_event.set() # Signal threads/processes to stop

        # --- Stop Audio ---
        if self.audio_stream:
            try:
                print("AVManager: Stopping SoundDevice stream...")
                self.audio_stream.stop()
                self.audio_stream.close(ignore_errors=True)
                print("AVManager: SoundDevice stream stopped and closed.")
            except Exception as e:
                print(f"AVManager: Error stopping audio stream: {e}")
            self.audio_stream = None

        # --- Stop Visualization Process ---
        if self.visual_process and self.visual_process.is_alive():
             print(f"AVManager: Waiting for {self.visualization_mode} process to finish...")
             # Process should detect stop_event and exit gracefully
             self.visual_process.join(timeout=5.0) # Wait for process to exit
             if self.visual_process.is_alive():
                  print(f"AVManager: Warning: {self.visualization_mode} process did not exit gracefully. Terminating.")
                  self.visual_process.terminate() # Force terminate if needed
                  self.visual_process.join(timeout=1.0) # Wait after terminate
             else:
                  print(f"AVManager: {self.visualization_mode} process joined.")
        if self.visual_data_queue:
            self.visual_data_queue.close()
            self.visual_data_queue.join_thread() # Ensure queue feeder threads are finished

        self.visual_process = None
        self.visual_data_queue = None
        self.is_setup = False
        print("AVManager: Stop sequence complete.")

# --- Global Functions for Server Interaction ---

# Keep track of the manager instance globally within this module
_current_av_manager = None

def setup_av_system(num_bots, dim, device, enable_output, visualization_mode, socketio_instance=None):
    """ Creates, sets up, and returns the AVManager instance. """
    global _current_av_manager
    if _current_av_manager:
        print("Warning: Existing AVManager found during setup. Stopping previous one.")
        stop_av_system(_current_av_manager)
        _current_av_manager = None

    # Check library availability based on mode
    libs_ok = True
    if visualization_mode == 'vispy' and not VISPY_AVAILABLE: libs_ok = False
    if visualization_mode == 'matplotlib' and not MATPLOTLIB_AVAILABLE: libs_ok = False
    if not SOUNDDEVICE_AVAILABLE and enable_output: # Audio is independent of visual mode
        print("Warning: SoundDevice not available, audio output will be disabled.")
        # Don't set libs_ok to False, just disable audio

    if not libs_ok and enable_output:
         print(f"AV Setup Skipped: Required library for mode '{visualization_mode}' is not available.")
         return None

    try:
        print(f"Setting up AV System: Mode={visualization_mode}, Bots={num_bots}, Dim={dim}, Device={device}, Output={enable_output}")
        manager = AVManager(num_bots, dim, device, enable_output, visualization_mode)

        # Pass SocketIO instance if using web mode
        if visualization_mode == 'web' and socketio_instance:
            manager.set_socketio_instance(socketio_instance)

        manager.setup() # Initialize VisPy/SoundDevice/etc.

        if manager.is_setup or not enable_output: # Consider setup successful if output was disabled
             _current_av_manager = manager
             print("AV System setup successful.")
             return manager
        else:
             print("AV System setup failed.")
             manager.stop() # Ensure cleanup even if setup failed
             return None
    except Exception as e:
        print(f"FATAL ERROR during AV setup: {e}")
        traceback.print_exc()
        if _current_av_manager: # Cleanup if partially created
             _current_av_manager.stop()
             _current_av_manager = None
        return None

def update_av_system(manager: AVManager, av_data_batch: dict):
    """ Updates the AV system with new data. """
    # Check manager exists, is setup, AND has at least one output enabled
    if manager and manager.is_setup and (manager.enable_audio_output or manager.enable_visual_output):
        try:
            manager.update(av_data_batch)
        except Exception as e:
            print(f"Error during AV update: {e}")
            # traceback.print_exc() # Can be noisy

def stop_av_system(manager: AVManager):
    """ Stops the AV system and cleans up resources. """
    global _current_av_manager
    if manager:
        print("Stopping AV System...")
        try:
            manager.stop()
            print("AV System stopped.")
        except Exception as e:
            print(f"Error during AV stop: {e}")
            traceback.print_exc()
    if manager is _current_av_manager:
         _current_av_manager = None # Clear global ref if it's the current one

print("Audiovisualization Library Loaded.")
# Add a check for MacOS backend issue with multiprocessing and VisPy/Matplotlib
if sys.platform == "darwin":
    try:
        # Check if the backend is 'tk' which often causes issues on MacOS with multiprocessing
        # This might need adjustment based on the actual backend used by VisPy/Matplotlib
        # vispy.use(app='glfw') # Example: Try forcing GLFW backend for VisPy if available
        # import matplotlib
        # current_backend = matplotlib.get_backend()
        # if 'tk' in current_backend.lower():
        #      print("\nWARNING: Detected 'TkAgg' Matplotlib backend on MacOS.")
        #      print("Visualization via Matplotlib in a separate process might be unstable or require specific setup.")
        #      print("Consider using 'vispy' mode or 'web' mode if issues occur.\n")
        # Forcing backend might need to happen earlier or via environment variables
        pass # Keep check simple for now
    except Exception as e:
        print(f"Note: Could not check visualization backend details ({e})")
