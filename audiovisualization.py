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
        # Temporary storage for data associated with vectors for PCA
        data_for_indices_in_batch = {}


        # --- Collect data for PCA (if applicable) ---
        if pca or dim == 2:
            for av_idx, data in av_data_batch.items():
                if 0 <= av_idx < num_bots and data.get('retrieved_memory_vector') is not None:
                    # Data should already be numpy array from the main process
                    vec = data['retrieved_memory_vector'] # Expecting numpy array
                    if isinstance(vec, np.ndarray) and vec.shape == (dim,):
                        all_retrieved_vectors.append(vec)
                        indices_in_batch.append(av_idx)
                        data_for_indices_in_batch[av_idx] = data # Store full data for this index

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
                    # Use the stored data for this index to get anomaly
                    anomaly = data_for_indices_in_batch[av_idx].get('anomaly_score', 0.0)
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
                elif mode_code == 5: color = [0.5, 0.5, 0.5, alpha * 0.7] # Frozen/Idle
                elif mode_code == 6: color = [0.6, 0.2, 0.8, alpha] # Hallucinating (Purplish)


                bot_colors[av_idx] = color

                # Size based on weight change
                base_size = 8
                size_boost = np.clip(weight_change * 50.0, 0, 15)
                bot_sizes[av_idx] = base_size + size_boost

        # If no PCA/2D, use a default layout (ensure this part also uses the full data from av_data_batch)
        if not pca and dim != 2:
             sqrt_num = math.ceil(math.sqrt(num_bots)) if num_bots > 0 else 1
             spacing = 10.0 / (sqrt_num + 1) if sqrt_num > 0 else 10.0
             for av_idx_from_batch, data_from_batch in av_data_batch.items(): # Iterate over the received batch
                  if 0 <= av_idx_from_batch < num_bots:
                       row = av_idx_from_batch // sqrt_num
                       col = av_idx_from_batch % sqrt_num
                       bot_positions[av_idx_from_batch, 0] = (col - sqrt_num / 2 + 0.5) * spacing
                       bot_positions[av_idx_from_batch, 1] = (row - sqrt_num / 2 + 0.5) * spacing
                       anomaly_val = data_from_batch.get('anomaly_score', 0.0) # Get anomaly from this bot's data
                       bot_positions[av_idx_from_batch, 2] = np.clip(anomaly_val * 10, -5, 5)


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
                    if vispy and vispy.app: vispy.app.quit()
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
                if vispy and vispy.app: vispy.app.quit()


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
        # Set initial data for markers (can be all zeros/defaults)
        bot_visuals.set_data(pos=bot_positions, size=bot_sizes, face_color=bot_colors, edge_color=None)


        # Add axes for context
        visuals.XYZAxis(parent=view.scene)

        # Setup a timer to process data from the queue
        timer = vispy.app.Timer(interval=1.0/VISUAL_UPDATE_RATE_HZ, connect=update_vispy_callback, start=True)

        print("VisPy Process: Setup complete. Running event loop...")
        if vispy and vispy.app: vispy.app.run() # Start the event loop - THIS WILL BLOCK THIS PROCESS


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
        nonlocal bot_positions, bot_colors, bot_sizes
        if not av_data_batch: return

        all_retrieved_vectors = []
        indices_in_batch = []
        data_for_indices_in_batch = {}

        if pca or dim == 2:
            for av_idx, data in av_data_batch.items():
                if 0 <= av_idx < num_bots and data.get('retrieved_memory_vector') is not None:
                    vec = data['retrieved_memory_vector']
                    if isinstance(vec, np.ndarray) and vec.shape == (dim,):
                        all_retrieved_vectors.append(vec)
                        indices_in_batch.append(av_idx)
                        data_for_indices_in_batch[av_idx] = data


            if indices_in_batch:
                vectors_np = np.array(all_retrieved_vectors)
                transformed_positions = np.zeros((len(vectors_np), 2))
                if pca and len(vectors_np) >= 2:
                    transformed_positions = pca.fit_transform(vectors_np)
                elif dim == 2:
                    transformed_positions = vectors_np

                for i, av_idx in enumerate(indices_in_batch):
                    bot_positions[av_idx, 0] = transformed_positions[i, 0] * 5
                    bot_positions[av_idx, 1] = transformed_positions[i, 1] * 5
                    anomaly = data_for_indices_in_batch[av_idx].get('anomaly_score', 0.0)
                    bot_positions[av_idx, 2] = np.clip(anomaly * 10, -5, 5)

        for av_idx, data in av_data_batch.items():
             if 0 <= av_idx < num_bots:
                 anomaly = data.get('anomaly_score', 0.0)
                 weight_change = data.get('weight_change_metric', 0.0)
                 mode_code = data.get('mode_code', 5)
                 is_player = data.get('is_player_controlled', False)
                 
                 alpha = np.clip(0.5 + anomaly * 5.0, 0.5, 1.0)
                 color = [0.5,0.5,0.5, alpha*0.8]
                 if is_player: color = [0.1, np.clip(0.5+anomaly*5,0.5,1.0), 0.1, alpha]
                 elif mode_code == 0: color = [np.clip(0.5+anomaly*5,0.5,1.0),0.1,0.1,alpha]
                 elif mode_code == 1 or mode_code == 2: color = [0.1,0.1,np.clip(0.5+anomaly*5,0.5,1.0),alpha]
                 elif mode_code == 3 or mode_code == 4: color = [np.clip(0.5+anomaly*5,0.5,1.0),np.clip(0.5+anomaly*5,0.5,1.0),0.1,alpha]
                 elif mode_code == 5: color = [0.5,0.5,0.5, alpha*0.7]
                 elif mode_code == 6: color = [0.6,0.2,0.8,alpha]

                 bot_colors[av_idx] = color
                 bot_sizes[av_idx] = 20 + np.clip(weight_change * 100.0, 0, 60)
                 # If not using PCA for X,Y, set Z based on anomaly (already done if PCA path taken)
                 if not (pca or dim == 2): # If positions were set by grid layout
                    bot_positions[av_idx, 2] = np.clip(anomaly * 10, -5, 5)


        if not pca and dim != 2:
             sqrt_num = math.ceil(math.sqrt(num_bots)) if num_bots > 0 else 1
             spacing = 10.0 / (sqrt_num + 1) if sqrt_num > 0 else 10.0
             for av_idx_from_batch, data_from_batch in av_data_batch.items():
                  if 0 <= av_idx_from_batch < num_bots:
                       row = av_idx_from_batch // sqrt_num
                       col = av_idx_from_batch % sqrt_num
                       bot_positions[av_idx_from_batch, 0] = (col - sqrt_num / 2 + 0.5) * spacing
                       bot_positions[av_idx_from_batch, 1] = (row - sqrt_num / 2 + 0.5) * spacing
                       # Z is already set by the anomaly loop above or PCA loop


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
                scatter = ax.scatter(bot_positions[:, 0], bot_positions[:, 1], bot_positions[:, 2],
                                     c=bot_colors, s=bot_sizes, depthshade=True)
                ax.set_xlabel("PCA 1 / X")
                ax.set_ylabel("PCA 2 / Y")
                ax.set_zlabel("Anomaly / Z")
                ax.set_title("Bot Memory State (Matplotlib)")
                ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_zlim(-6, 6)
            else:
                scatter._offsets3d = (bot_positions[:, 0], bot_positions[:, 1], bot_positions[:, 2])
                scatter.set_sizes(bot_sizes) 
                scatter.set_facecolors(bot_colors) 


            if stop_event.is_set():
                print("Matplotlib Process: Stop event received, closing plot.")
                plt.close(fig)
                return None 

            return scatter, 

        except Exception as e:
            print(f"Matplotlib Process: Error during update: {e}")
            return scatter,

    try:
        ani = animation.FuncAnimation(fig, update_plot,
                                      interval=1000.0/VISUAL_UPDATE_RATE_HZ, 
                                      blit=False, 
                                      cache_frame_data=False)
        plt.show(block=True) 

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
        self.stop_event = mp.Event() if self.enable_visual_output and visualization_mode in ['vispy', 'matplotlib'] else threading.Event()


        # --- Visual State ---
        self.pca = None
        self.pca_available = SKLEARN_AVAILABLE
        if self.pca_available and self.dim > 2:
            try:
                self.pca = PCA(n_components=2)
                print("AVManager: PCA initialized for data preparation.")
            except Exception as e:
                print(f"AVManager: Warning - PCA init failed: {e}")
                self.pca = None
                self.pca_available = False 
        elif self.dim == 2:
             print("AVManager: Using raw 2D vectors for visual positioning.")
        else:
             print("AVManager: PCA not available or dim <= 2. Using basic positioning.")


        # --- Audio State ---
        self.audio_stream = None
        self.audio_params = {} # bot_av_idx -> {freq, amp, decay, phase, waveform}
        self.last_audio_update_time = 0
        self.active_audio_indices = set() # Track which bots are making sound
        self.last_audio_frame = 0 

        # --- Web Visualization State ---
        self.socketio_instance = None 
        self.socketio_namespace = '/visuals' 

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

                    signal = np.zeros(frames, dtype=np.float32)
                    if waveform_type == 'sine': signal = np.sin(2 * np.pi * freq * t + phase)
                    elif waveform_type == 'square': signal = np.sign(np.sin(2 * np.pi * freq * t + phase))
                    elif waveform_type == 'chirp':
                         chirp_end_freq = freq * 1.2
                         signal = chirp(t, f0=freq, f1=chirp_end_freq, t1=frames/SAMPLE_RATE, method='linear', phi=np.degrees(phase)) # Approx phase
                    else: signal = np.sin(2 * np.pi * freq * t + phase)

                    outdata[:, 0] += (signal * amp).astype(np.float32)
                    num_active_audio += 1

                    if av_idx in self.audio_params: # Check again as it might have been removed
                        self.audio_params[av_idx]['decay'] *= 0.99
                        new_phase = (phase + 2 * np.pi * freq * (frames / SAMPLE_RATE)) % (2 * np.pi)
                        self.audio_params[av_idx]['phase'] = new_phase
                        if self.audio_params[av_idx]['decay'] < 0.01:
                            # Schedule removal more safely or let _process_data_for_audio handle it
                            # For now, just mark for removal by main thread by setting decay to very low
                            self.audio_params[av_idx]['decay'] = 0.0 


            if num_active_audio > 0:
                 max_val = np.max(np.abs(outdata))
                 if max_val > 1.0: outdata /= max_val

            outdata[:] = np.ascontiguousarray(outdata, dtype=np.float32)
            self.last_audio_frame += frames

        except Exception as e:
            print(f"Error in audio callback: {e}") # Avoid traceback in audio thread
            outdata.fill(0)

    def _process_data_for_audio(self, av_data_batch):
        """ Updates internal audio parameters based on batch data. (Runs in main server thread/loop) """
        if not self.enable_audio_output: return

        now = time.monotonic()
        if now - self.last_audio_update_time < AUDIO_UPDATE_INTERVAL_S: return
        self.last_audio_update_time = now

        # Clean up sounds that have fully decayed
        indices_to_remove_from_active = set()
        params_keys_to_delete = []

        for idx in list(self.active_audio_indices): # Iterate over a copy for safe modification
            if idx not in self.audio_params or self.audio_params[idx].get('decay', 1.0) <= 0.005 : # Check actual decay value
                indices_to_remove_from_active.add(idx)
                if idx in self.audio_params:
                    params_keys_to_delete.append(idx)
        
        for idx in indices_to_remove_from_active:
            self.active_audio_indices.discard(idx)
        for key in params_keys_to_delete:
            if key in self.audio_params:
                del self.audio_params[key]


        num_can_add = MAX_BOTS_FOR_AUDIO - len(self.active_audio_indices)
        added_count = 0

        # Sort bots by anomaly score to prioritize more "surprised" bots for audio
        sorted_indices = sorted(av_data_batch.keys(), key=lambda idx: av_data_batch[idx].get('anomaly_score', torch.tensor(0.0)).item(), reverse=True)

        for av_idx in sorted_indices:
            if 0 <= av_idx < self.num_bots:
                data = av_data_batch[av_idx]
                anomaly = data.get('anomaly_score', torch.tensor(0.0)).item()
                weight_change = data.get('weight_change_metric', torch.tensor(0.0)).item()
                mode_code = data.get('mode_code', 5) 
                is_player = data.get('is_player_controlled', False) 

                # Trigger sound if anomaly is high, or significant weight change, or player action
                should_trigger_or_update = anomaly > 0.1 or weight_change > 0.05 or (is_player and (mode_code==3 or mode_code==4))

                if av_idx in self.active_audio_indices and av_idx in self.audio_params:
                     if should_trigger_or_update: # Update existing sound parameters
                         params = self.audio_params[av_idx]
                         params['freq'] = 220 + np.clip(anomaly * 2000, 0, 2000) + (500 if is_player else 0)
                         params['amp'] = np.clip(0.1 + weight_change * 5 + (0.2 if is_player else 0), 0.05, 0.8)
                         params['decay'] = 1.0 # Reset decay
                         if is_player: params['waveform'] = 'square'
                         elif mode_code == 0: params['waveform'] = 'sine' # Exploit
                         elif mode_code == 1 or mode_code == 2: params['waveform'] = 'chirp' # Explore
                         elif mode_code == 6: params['waveform'] = 'sine' # Hallucinating (sine with diff params)
                         else: params['waveform'] = 'sine' # Default/Idle

                elif should_trigger_or_update and added_count < num_can_add:
                     # Add new sound
                     new_params = {
                         'freq': 220 + np.clip(anomaly * 2000, 0, 2000) + (500 if is_player else 0),
                         'amp': np.clip(0.1 + weight_change * 5 + (0.2 if is_player else 0), 0.05, 0.8),
                         'decay': 1.0,
                         'phase': np.random.rand() * 2 * np.pi,
                         'waveform': 'sine'
                     }
                     if is_player: new_params['waveform'] = 'square'
                     elif mode_code == 0: new_params['waveform'] = 'sine'
                     elif mode_code == 1 or mode_code == 2: new_params['waveform'] = 'chirp'
                     elif mode_code == 6: new_params['freq'] = 100 + np.clip(anomaly*500,0,500); new_params['waveform'] = 'sine'


                     self.audio_params[av_idx] = new_params
                     self.active_audio_indices.add(av_idx)
                     added_count += 1

    def _prepare_data_for_visual_queue(self, av_data_batch):
        """ Converts tensor data to numpy/primitives suitable for inter-process queue. """
        prepared_batch = {}
        for av_idx, data in av_data_batch.items():
            prep_data = {}
            retrieved = data.get('retrieved_memory_vector')
            if retrieved is not None:
                prep_data['retrieved_memory_vector'] = retrieved.squeeze().detach().cpu().numpy()

            anomaly = data.get('anomaly_score')
            if anomaly is not None:
                prep_data['anomaly_score'] = anomaly.item() 

            weight_change = data.get('weight_change_metric')
            if weight_change is not None:
                prep_data['weight_change_metric'] = weight_change.item() 

            prep_data['mode_code'] = data.get('mode_code', 5)
            prep_data['is_player_controlled'] = data.get('is_player_controlled', False)
            
            prepared_batch[av_idx] = prep_data
        return prepared_batch

    def _prepare_data_for_web(self, av_data_batch):
        """ Converts data to JSON-serializable format for SocketIO emission. """
        # Reusing queue prep logic and then converting numpy to list for JSON
        prepared_data_for_queue = self._prepare_data_for_visual_queue(av_data_batch)
        json_batch = {}
        for av_idx, data in prepared_data_for_queue.items():
            json_data = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    json_data[key] = value.tolist() 
                else:
                    json_data[key] = value 
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
            if self.enable_audio_output:
                print(f"AVManager: Setting up SoundDevice stream (Rate: {SAMPLE_RATE}, Blocksize: {AUDIO_BLOCK_SIZE})...")
                self.last_audio_frame = 0
                self.audio_params.clear() # Clear previous params
                self.active_audio_indices.clear()
                self.audio_stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=AUDIO_BLOCK_SIZE,
                    channels=1, 
                    callback=self._audio_callback,
                    dtype='float32'
                )
                self.audio_stream.start()
                print("AVManager: SoundDevice stream started.")
            else: print("AVManager: Audio output disabled.")

            if self.enable_visual_output:
                if self.visualization_mode == 'vispy' and VISPY_AVAILABLE:
                    print("AVManager: Starting VisPy visualization process...")
                    self.visual_data_queue = mp.Queue()
                    self.stop_event.clear() # Clear event before starting new process
                    self.visual_process = mp.Process(
                        target=_vispy_process_target,
                        args=(self.visual_data_queue, self.stop_event, self.num_bots, self.dim, self.pca_available),
                        daemon=True 
                    )
                    self.visual_process.start()
                elif self.visualization_mode == 'matplotlib' and MATPLOTLIB_AVAILABLE:
                    print("AVManager: Starting Matplotlib visualization process...")
                    self.visual_data_queue = mp.Queue()
                    self.stop_event.clear()
                    self.visual_process = mp.Process(
                        target=_matplotlib_process_target,
                         args=(self.visual_data_queue, self.stop_event, self.num_bots, self.dim, self.pca_available),
                        daemon=True
                    )
                    self.visual_process.start()
                elif self.visualization_mode == 'web':
                    print("AVManager: Web visualization mode enabled. SocketIO instance should be set.")
                else:
                    print(f"AVManager: Visual output mode '{self.visualization_mode}' disabled or library not available.")
                    self.enable_visual_output = False 

            self.is_setup = True
            print("AVManager: Setup complete.")

        except Exception as e:
            print(f"AVManager: ERROR during setup: {e}")
            traceback.print_exc()
            self.is_setup = False
            self.stop() 

    def update(self, av_data_batch: dict):
        """ Receives data from the server and processes/queues it. """
        if not self.is_setup or not av_data_batch : # check if batch is empty
            return
        
        if not any(av_data_batch.values()): # check if batch values are empty
             return


        if self.enable_audio_output:
             try: self._process_data_for_audio(av_data_batch)
             except Exception as e: print(f"Error processing audio data: {e}")

        if self.enable_visual_output:
            if self.visualization_mode in ['vispy', 'matplotlib'] and self.visual_process and self.visual_process.is_alive():
                if self.visual_data_queue:
                    try:
                        prepared_data = self._prepare_data_for_visual_queue(av_data_batch)
                        self.visual_data_queue.put(prepared_data, block=False, timeout=QUEUE_TIMEOUT_S/2)
                    except queue.Full:
                        pass # print("Warning: Visual data queue full, dropping frame.") 
                    except Exception as e:
                        print(f"Error preparing/queuing visual data: {e}")
            elif self.visualization_mode == 'web' and self.socketio_instance:
                 try:
                     json_data = self._prepare_data_for_web(av_data_batch)
                     self.socketio_instance.emit('visual_update', json_data, namespace=self.socketio_namespace)
                 except Exception as e:
                     print(f"Error preparing/emitting web visual data: {e}")


    def stop(self):
        """ Stops output threads/processes and cleans up resources. """
        print("AVManager: Stopping...")
        self.stop_event.set() 

        if self.audio_stream:
            try:
                print("AVManager: Stopping SoundDevice stream...")
                self.audio_stream.stop(ignore_errors=True)
                self.audio_stream.close(ignore_errors=True)
                print("AVManager: SoundDevice stream stopped and closed.")
            except Exception as e:
                print(f"AVManager: Error stopping audio stream: {e}")
            self.audio_stream = None

        if self.visual_process and self.visual_process.is_alive():
             print(f"AVManager: Waiting for {self.visualization_mode} process to finish...")
             self.visual_process.join(timeout=3.0) 
             if self.visual_process.is_alive():
                  print(f"AVManager: Warning: {self.visualization_mode} process did not exit gracefully. Terminating.")
                  self.visual_process.terminate() 
                  self.visual_process.join(timeout=1.0) 
             else:
                  print(f"AVManager: {self.visualization_mode} process joined.")
        
        if self.visual_data_queue:
            # It's good practice to empty the queue before closing to help joining threads
            while not self.visual_data_queue.empty():
                try:
                    self.visual_data_queue.get_nowait()
                except queue.Empty:
                    break
                except (EOFError, BrokenPipeError): # Handle cases where queue might be broken
                    break
            self.visual_data_queue.close()
            try:
                self.visual_data_queue.join_thread() 
            except (AssertionError, AttributeError): # join_thread might not be available or raise if already closed/empty
                pass


        self.visual_process = None
        self.visual_data_queue = None
        self.is_setup = False
        print("AVManager: Stop sequence complete.")

# --- Global Functions for Server Interaction ---

_current_av_manager = None

def setup_av_system(num_bots, dim, device, enable_output, visualization_mode, socketio_instance=None):
    """ Creates, sets up, and returns the AVManager instance. """
    global _current_av_manager
    if _current_av_manager:
        print("Warning: Existing AVManager found during setup. Stopping previous one.")
        stop_av_system(_current_av_manager) # This will set _current_av_manager to None
        _current_av_manager = None


    libs_ok = True
    if visualization_mode == 'vispy' and not VISPY_AVAILABLE: libs_ok = False
    if visualization_mode == 'matplotlib' and not MATPLOTLIB_AVAILABLE: libs_ok = False
    
    # Audio can be enabled independently of visual mode, but still needs SoundDevice
    effective_enable_audio = enable_output and SOUNDDEVICE_AVAILABLE
    if enable_output and not SOUNDDEVICE_AVAILABLE:
        print("Warning: SoundDevice not available, audio output will be disabled for this AVManager instance.")


    if not libs_ok and enable_output and visualization_mode not in ['web', 'none']: # Web/none don't depend on these libs_ok
         print(f"AV Setup Skipped for Visuals: Required library for mode '{visualization_mode}' is not available.")
         # If only visual fails, audio might still proceed if enabled
         if not effective_enable_audio: # If audio also not possible
             return None


    try:
        print(f"Setting up AV System: Mode={visualization_mode}, Bots={num_bots}, Dim={dim}, Device={device}, OutputEn={enable_output}")
        # Pass effective_enable_audio to AVManager constructor
        manager = AVManager(num_bots, dim, device, enable_output, visualization_mode) # enable_output here controls both potential outputs

        if visualization_mode == 'web' and socketio_instance:
            manager.set_socketio_instance(socketio_instance)

        manager.setup() 

        # Setup is successful if the manager says it is, OR if all outputs were intentionally disabled
        if manager.is_setup or not enable_output: 
             _current_av_manager = manager
             print("AV System setup successful.")
             return manager
        else:
             print("AV System setup failed (manager.is_setup is false).")
             manager.stop() 
             return None
    except Exception as e:
        print(f"FATAL ERROR during AV setup: {e}")
        traceback.print_exc()
        if _current_av_manager: 
             _current_av_manager.stop()
             _current_av_manager = None
        # Try to stop a partially created manager if it's not the global one yet
        elif 'manager' in locals() and manager:
             manager.stop()
        return None

def update_av_system(manager: AVManager, av_data_batch: dict):
    """ Updates the AV system with new data. """
    if manager and manager.is_setup and (manager.enable_audio_output or manager.enable_visual_output):
        try:
            manager.update(av_data_batch)
        except Exception as e:
            print(f"Error during AV update: {e}")

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
    if manager is _current_av_manager: # Check if the manager being stopped is the current global one
         _current_av_manager = None 


print("Audiovisualization Library Loaded.")
if sys.platform == "darwin":
    try:
        pass 
    except Exception as e:
        print(f"Note: Could not check visualization backend details ({e})")

