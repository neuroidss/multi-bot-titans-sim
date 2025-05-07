# Filename: eeg_feature_extractor_lib.py
# Description: Library for EEG signal processing and feature extraction for neurofeedback.
# Version: 1.1.0 (Added rolling buffer for windowed processing)

import numpy as np
import itertools # For generating all channel pairs
from typing import List, Dict, Tuple, Optional, Any
import traceback
from collections import deque

# --- MNE Imports ---
try:
    import mne
    from mne.utils import logger, set_log_level
    MNE_AVAILABLE = True
    set_log_level(verbose='WARNING') # Reduce MNE verbosity
    if hasattr(mne, '__version__'):
        print(f"eeg_feature_extractor_lib: MNE-Python version {mne.__version__} loaded.")
    else:
        print("eeg_feature_extractor_lib: MNE-Python loaded (version not determined).")
except ImportError:
    MNE_AVAILABLE = False
    print("eeg_feature_extractor_lib: Warning - MNE-Python not found. Some features will be disabled.")

# --- MNE-Connectivity Import ---
try:
    import mne_connectivity
    MNE_CONNECTIVITY_AVAILABLE = True
    if hasattr(mne_connectivity, '__version__'):
        print(f"eeg_feature_extractor_lib: MNE-Connectivity version {mne_connectivity.__version__} loaded.")
    else:
        print("eeg_feature_extractor_lib: MNE-Connectivity loaded (version not determined).")
except ImportError:
    MNE_CONNECTIVITY_AVAILABLE = False
    print("eeg_feature_extractor_lib: Warning - MNE-Connectivity not found. Coherence calculations will be disabled.")

# --- SciPy Import ---
try:
    import scipy
    from scipy.signal import butter, sosfiltfilt
    SCIPY_AVAILABLE = True
    if hasattr(scipy, '__version__'):
        print(f"eeg_feature_extractor_lib: SciPy version {scipy.__version__} loaded.")
    else:
        print("eeg_feature_extractor_lib: SciPy loaded (version not determined).")
except ImportError:
    SCIPY_AVAILABLE = False
    print("eeg_feature_extractor_lib: Warning - SciPy not found. High-pass filtering will be disabled.")


class SignalProcessor:
    def __init__(self, sfreq: float, channel_names: List[str], processing_config: dict, processing_window_samples: int, processing_interval_new_samples: int = 0):
        self.sfreq = sfreq
        self.input_channel_names = channel_names
        self.input_channel_indices = {name: i for i, name in enumerate(channel_names)}
        self.n_input_channels = len(channel_names)
        self.processing_config = processing_config
        self.processing_window_samples = processing_window_samples # This is the analysis window size (e.g., 157 samples)
        self.processing_interval_new_samples = max(1, processing_interval_new_samples if processing_interval_new_samples > 0 else self.processing_window_samples // 4) # How many *new* samples trigger a re-calculation
        
        self.buffer_maxlen = self.processing_window_samples * 2 # Keep a bit more than needed for flexibility
        self.eeg_buffer = deque(maxlen=self.buffer_maxlen) # Stores (channels, samples) arrays
        self.samples_since_last_processing = 0
        self.total_samples_in_buffer = 0

        print(f"SignalProcessor (lib): Initialized for sfreq={sfreq} Hz, processing_window={self.processing_window_samples} samples, processing_interval_new_samples={self.processing_interval_new_samples}, {self.n_input_channels} channels.")

        self.hp_sos = None
        hp_config = self.processing_config.get('filtering', {})
        hp_freq = hp_config.get('highpass_hz')
        hp_order = hp_config.get('highpass_order', 5)
        if SCIPY_AVAILABLE and hp_freq is not None and hp_freq > 0 and self.sfreq > 0:
            nyquist = 0.5 * self.sfreq
            if hp_freq < nyquist:
                try:
                    self.hp_sos = butter(hp_order, hp_freq / nyquist, btype='high', analog=False, output='sos')
                    print(f"SignalProcessor (lib): High-pass filter enabled at {hp_freq} Hz, order {hp_order}.")
                except Exception as e: print(f"SignalProcessor (lib) Warning: Failed to create high-pass filter: {e}")
            else: print(f"SignalProcessor (lib) Warning: High-pass frequency ({hp_freq} Hz) is >= Nyquist ({nyquist} Hz). Filter disabled.")
        elif hp_freq is not None and hp_freq > 0:
             print("SignalProcessor (lib): SciPy not available or sfreq invalid. High-pass filter disabled.")

        self.feature_bands = self.processing_config.get('feature_bands', {})
        self.feature_band_names = list(self.feature_bands.keys())
        print(f"SignalProcessor (lib): Feature bands configured: {self.feature_bands}")

        self.coherence_config = self.processing_config.get('coherence', {})
        self.coherence_methods_to_calc = self.coherence_config.get('methods_to_calculate', [])
        self.calculate_coherence = (MNE_CONNECTIVITY_AVAILABLE and
                                    self.sfreq > 0 and
                                    self.coherence_config.get('enable', False) and
                                    self.coherence_config.get('bands') and
                                    self.coherence_methods_to_calc and
                                    self.n_input_channels >=2)

        self.coherence_pairs_indices_tuple: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.coherence_pairs_names_list: List[Tuple[str, str]] = []
        self.coherence_band_map: Dict[str, Tuple[float, float]] = {}
        self.coh_fmin_overall: float = 1.0
        self.coh_fmax_overall: float = 40.0

        if self.calculate_coherence:
            print(f"SignalProcessor (lib): Coherence calculation will be attempted. Methods: {self.coherence_methods_to_calc}")
            bands_to_calc_for_coh = self.coherence_config['bands']
            src_indices_list, tgt_indices_list = [], []
            valid_pairs_names = []
            missing_channels_in_pairs_set = set()
            use_all_pairs_flag = self.coherence_config.get('use_all_channel_pairs', False)

            if use_all_pairs_flag:
                num_possible_pairs = len(list(itertools.combinations(range(self.n_input_channels), 2)))
                print(f"SignalProcessor (lib): Coherence using ALL available unique channel pairs ({num_possible_pairs} pairs) due to 'use_all_channel_pairs: true'.")
                indices_pairs_gen = itertools.combinations(range(self.n_input_channels), 2)
                for idx0, idx1 in indices_pairs_gen:
                    src_indices_list.append(idx0); tgt_indices_list.append(idx1)
                    valid_pairs_names.append((self.input_channel_names[idx0], self.input_channel_names[idx1]))
            else:
                pair_list_from_config = self.coherence_config.get('pairs', [])
                if not pair_list_from_config:
                    print("SignalProcessor (lib) Warning: Coherence 'pairs' list is empty and 'use_all_channel_pairs' is false. Disabling coherence.")
                    self.calculate_coherence = False
                else:
                    for pair_spec in pair_list_from_config:
                        if isinstance(pair_spec, (list, tuple)) and len(pair_spec) == 2 and isinstance(pair_spec[0], str) and isinstance(pair_spec[1], str):
                            ch_name1, ch_name2 = pair_spec[0], pair_spec[1]
                            if ch_name1 in self.input_channel_indices and ch_name2 in self.input_channel_indices:
                                idx0 = self.input_channel_indices[ch_name1]; idx1 = self.input_channel_indices[ch_name2]
                                if idx0 != idx1 : src_indices_list.append(idx0); tgt_indices_list.append(idx1); valid_pairs_names.append((ch_name1, ch_name2))
                            else:
                                if ch_name1 not in self.input_channel_indices: missing_channels_in_pairs_set.add(ch_name1)
                                if ch_name2 not in self.input_channel_indices: missing_channels_in_pairs_set.add(ch_name2)
                    if missing_channels_in_pairs_set: print(f"SignalProcessor (lib) Warning: Coherence channels not in input: {', '.join(sorted(list(missing_channels_in_pairs_set)))}")
            
            if not src_indices_list and self.calculate_coherence: self.calculate_coherence = False; print("SignalProcessor (lib) Warning: No valid coherence pairs. Disabling coherence.")

            if self.calculate_coherence:
                self.coherence_pairs_indices_tuple = (np.array(src_indices_list), np.array(tgt_indices_list))
                self.coherence_pairs_names_list = valid_pairs_names
                num_pairs_to_calc = len(valid_pairs_names)
                print(f"  SignalProcessor (lib): Using {num_pairs_to_calc} coherence pairs.")

                valid_coh_bands_dict = {b_name: self.feature_bands[b_name] for b_name in bands_to_calc_for_coh if b_name in self.feature_bands}
                if not valid_coh_bands_dict: self.calculate_coherence = False; print("SignalProcessor (lib) Warning: No valid freq bands for coherence. Disabling coherence.")
                else:
                    self.coherence_band_map = valid_coh_bands_dict
                    self.coh_fmin_overall = min(f_range[0] for f_range in valid_coh_bands_dict.values())
                    self.coh_fmax_overall = max(f_range[1] for f_range in valid_coh_bands_dict.values())
                    if self.sfreq > 0 and self.coh_fmax_overall >= self.sfreq / 2: self.coh_fmax_overall = (self.sfreq / 2) - 0.01
                    
                    min_freq_for_coh = self.coh_fmin_overall
                    if self.sfreq > 0 and self.processing_window_samples > 0 and min_freq_for_coh > 0:
                        epoch_duration_s = self.processing_window_samples / self.sfreq
                        cycles_for_fmin = epoch_duration_s * min_freq_for_coh; min_recommended_cycles = 5.0
                        if cycles_for_fmin < min_recommended_cycles:
                            print(f"SignalProcessor (lib) WARNING: Coherence calculation for {min_freq_for_coh:.2f} Hz with window {self.processing_window_samples} samples ({epoch_duration_s:.3f}s) provides only {cycles_for_fmin:.2f} cycles. Recommend >= {min_recommended_cycles} cycles (i.e., window of at least {int(np.ceil(min_recommended_cycles/min_freq_for_coh * self.sfreq))} samples).")
                    print(f"  SignalProcessor (lib): Overall freq range for MNE Connectivity: {self.coh_fmin_overall:.1f}-{self.coh_fmax_overall:.1f} Hz")
        else: print("SignalProcessor (lib): Coherence disabled (check MNE-Connectivity, config, or channel counts).")

    def append_data(self, new_chunk: np.ndarray):
        """ Appends a new chunk of EEG data (shape: n_channels, n_new_samples) to the buffer. """
        if new_chunk.ndim != 2 or new_chunk.shape[0] != self.n_input_channels or new_chunk.shape[1] == 0:
            print(f"SignalProcessor (lib) Warning: Invalid new_chunk shape: {new_chunk.shape}. Expected ({self.n_input_channels}, n_samples>0).")
            return

        n_new_samples = new_chunk.shape[1]
        
        # If buffer is full or nearly full, trim old data before appending
        while self.total_samples_in_buffer + n_new_samples > self.buffer_maxlen and self.eeg_buffer:
            oldest_chunk = self.eeg_buffer.popleft()
            self.total_samples_in_buffer -= oldest_chunk.shape[1]
            
        self.eeg_buffer.append(new_chunk)
        self.total_samples_in_buffer += n_new_samples
        self.samples_since_last_processing += n_new_samples

    def should_process(self) -> bool:
        """ Checks if enough new data has arrived and total buffer is sufficient. """
        return (self.total_samples_in_buffer >= self.processing_window_samples and
                self.samples_since_last_processing >= self.processing_interval_new_samples)

    def get_processing_window(self) -> Optional[np.ndarray]:
        """ Returns the latest self.processing_window_samples from the buffer. Resets samples_since_last_processing. """
        if self.total_samples_in_buffer < self.processing_window_samples:
            return None

        # Concatenate all chunks in the buffer
        full_buffered_data = np.concatenate(list(self.eeg_buffer), axis=1)
        
        # Extract the latest window_samples
        window_to_process = full_buffered_data[:, -self.processing_window_samples:]
        self.samples_since_last_processing = 0 # Reset counter
        return window_to_process

    def preprocess(self, raw_window: np.ndarray) -> np.ndarray:
        """ raw_window shape: (n_channels, window_samples) """
        processed_window = raw_window.astype(np.float32)
        if self.hp_sos is not None and SCIPY_AVAILABLE:
            try: processed_window = sosfiltfilt(self.hp_sos, processed_window, axis=1)
            except Exception as e: print(f"SignalProcessor (lib) Warning: Preprocessing filter failed: {e}"); return raw_window.astype(np.float32)
        return processed_window

    def extract_features(self, processed_window: np.ndarray, nf_protocol_config: dict) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """ processed_window shape: (n_channels, window_samples) """
        nf_metrics = {}
        viz_features = {'band_power_per_channel': np.zeros((self.n_input_channels, len(self.feature_band_names)), dtype=np.float32)} 
        n_channels_in_chunk, n_samples_in_window = processed_window.shape

        if self.sfreq <= 0 or n_samples_in_window == 0:
            return nf_metrics, viz_features

        if MNE_AVAILABLE and self.feature_bands:
            data_for_psd = processed_window[np.newaxis, :, :] 
            n_fft = min(n_samples_in_window, int(self.sfreq)) 
            n_overlap = n_fft // 2 
            psd_fmax = max(50.0, self.coh_fmax_overall + 5.0) if self.calculate_coherence else 50.0
            psd_fmax = min(psd_fmax, self.sfreq / 2.0 - 0.01) 

            try:
                psd_welch_results = mne.time_frequency.psd_array_welch(
                    data_for_psd, sfreq=self.sfreq, fmin=0.5, fmax=psd_fmax,
                    n_fft=n_fft, n_overlap=n_overlap, average='mean', window='hann', verbose=False
                )
                psds, freqs = psd_welch_results[0][0], psd_welch_results[1]
                temp_band_powers_for_viz = np.zeros((self.n_input_channels, len(self.feature_band_names)), dtype=np.float32)
                for band_idx, band_name in enumerate(self.feature_band_names):
                    fmin_band, fmax_band = self.feature_bands[band_name]
                    band_freq_mask_psd = (freqs >= fmin_band) & (freqs < fmax_band)
                    if np.any(band_freq_mask_psd) and psds[:, band_freq_mask_psd].size > 0:
                        power_in_band_per_ch = np.nanmean(psds[:, band_freq_mask_psd], axis=1)
                        for ch_idx, ch_name in enumerate(self.input_channel_names):
                            ch_power = power_in_band_per_ch[ch_idx]
                            if np.isfinite(ch_power): nf_metrics[f"{band_name}_power_{ch_name}"] = ch_power
                        log_power_for_viz = np.log10(power_in_band_per_ch + 1e-12); log_power_for_viz[~np.isfinite(log_power_for_viz)] = -12.0
                        temp_band_powers_for_viz[:, band_idx] = log_power_for_viz
                viz_features['band_power_per_channel'] = temp_band_powers_for_viz
            except Exception as e: print(f"SignalProcessor (lib) Error PSD/Band Power: {e}"); traceback.print_exc()

        if self.calculate_coherence and self.coherence_pairs_indices_tuple is not None and MNE_CONNECTIVITY_AVAILABLE:
            data_for_con = processed_window[np.newaxis, :, :] 
            for method_name in self.coherence_methods_to_calc:
                try:
                    fmin_list_for_mne = [float(self.coh_fmin_overall)]; fmax_list_for_mne = [float(self.coh_fmax_overall)]
                    con_object = mne_connectivity.spectral_connectivity_epochs(
                        data_for_con, method=method_name, mode='multitaper', sfreq=self.sfreq, indices=self.coherence_pairs_indices_tuple,
                        fmin=fmin_list_for_mne, fmax=fmax_list_for_mne, faverage=False, mt_adaptive=False, n_jobs=1, verbose=False
                    )
                    if not isinstance(con_object, mne_connectivity.base.BaseConnectivity): continue
                    con_values_full_matrix = con_object.get_data(output='dense')
                    raw_con_frequencies = con_object.freqs
                    if isinstance(raw_con_frequencies, list): con_frequencies = np.array(raw_con_frequencies, dtype=np.float32)
                    elif isinstance(raw_con_frequencies, np.ndarray): con_frequencies = raw_con_frequencies.astype(np.float32)
                    else: 
                        try: con_frequencies = np.array(list(raw_con_frequencies), dtype=np.float32) if not isinstance(raw_con_frequencies, np.ndarray) else raw_con_frequencies.astype(np.float32)
                        except: continue
                    if con_frequencies.ndim == 0 or con_frequencies.size == 0: continue
                    if con_values_full_matrix.ndim != 3 or con_values_full_matrix.shape[0] != self.n_input_channels or \
                       con_values_full_matrix.shape[1] != self.n_input_channels or con_values_full_matrix.shape[2] != len(con_frequencies): continue
                    
                    row_indices = self.coherence_pairs_indices_tuple[0]; col_indices = self.coherence_pairs_indices_tuple[1]
                    con_values_selected_pairs = con_values_full_matrix[row_indices, col_indices, :]
                    if con_values_selected_pairs.shape[0] != len(self.coherence_pairs_names_list) or con_values_selected_pairs.shape[1] != len(con_frequencies): continue
                    
                    for pair_idx, (ch1_name, ch2_name) in enumerate(self.coherence_pairs_names_list):
                        con_values_this_pair_spectrum = con_values_selected_pairs[pair_idx, :] 
                        for band_name_coh, (fmin_band_coh, fmax_band_coh) in self.coherence_band_map.items():
                            actual_fmin_in_con = max(fmin_band_coh, con_frequencies[0]); actual_fmax_in_con = min(fmax_band_coh, con_frequencies[-1])
                            band_freq_mask_coh = (con_frequencies >= actual_fmin_in_con) & (con_frequencies < actual_fmax_in_con)
                            if np.any(band_freq_mask_coh) and con_values_this_pair_spectrum[band_freq_mask_coh].size > 0:
                                avg_coh_for_pair_in_band = np.nanmean(con_values_this_pair_spectrum[band_freq_mask_coh])
                                if np.isfinite(avg_coh_for_pair_in_band): nf_metrics[f"{band_name_coh}_{method_name}_{ch1_name}-{ch2_name}"] = float(avg_coh_for_pair_in_band)
                except Exception as e: print(f"SignalProcessor (lib) Error Coherence (method '{method_name}'): {e}"); traceback.print_exc()
        
        nf_metrics_clean = {k: v for k, v in nf_metrics.items() if isinstance(v, (float, np.floating)) and np.isfinite(v)}
        return nf_metrics_clean, viz_features

print("eeg_feature_extractor_lib.py loaded (Version 1.1.0).")

