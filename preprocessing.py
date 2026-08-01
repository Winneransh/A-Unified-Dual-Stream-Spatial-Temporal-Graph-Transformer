import os
import numpy as np
import mne
from scipy import signal
from scipy.stats import entropy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# VECTORIZED PREPROCESSING FUNCTIONS
# ============================================================================

def apply_bandpass_filter_vectorized(data, sfreq, lowcut=0.5, highcut=50.0):
    """
    Vectorized bandpass filter for all channels at once.
    
    Args:
        data: (n_channels, n_samples)
        sfreq: Sampling frequency
    Returns:
        filtered_data: (n_channels, n_samples)
    """
    nyquist = sfreq / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Apply filter to all channels at once (vectorized)
    filtered_data = signal.filtfilt(b, a, data, axis=1)
    
    return filtered_data


def segment_into_epochs_vectorized(data, sfreq, epoch_length_sec=8, overlap_sec=1):
    """
    Vectorized segmentation into epochs.
    
    Args:
        data: (n_channels, n_samples)
        sfreq: Sampling frequency
        epoch_length_sec: Epoch length in seconds
        overlap_sec: Overlap in seconds
    Returns:
        epochs: (n_epochs, n_channels, epoch_samples)
    """
    n_channels, n_samples = data.shape
    epoch_samples = int(epoch_length_sec * sfreq)
    step_samples = int((epoch_length_sec - overlap_sec) * sfreq)
    
    # Calculate number of epochs
    n_epochs = (n_samples - epoch_samples) // step_samples + 1
    
    # Preallocate array
    epochs = np.zeros((n_epochs, n_channels, epoch_samples))
    
    # Vectorized slicing
    for i in range(n_epochs):
        start = i * step_samples
        end = start + epoch_samples
        epochs[i] = data[:, start:end]
    
    return epochs


# ============================================================================
# FRACTAL DIMENSION COMPUTATION FUNCTIONS
# ============================================================================

def compute_higuchi_fd(signal_data, kmax=10):
    """
    Compute Higuchi Fractal Dimension for a single signal.
    
    Args:
        signal_data: 1D array
        kmax: Maximum k value
    Returns:
        hfd: Higuchi Fractal Dimension
    """
    N = len(signal_data)
    L = np.zeros(kmax)
    
    for k in range(1, kmax + 1):
        Lk = 0
        for m in range(k):
            Lmk = 0
            max_index = int(np.floor((N - m - 1) / k))
            
            for i in range(1, max_index + 1):
                Lmk += abs(signal_data[m + i * k] - signal_data[m + (i - 1) * k])
            
            Lmk = (Lmk * (N - 1)) / (max_index * k * k)
            Lk += Lmk
        
        L[k - 1] = Lk / k
    
    # Fit log-log plot
    x = np.log(1.0 / np.arange(1, kmax + 1))
    y = np.log(L)
    
    # Linear regression
    coeffs = np.polyfit(x, y, 1)
    hfd = coeffs[0]
    
    return hfd


def compute_petrosian_fd(signal_data):
    """
    Compute Petrosian Fractal Dimension for a single signal.
    
    Args:
        signal_data: 1D array
    Returns:
        pfd: Petrosian Fractal Dimension
    """
    N = len(signal_data)
    
    # Compute first derivative
    diff = np.diff(signal_data)
    
    # Count sign changes
    N_delta = np.sum(diff[:-1] * diff[1:] < 0)
    
    # Petrosian FD formula
    pfd = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * N_delta)))
    
    return pfd


def compute_katz_fd(signal_data):
    """
    Compute Katz Fractal Dimension for a single signal.
    
    Args:
        signal_data: 1D array
    Returns:
        kfd: Katz Fractal Dimension
    """
    N = len(signal_data)
    
    # Compute distances
    dists = np.abs(np.diff(signal_data))
    L = np.sum(dists)  # Total length
    
    # Compute diameter (max distance from first point)
    d = np.max(np.abs(signal_data - signal_data[0]))
    
    if d == 0 or L == 0:
        return 0
    
    # Katz FD formula
    kfd = np.log10(N) / (np.log10(N) + np.log10(d / L))
    
    return kfd


def compute_dfa(signal_data, min_scale=4, max_scale=None):
    """
    Compute Detrended Fluctuation Analysis (DFA) exponent for a single signal.
    
    Args:
        signal_data: 1D array
        min_scale: Minimum scale
        max_scale: Maximum scale (default: N/4)
    Returns:
        alpha: DFA exponent (scaling exponent)
    """
    N = len(signal_data)
    
    if max_scale is None:
        max_scale = N // 4
    
    # Compute cumulative sum (integration)
    y = np.cumsum(signal_data - np.mean(signal_data))
    
    # Range of scales
    scales = np.unique(np.logspace(np.log10(min_scale), 
                                   np.log10(max_scale), 
                                   num=15, dtype=int))
    
    F = np.zeros(len(scales))
    
    for i, scale in enumerate(scales):
        # Divide into segments
        n_segments = N // scale
        
        if n_segments < 1:
            continue
        
        fluctuations = []
        
        for seg in range(n_segments):
            start = seg * scale
            end = start + scale
            segment = y[start:end]
            
            # Fit polynomial trend
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            
            # Compute fluctuation
            fluctuation = np.sqrt(np.mean((segment - trend) ** 2))
            fluctuations.append(fluctuation)
        
        F[i] = np.mean(fluctuations)
    
    # Remove zeros
    valid = F > 0
    scales = scales[valid]
    F = F[valid]
    
    if len(scales) < 2:
        return 1.0
    
    # Fit log-log plot
    coeffs = np.polyfit(np.log10(scales), np.log10(F), 1)
    alpha = coeffs[0]
    
    return alpha


# ============================================================================
# VECTORIZED TIME DOMAIN FEATURES WITH FRACTAL DIMENSIONS
# ============================================================================

def extract_time_domain_features_vectorized(epochs):
    """
    Vectorized extraction of time domain features with FRACTAL DIMENSIONS.
    
    Args:
        epochs: (n_epochs, n_channels, epoch_samples)
    Returns:
        features: (n_epochs, n_channels, 7)
        Features: [Mean, Variance, Std, Higuchi FD, Petrosian FD, Katz FD, DFA]
    """
    n_epochs, n_channels, n_samples = epochs.shape
    features = np.zeros((n_epochs, n_channels, 7))
    
    # 1. Mean (vectorized)
    features[:, :, 0] = np.mean(epochs, axis=2)
    
    # 2. Variance (vectorized)
    features[:, :, 1] = np.var(epochs, axis=2)
    
    # 3. Standard Deviation (vectorized)
    features[:, :, 2] = np.std(epochs, axis=2)
    
    # 4-7. Fractal Dimensions (require loop)
    for epoch_idx in range(n_epochs):
        for ch in range(n_channels):
            signal_data = epochs[epoch_idx, ch]
            
            # 4. Higuchi Fractal Dimension
            features[epoch_idx, ch, 3] = compute_higuchi_fd(signal_data, kmax=10)
            
            # 5. Petrosian Fractal Dimension
            features[epoch_idx, ch, 4] = compute_petrosian_fd(signal_data)
            
            # 6. Katz Fractal Dimension
            features[epoch_idx, ch, 5] = compute_katz_fd(signal_data)
            
            # 7. DFA (Detrended Fluctuation Analysis)
            features[epoch_idx, ch, 6] = compute_dfa(signal_data)
    
    return features


# ============================================================================
# VECTORIZED FREQUENCY DOMAIN FEATURES
# ============================================================================

def extract_frequency_domain_features_vectorized(epochs, sfreq):
    """
    Vectorized extraction of frequency domain features for all epochs.
    
    Args:
        epochs: (n_epochs, n_channels, epoch_samples)
        sfreq: Sampling frequency
    Returns:
        features: (n_epochs, n_channels, 7)
    """
    n_epochs, n_channels, n_samples = epochs.shape
    features = np.zeros((n_epochs, n_channels, 7))
    
    # Frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    # Process each epoch and channel
    for epoch_idx in range(n_epochs):
        for ch in range(n_channels):
            signal_data = epochs[epoch_idx, ch]
            
            # Compute PSD using Welch's method
            freqs, psd = signal.welch(signal_data, fs=sfreq, 
                                     nperseg=min(256, len(signal_data)))
            
            # Normalize PSD
            psd_norm = psd / (np.sum(psd) + 1e-10)
            
            # 1. Spectral Entropy
            features[epoch_idx, ch, 0] = entropy(psd_norm + 1e-10)
            
            # 2. Peak Frequency
            features[epoch_idx, ch, 1] = freqs[np.argmax(psd)]
            
            # 3-7. Band Powers
            for band_idx, (band_name, (low, high)) in enumerate(bands.items()):
                idx_band = np.logical_and(freqs >= low, freqs <= high)
                band_power = np.sum(psd[idx_band])
                features[epoch_idx, ch, 2 + band_idx] = band_power
    
    return features


# ============================================================================
# VECTORIZED GRAPH CONSTRUCTION
# ============================================================================

def compute_coherence_matrix_vectorized(epochs, sfreq):
    """
    Vectorized coherence matrix computation for all epochs.
    
    Args:
        epochs: (n_epochs, n_channels, epoch_samples)
        sfreq: Sampling frequency
    Returns:
        coherence_matrices: (n_epochs, n_channels, n_channels)
    """
    n_epochs, n_channels, n_samples = epochs.shape
    coherence_matrices = np.zeros((n_epochs, n_channels, n_channels))
    
    for epoch_idx in range(n_epochs):
        epoch = epochs[epoch_idx]
        
        for i in range(n_channels):
            coherence_matrices[epoch_idx, i, i] = 1.0
            
            for j in range(i + 1, n_channels):
                # Compute coherence
                freqs, Cxy = signal.coherence(epoch[i], epoch[j], 
                                             fs=sfreq, 
                                             nperseg=min(256, n_samples))
                
                # Average coherence across frequencies
                avg_coherence = np.mean(Cxy)
                
                coherence_matrices[epoch_idx, i, j] = avg_coherence
                coherence_matrices[epoch_idx, j, i] = avg_coherence
    
    return coherence_matrices


def compute_plv_matrix_vectorized(epochs):
    """
    Vectorized PLV matrix computation for all epochs.
    
    Args:
        epochs: (n_epochs, n_channels, epoch_samples)
    Returns:
        plv_matrices: (n_epochs, n_channels, n_channels)
    """
    n_epochs, n_channels, n_samples = epochs.shape
    plv_matrices = np.zeros((n_epochs, n_channels, n_channels))
    
    # Compute instantaneous phase for all epochs at once (vectorized)
    analytic_signal = signal.hilbert(epochs, axis=2)
    instantaneous_phase = np.angle(analytic_signal)
    
    for epoch_idx in range(n_epochs):
        phases = instantaneous_phase[epoch_idx]
        
        for i in range(n_channels):
            plv_matrices[epoch_idx, i, i] = 1.0
            
            for j in range(i + 1, n_channels):
                # Compute phase difference
                phase_diff = phases[i] - phases[j]
                
                # Compute PLV
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                
                plv_matrices[epoch_idx, i, j] = plv
                plv_matrices[epoch_idx, j, i] = plv
    
    return plv_matrices


def construct_adjacency_matrices_vectorized(epochs, sfreq, method='combined'):
    """
    Vectorized adjacency matrix construction for all epochs.
    
    Args:
        epochs: (n_epochs, n_channels, epoch_samples)
        sfreq: Sampling frequency
        method: 'coherence', 'plv', or 'combined'
    Returns:
        adjacency_matrices: (n_epochs, n_channels, n_channels)
    """
    if method == 'coherence':
        return compute_coherence_matrix_vectorized(epochs, sfreq)
    elif method == 'plv':
        return compute_plv_matrix_vectorized(epochs)
    elif method == 'combined':
        coherence = compute_coherence_matrix_vectorized(epochs, sfreq)
        plv = compute_plv_matrix_vectorized(epochs)
        return (coherence + plv) / 2.0
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# MAIN PREPROCESSING FUNCTION (OPTIMIZED)
# ============================================================================

def preprocess_edf_file(edf_path, epoch_length_sec=8, overlap_sec=1):
    """
    OPTIMIZED preprocessing pipeline for one EDF file with TEMPORAL SEQUENCE preservation.
    Now includes FRACTAL FEATURES instead of RMS, ZCR, Hjorth Mobility/Complexity.
    
    Args:
        edf_path: Path to .edf file
        epoch_length_sec: Epoch length in seconds (5 or 8)
        overlap_sec: Overlap in seconds (1 sec)
    Returns:
        result_dict: Dictionary containing:
            - 'features': (n_epochs, n_channels, 14) - Node features for each epoch
            - 'adjacency': (n_epochs, n_channels, n_channels) - Adjacency matrices
            - 'n_epochs': Number of epochs (TEMPORAL SEQUENCE LENGTH)
            - 'n_channels': Number of channels
            - 'filename': Original filename
    """
    filename = os.path.basename(edf_path)
    
    # Load EDF file
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    data = raw.get_data()  # (n_channels, n_samples)
    n_channels = data.shape[0]
    
    # Step 1: Bandpass filter (VECTORIZED)
    filtered_data = apply_bandpass_filter_vectorized(data, sfreq, lowcut=0.5, highcut=50.0)
    
    # Step 2: Segment into epochs (VECTORIZED)
    epochs = segment_into_epochs_vectorized(filtered_data, sfreq, epoch_length_sec, overlap_sec)
    n_epochs = epochs.shape[0]
    
    # Step 3: Extract time domain features with FRACTAL DIMENSIONS (VECTORIZED + FRACTAL)
    time_features = extract_time_domain_features_vectorized(epochs)
    
    # Step 4: Extract frequency domain features (VECTORIZED)
    freq_features = extract_frequency_domain_features_vectorized(epochs, sfreq)
    
    # Step 5: Combine features (n_epochs, n_channels, 14)
    all_features = np.concatenate([time_features, freq_features], axis=2)
    
    # Step 6: Construct adjacency matrices (VECTORIZED)
    adjacency_matrices = construct_adjacency_matrices_vectorized(epochs, sfreq, method='combined')
    
    # Return as dictionary with TEMPORAL SEQUENCE preserved
    result_dict = {
        'features': all_features,  # (n_epochs, n_channels, 14)
        'adjacency': adjacency_matrices,  # (n_epochs, n_channels, n_channels)
        'n_epochs': n_epochs,  # TEMPORAL SEQUENCE LENGTH
        'n_channels': n_channels,
        'filename': filename
    }
    
    return result_dict


# ============================================================================
# PARALLEL PROCESSING WITH PROGRESS BAR
# ============================================================================

def process_single_file(edf_path, epoch_length_sec, overlap_sec):
    """
    Wrapper function for parallel processing.
    """
    try:
        # Determine label from filename
        filename = os.path.basename(edf_path)
        if filename.startswith('h'):
            label = 0  # Healthy
        elif filename.startswith('s'):
            label = 1  # Schizophrenia
        else:
            return None
        
        # Preprocess file
        result = preprocess_edf_file(edf_path, epoch_length_sec, overlap_sec)
        result['label'] = label
        result['filepath'] = edf_path
        
        return result
        
    except Exception as e:
        print(f"\nError processing {os.path.basename(edf_path)}: {str(e)}")
        return None


def load_and_preprocess_dataset(data_dir, epoch_length_sec=8, overlap_sec=1, 
                                n_jobs=-1, save_pickle=True, pickle_path=None):
    """
    Load and preprocess all EDF files with parallel processing.
    PRESERVES TEMPORAL SEQUENCES for GCN-LSTM.
    Now includes FRACTAL FEATURES.
    
    Args:
        data_dir: Directory containing .edf files
        epoch_length_sec: Epoch length (5 or 8 seconds)
        overlap_sec: Overlap (1 second)
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        save_pickle: Whether to save preprocessed data
        pickle_path: Path to save pickle file
    Returns:
        preprocessed_data: List of dictionaries, each containing:
            - 'features': (n_epochs, n_channels, 14)
            - 'adjacency': (n_epochs, n_channels, n_channels)
            - 'label': 0 or 1
            - 'filename': str
            - 'n_epochs': int
            - 'n_channels': int
    """
    print("\n" + "="*80)
    print("LOADING AND PREPROCESSING DATASET (GCN-LSTM with FRACTAL FEATURES)")
    print("="*80)
    
    edf_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.edf')])
    edf_paths = [os.path.join(data_dir, f) for f in edf_files]
    
    print(f"\nFound {len(edf_files)} EDF files in {data_dir}")
    print(f"Using {n_jobs if n_jobs > 0 else 'all'} CPU cores for parallel processing")
    
    # Parallel processing with progress bar
    print("\nProcessing files in parallel...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_file)(path, epoch_length_sec, overlap_sec) 
        for path in tqdm(edf_paths, desc="Processing EDF files")
    )
    
    # Filter out failed files
    preprocessed_data = [r for r in results if r is not None]
    
    print(f"\nSuccessfully processed {len(preprocessed_data)} files")
    
    # Count labels
    n_healthy = sum(1 for d in preprocessed_data if d['label'] == 0)
    n_schizo = sum(1 for d in preprocessed_data if d['label'] == 1)
    
    print(f"  Healthy: {n_healthy}")
    print(f"  Schizophrenia: {n_schizo}")
    
    # Print temporal sequence info
    avg_epochs = np.mean([d['n_epochs'] for d in preprocessed_data])
    print(f"\nAverage temporal sequence length: {avg_epochs:.1f} epochs per file")
    print(f"Each epoch represents {epoch_length_sec} seconds of EEG data")
    
    # Save to pickle
    if save_pickle:
        if pickle_path is None:
            pickle_path = os.path.join(data_dir, 
                                      f'preprocessed_gcn_lstm_fractal_epoch{epoch_length_sec}s.pkl')
        
        print(f"\nSaving preprocessed data to: {pickle_path}")
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'data': preprocessed_data,
                'epoch_length_sec': epoch_length_sec,
                'overlap_sec': overlap_sec,
                'n_files': len(preprocessed_data),
                'n_healthy': n_healthy,
                'n_schizophrenia': n_schizo,
                'feature_info': 'Time: [Mean, Var, Std, Higuchi_FD, Petrosian_FD, Katz_FD, DFA] + Freq: [Spectral_Entropy, Peak_Freq, Delta, Theta, Alpha, Beta, Gamma]'
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Get file size
        file_size_mb = os.path.getsize(pickle_path) / (1024 * 1024)
        print(f"Saved! File size: {file_size_mb:.2f} MB")
    
    return preprocessed_data