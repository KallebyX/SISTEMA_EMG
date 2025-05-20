"""
Módulo de processamento de sinais EMG.

Este módulo é responsável pelo processamento de sinais EMG, incluindo
filtragem, janelamento e extração de características.
"""

from .filters import bandpass_filter, notch_filter, highpass_filter, lowpass_filter, detrend_signal, normalize_signal, apply_all_filters
from .feature_extraction import calculate_rms, calculate_mav, calculate_variance, calculate_waveform_length, calculate_zero_crossings, calculate_slope_sign_changes, calculate_willison_amplitude, extract_features
from .windowing import apply_window, segment_signal, process_window, process_signal_stream, create_sliding_window_processor

__all__ = [
    # Filtros
    'bandpass_filter', 'notch_filter', 'highpass_filter', 'lowpass_filter', 
    'detrend_signal', 'normalize_signal', 'apply_all_filters',
    
    # Extração de características
    'calculate_rms', 'calculate_mav', 'calculate_variance', 'calculate_waveform_length',
    'calculate_zero_crossings', 'calculate_slope_sign_changes', 'calculate_willison_amplitude',
    'extract_features',
    
    # Janelamento
    'apply_window', 'segment_signal', 'process_window', 'process_signal_stream',
    'create_sliding_window_processor'
]
