"""
Implementação de filtros digitais para processamento de sinais EMG.

Este módulo contém funções para filtragem de sinais EMG, incluindo
filtros passa-banda, passa-alta, passa-baixa e notch.
"""

import numpy as np
from scipy import signal

def bandpass_filter(signal_data, lowcut=10, highcut=500, fs=1000, order=4):
    """
    Aplica um filtro passa-banda ao sinal.
    
    Args:
        signal_data (numpy.ndarray): Sinal a ser filtrado.
        lowcut (float): Frequência de corte inferior em Hz.
        highcut (float): Frequência de corte superior em Hz.
        fs (float): Frequência de amostragem em Hz.
        order (int): Ordem do filtro.
    
    Returns:
        numpy.ndarray: Sinal filtrado.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, signal_data)
    
    return filtered_signal

def notch_filter(signal_data, freq=60, q=30, fs=1000):
    """
    Aplica um filtro notch para remover interferência de frequência específica.
    
    Args:
        signal_data (numpy.ndarray): Sinal a ser filtrado.
        freq (float): Frequência a ser removida em Hz.
        q (float): Fator de qualidade do filtro.
        fs (float): Frequência de amostragem em Hz.
    
    Returns:
        numpy.ndarray: Sinal filtrado.
    """
    nyq = 0.5 * fs
    w0 = freq / nyq
    
    b, a = signal.iirnotch(w0, q)
    filtered_signal = signal.filtfilt(b, a, signal_data)
    
    return filtered_signal

def highpass_filter(signal_data, cutoff=10, fs=1000, order=4):
    """
    Aplica um filtro passa-alta ao sinal.
    
    Args:
        signal_data (numpy.ndarray): Sinal a ser filtrado.
        cutoff (float): Frequência de corte em Hz.
        fs (float): Frequência de amostragem em Hz.
        order (int): Ordem do filtro.
    
    Returns:
        numpy.ndarray: Sinal filtrado.
    """
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    filtered_signal = signal.filtfilt(b, a, signal_data)
    
    return filtered_signal

def lowpass_filter(signal_data, cutoff=500, fs=1000, order=4):
    """
    Aplica um filtro passa-baixa ao sinal.
    
    Args:
        signal_data (numpy.ndarray): Sinal a ser filtrado.
        cutoff (float): Frequência de corte em Hz.
        fs (float): Frequência de amostragem em Hz.
        order (int): Ordem do filtro.
    
    Returns:
        numpy.ndarray: Sinal filtrado.
    """
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    filtered_signal = signal.filtfilt(b, a, signal_data)
    
    return filtered_signal

def detrend_signal(signal_data):
    """
    Remove a tendência (drift) do sinal.
    
    Args:
        signal_data (numpy.ndarray): Sinal a ser processado.
    
    Returns:
        numpy.ndarray: Sinal sem tendência.
    """
    return signal.detrend(signal_data)

def normalize_signal(signal_data):
    """
    Normaliza o sinal para média zero e desvio padrão unitário.
    
    Args:
        signal_data (numpy.ndarray): Sinal a ser normalizado.
    
    Returns:
        numpy.ndarray: Sinal normalizado.
    """
    return (signal_data - np.mean(signal_data)) / np.std(signal_data)

def apply_all_filters(signal_data, fs=1000, notch_freq=60, bandpass_low=10, bandpass_high=500):
    """
    Aplica todos os filtros em sequência ao sinal.
    
    Args:
        signal_data (numpy.ndarray): Sinal a ser filtrado.
        fs (float): Frequência de amostragem em Hz.
        notch_freq (float): Frequência do filtro notch em Hz.
        bandpass_low (float): Frequência de corte inferior do filtro passa-banda em Hz.
        bandpass_high (float): Frequência de corte superior do filtro passa-banda em Hz.
    
    Returns:
        numpy.ndarray: Sinal filtrado.
    """
    # Remove tendência
    signal_data = detrend_signal(signal_data)
    
    # Aplica filtro notch para remover interferência da rede elétrica
    signal_data = notch_filter(signal_data, freq=notch_freq, fs=fs)
    
    # Aplica filtro passa-banda
    signal_data = bandpass_filter(signal_data, lowcut=bandpass_low, highcut=bandpass_high, fs=fs)
    
    # Normaliza o sinal
    signal_data = normalize_signal(signal_data)
    
    return signal_data
