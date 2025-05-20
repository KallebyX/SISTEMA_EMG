"""
Módulo para janelamento e segmentação de sinais EMG.

Este módulo contém funções para aplicar janelas, segmentar sinais
e processar janelas de dados EMG para análise em tempo real.
"""

import numpy as np
from scipy import signal
from . import filters, feature_extraction

def apply_window(signal_data, window_type='hamming'):
    """
    Aplica uma janela ao sinal.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
        window_type (str): Tipo de janela ('hamming', 'hanning', 'blackman', 'rectangular').
    
    Returns:
        numpy.ndarray: Sinal com janela aplicada.
    """
    if window_type == 'rectangular':
        window = np.ones_like(signal_data)
    else:
        if window_type == 'hamming':
            window = np.hamming(len(signal_data))
        elif window_type == 'hanning':
            window = np.hanning(len(signal_data))
        elif window_type == 'blackman':
            window = np.blackman(len(signal_data))
        else:
            raise ValueError(f"Tipo de janela não suportado: {window_type}")
    
    return signal_data * window

def segment_signal(signal_data, window_size=256, overlap=0.5):
    """
    Segmenta o sinal em janelas com sobreposição.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
        window_size (int): Tamanho da janela em amostras.
        overlap (float): Fração de sobreposição entre janelas (0 a 1).
    
    Returns:
        list: Lista de janelas (arrays numpy).
    """
    # Verifica parâmetros
    if overlap < 0 or overlap >= 1:
        raise ValueError("A sobreposição deve estar entre 0 e 1 (exclusivo)")
    
    # Calcula o passo entre janelas
    step = int(window_size * (1 - overlap))
    
    # Segmenta o sinal
    windows = []
    for i in range(0, len(signal_data) - window_size + 1, step):
        window = signal_data[i:i + window_size]
        windows.append(window)
    
    return windows

def process_window(signal_data, fs=1000, apply_filtering=True, window_type='hamming'):
    """
    Processa uma janela de sinal EMG, aplicando filtros e extraindo características.
    
    Args:
        signal_data (numpy.ndarray): Janela de sinal EMG.
        fs (float): Frequência de amostragem em Hz.
        apply_filtering (bool): Se True, aplica filtros ao sinal.
        window_type (str): Tipo de janela a ser aplicada.
    
    Returns:
        dict: Dicionário com o sinal original, sinal filtrado e características extraídas.
    """
    # Cria o dicionário de resultado
    result = {
        'signal': signal_data,
        'timestamp': np.datetime64('now')
    }
    
    # Aplica filtros se solicitado
    if apply_filtering and len(signal_data) > 20:  # Garante que o sinal é longo o suficiente para filtragem
        # Aplica filtros individualmente para evitar problemas com frequências de corte
        filtered_signal = filters.detrend_signal(signal_data)
        
        # Só aplica filtros de frequência se o sinal for longo o suficiente
        if len(signal_data) >= 50:
            filtered_signal = filters.notch_filter(filtered_signal, freq=60, fs=fs)
            
            # Ajusta as frequências de corte para garantir que estejam dentro do intervalo válido
            nyq = 0.5 * fs
            lowcut = min(10, nyq * 0.8)  # Garante que lowcut < nyq
            highcut = min(500, nyq * 0.9)  # Garante que highcut < nyq
            
            filtered_signal = filters.bandpass_filter(filtered_signal, lowcut=lowcut, highcut=highcut, fs=fs)
    else:
        filtered_signal = signal_data
    
    # Aplica janela
    windowed_signal = apply_window(filtered_signal, window_type=window_type)
    
    # Adiciona sinais processados ao resultado
    result['filtered_signal'] = filtered_signal
    result['windowed_signal'] = windowed_signal
    
    # Extrai características
    features = feature_extraction.extract_features(windowed_signal, fs=fs)
    result['features'] = features
    
    return result

def process_signal_stream(signal_stream, window_size=256, overlap=0.5, fs=1000):
    """
    Processa um fluxo contínuo de sinal EMG em tempo real.
    
    Args:
        signal_stream (numpy.ndarray): Fluxo de sinal EMG.
        window_size (int): Tamanho da janela em amostras.
        overlap (float): Fração de sobreposição entre janelas (0 a 1).
        fs (float): Frequência de amostragem em Hz.
    
    Returns:
        list: Lista de dicionários com os resultados do processamento de cada janela.
    """
    # Segmenta o sinal em janelas
    windows = segment_signal(signal_stream, window_size=window_size, overlap=overlap)
    
    # Processa cada janela
    results = []
    for window in windows:
        result = process_window(window, fs=fs)
        results.append(result)
    
    return results

def create_sliding_window_processor(window_size=256, overlap=0.5, fs=1000):
    """
    Cria um processador de janela deslizante para processamento em tempo real.
    
    Args:
        window_size (int): Tamanho da janela em amostras.
        overlap (float): Fração de sobreposição entre janelas (0 a 1).
        fs (float): Frequência de amostragem em Hz.
    
    Returns:
        callable: Função para processar novas amostras.
    """
    # Buffer para armazenar amostras
    buffer = np.zeros(window_size)
    step = int(window_size * (1 - overlap))
    position = 0
    
    def process_new_samples(new_samples):
        """
        Processa novas amostras e retorna resultados quando uma janela completa está disponível.
        
        Args:
            new_samples (numpy.ndarray): Novas amostras de sinal EMG.
        
        Returns:
            list: Lista de dicionários com os resultados do processamento, ou lista vazia se nenhuma janela completa estiver disponível.
        """
        nonlocal buffer, position
        
        # Adiciona novas amostras ao buffer
        n_new = len(new_samples)
        if n_new >= window_size:
            # Se há amostras suficientes, usa apenas as mais recentes
            buffer = new_samples[-window_size:]
            position = 0
        else:
            # Desloca o buffer e adiciona novas amostras
            buffer = np.roll(buffer, -n_new)
            buffer[-n_new:] = new_samples
            position += n_new
        
        # Verifica se há janelas completas disponíveis
        results = []
        while position >= step:
            # Processa a janela atual
            result = process_window(buffer, fs=fs)
            results.append(result)
            
            # Atualiza a posição
            position -= step
        
        return results
    
    return process_new_samples
