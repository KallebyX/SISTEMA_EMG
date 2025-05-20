"""
Testes unitários para o módulo de processamento de sinais.

Este script contém testes unitários para validar as funcionalidades
do módulo de processamento de sinais do SISTEMA_EMG.
"""

import os
import sys
import unittest
import numpy as np

# Adiciona o diretório raiz ao path para importação dos módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing import filters
from src.processing import feature_extraction
from src.processing import windowing

class TestFilters(unittest.TestCase):
    """Testes para o módulo de filtros."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        # Cria um sinal de teste com ruído
        self.t = np.linspace(0, 1, 1000)
        self.clean_signal = np.sin(2 * np.pi * 10 * self.t)  # Sinal de 10 Hz
        self.noise_60hz = 0.5 * np.sin(2 * np.pi * 60 * self.t)  # Ruído de 60 Hz
        self.noise_highfreq = 0.2 * np.sin(2 * np.pi * 200 * self.t)  # Ruído de alta frequência
        self.dc_offset = 2.0  # Offset DC
        
        # Sinal com ruído
        self.noisy_signal = self.clean_signal + self.noise_60hz + self.noise_highfreq + self.dc_offset
        
        # Parâmetros de filtro
        self.fs = 1000  # Taxa de amostragem
    
    def test_bandpass_filter(self):
        """Testa o filtro passa-banda."""
        # Aplica o filtro passa-banda
        filtered_signal = filters.bandpass_filter(self.noisy_signal, lowcut=5, highcut=50, fs=self.fs)
        
        # Verifica se o filtro removeu o ruído de alta frequência
        fft_original = np.abs(np.fft.fft(self.noisy_signal))
        fft_filtered = np.abs(np.fft.fft(filtered_signal))
        
        # Índices de frequência para 60 Hz e 200 Hz
        idx_60hz = int(60 * len(self.t) / self.fs)
        idx_200hz = int(200 * len(self.t) / self.fs)
        
        # Verifica se a amplitude em 200 Hz foi reduzida
        self.assertLess(fft_filtered[idx_200hz], fft_original[idx_200hz] * 0.5)
        
        # Verifica se o sinal de 10 Hz foi preservado
        idx_10hz = int(10 * len(self.t) / self.fs)
        self.assertGreater(fft_filtered[idx_10hz], fft_original[idx_10hz] * 0.5)
    
    def test_notch_filter(self):
        """Testa o filtro notch."""
        # Aplica o filtro notch
        filtered_signal = filters.notch_filter(self.noisy_signal, freq=60, fs=self.fs)
        
        # Verifica se o filtro removeu o ruído de 60 Hz
        fft_original = np.abs(np.fft.fft(self.noisy_signal))
        fft_filtered = np.abs(np.fft.fft(filtered_signal))
        
        # Índice de frequência para 60 Hz
        idx_60hz = int(60 * len(self.t) / self.fs)
        
        # Verifica se a amplitude em 60 Hz foi reduzida
        self.assertLess(fft_filtered[idx_60hz], fft_original[idx_60hz] * 0.5)
        
        # Verifica se o sinal de 10 Hz foi preservado
        idx_10hz = int(10 * len(self.t) / self.fs)
        self.assertGreater(fft_filtered[idx_10hz], fft_original[idx_10hz] * 0.5)
    
    def test_highpass_filter(self):
        """Testa o filtro passa-alta."""
        # Aplica o filtro passa-alta
        filtered_signal = filters.highpass_filter(self.noisy_signal, cutoff=5, fs=self.fs)
        
        # Verifica se o filtro removeu o offset DC
        self.assertLess(abs(np.mean(filtered_signal)), abs(np.mean(self.noisy_signal)) * 0.5)
        
        # Verifica se o sinal de 10 Hz foi preservado
        fft_original = np.abs(np.fft.fft(self.noisy_signal))
        fft_filtered = np.abs(np.fft.fft(filtered_signal))
        
        idx_10hz = int(10 * len(self.t) / self.fs)
        self.assertGreater(fft_filtered[idx_10hz], fft_original[idx_10hz] * 0.5)
    
    def test_lowpass_filter(self):
        """Testa o filtro passa-baixa."""
        # Aplica o filtro passa-baixa
        filtered_signal = filters.lowpass_filter(self.noisy_signal, cutoff=50, fs=self.fs)
        
        # Verifica se o filtro removeu o ruído de alta frequência
        fft_original = np.abs(np.fft.fft(self.noisy_signal))
        fft_filtered = np.abs(np.fft.fft(filtered_signal))
        
        # Índice de frequência para 200 Hz
        idx_200hz = int(200 * len(self.t) / self.fs)
        
        # Verifica se a amplitude em 200 Hz foi reduzida
        self.assertLess(fft_filtered[idx_200hz], fft_original[idx_200hz] * 0.5)
        
        # Verifica se o sinal de 10 Hz foi preservado
        idx_10hz = int(10 * len(self.t) / self.fs)
        self.assertGreater(fft_filtered[idx_10hz], fft_original[idx_10hz] * 0.5)

class TestFeatureExtraction(unittest.TestCase):
    """Testes para o módulo de extração de características."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        # Cria um sinal de teste
        self.t = np.linspace(0, 1, 1000)
        self.signal = np.sin(2 * np.pi * 10 * self.t)  # Sinal de 10 Hz
    
    def test_calculate_rms(self):
        """Testa o cálculo do RMS."""
        # Calcula o RMS
        rms = feature_extraction.calculate_rms(self.signal)
        
        # Calcula o RMS manualmente
        expected_rms = np.sqrt(np.mean(np.square(self.signal)))
        
        # Verifica se o resultado está correto
        self.assertAlmostEqual(rms, expected_rms, places=6)
    
    def test_calculate_mav(self):
        """Testa o cálculo do MAV (Mean Absolute Value)."""
        # Calcula o MAV
        mav = feature_extraction.calculate_mav(self.signal)
        
        # Calcula o MAV manualmente
        expected_mav = np.mean(np.abs(self.signal))
        
        # Verifica se o resultado está correto
        self.assertAlmostEqual(mav, expected_mav, places=6)
    
    def test_calculate_variance(self):
        """Testa o cálculo da variância."""
        # Calcula a variância
        var = feature_extraction.calculate_variance(self.signal)
        
        # Calcula a variância manualmente
        expected_var = np.var(self.signal)
        
        # Verifica se o resultado está correto
        self.assertAlmostEqual(var, expected_var, places=6)
    
    def test_calculate_waveform_length(self):
        """Testa o cálculo do comprimento da forma de onda (WL)."""
        # Calcula o WL
        wl = feature_extraction.calculate_waveform_length(self.signal)
        
        # Calcula o WL manualmente
        expected_wl = np.sum(np.abs(np.diff(self.signal)))
        
        # Verifica se o resultado está correto
        self.assertAlmostEqual(wl, expected_wl, places=6)
    
    def test_calculate_zero_crossings(self):
        """Testa o cálculo do número de cruzamentos por zero (ZC)."""
        # Calcula o ZC
        zc = feature_extraction.calculate_zero_crossings(self.signal)
        
        # Para um sinal senoidal de 10 Hz em 1 segundo, esperamos cerca de 20 cruzamentos por zero
        # (10 ciclos completos, cada um com 2 cruzamentos)
        self.assertGreaterEqual(zc, 18)
        self.assertLessEqual(zc, 22)
    
    def test_extract_features(self):
        """Testa a extração de múltiplas características."""
        # Extrai todas as características
        features = feature_extraction.extract_features(self.signal)
        
        # Verifica se todas as características esperadas estão presentes
        expected_features = ['rms', 'mav', 'var', 'wl', 'zc', 'ssc', 'wamp']
        for feature in expected_features:
            self.assertIn(feature, features)
        
        # Verifica se os valores são numéricos
        for feature, value in features.items():
            self.assertIsInstance(value, (int, float))

class TestWindowing(unittest.TestCase):
    """Testes para o módulo de janelamento."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        # Cria um sinal de teste
        self.t = np.linspace(0, 1, 1000)
        self.signal = np.sin(2 * np.pi * 10 * self.t)  # Sinal de 10 Hz
    
    def test_apply_window(self):
        """Testa a aplicação de uma janela."""
        # Aplica uma janela Hamming
        windowed_signal = windowing.apply_window(self.signal, window_type='hamming')
        
        # Verifica se o tamanho do sinal não mudou
        self.assertEqual(len(windowed_signal), len(self.signal))
        
        # Verifica se as extremidades foram atenuadas
        # Usamos o meio do sinal como referência para garantir que as bordas sejam menores
        mid_point = len(self.signal) // 2
        self.assertLessEqual(abs(windowed_signal[0]), abs(windowed_signal[mid_point]))
        self.assertLessEqual(abs(windowed_signal[-1]), abs(windowed_signal[mid_point]))
    
    def test_segment_signal(self):
        """Testa a segmentação do sinal em janelas."""
        # Segmenta o sinal em janelas de 100 amostras com sobreposição de 50%
        windows = windowing.segment_signal(self.signal, window_size=100, overlap=0.5)
        
        # Verifica se o número de janelas está correto
        expected_num_windows = 1 + int((len(self.signal) - 100) / (100 * (1 - 0.5)))
        self.assertEqual(len(windows), expected_num_windows)
        
        # Verifica se o tamanho das janelas está correto
        for window in windows:
            self.assertEqual(len(window), 100)
        
        # Verifica se a sobreposição está correta
        for i in range(len(windows) - 1):
            overlap_samples = np.sum(windows[i][50:] == windows[i+1][:50])
            self.assertEqual(overlap_samples, 50)
    
    def test_process_window(self):
        """Testa o processamento completo de uma janela."""
        # Processa uma janela
        window_data = windowing.process_window(self.signal)
        
        # Verifica se o resultado contém os campos esperados
        self.assertIn('signal', window_data)
        self.assertIn('filtered_signal', window_data)
        self.assertIn('features', window_data)
        
        # Verifica se o sinal filtrado tem o mesmo tamanho do original
        self.assertEqual(len(window_data['filtered_signal']), len(self.signal))
        
        # Verifica se as características foram extraídas
        self.assertIsInstance(window_data['features'], dict)
        self.assertGreater(len(window_data['features']), 0)

if __name__ == '__main__':
    unittest.main()
