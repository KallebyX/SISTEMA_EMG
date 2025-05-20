"""
Módulo para leitura e processamento de sinais EMG.

Este módulo é responsável pela leitura de sinais EMG a partir da interface Arduino
e pelo pré-processamento básico desses sinais.
"""

import numpy as np
import time
import logging
from threading import Thread, Lock

logger = logging.getLogger(__name__)

class EMGReader:
    """
    Classe para leitura e processamento de sinais EMG.
    
    Esta classe gerencia a leitura contínua de sinais EMG a partir da interface Arduino,
    realiza pré-processamento básico e disponibiliza os dados para outros módulos.
    """
    
    def __init__(self, arduino_interface, sampling_rate=1000, window_size=200, window_overlap=0.5):
        """
        Inicializa o leitor de sinais EMG.
        
        Args:
            arduino_interface: Instância de ArduinoInterface para comunicação com o hardware.
            sampling_rate (int, optional): Taxa de amostragem em Hz. Padrão é 1000 Hz.
            window_size (int, optional): Tamanho da janela de análise em amostras. Padrão é 200.
            window_overlap (float, optional): Sobreposição entre janelas consecutivas (0.0 a 1.0). Padrão é 0.5.
        """
        self.arduino = arduino_interface
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.window_overlap = window_overlap
        
        # Calcula o passo entre janelas consecutivas
        self.window_step = int(window_size * (1 - window_overlap))
        
        # Buffer para armazenar dados brutos
        self.raw_buffer = np.zeros(window_size * 10)  # Buffer maior para histórico
        self.raw_buffer_index = 0
        self.raw_buffer_lock = Lock()
        
        # Buffer para armazenar janelas processadas
        self.processed_windows = []
        self.processed_windows_lock = Lock()
        
        # Controle de threads
        self.is_processing = False
        self.process_thread = None
        
        # Callbacks para notificação de novas janelas
        self.window_callbacks = []
    
    def start(self):
        """
        Inicia a leitura e processamento de sinais EMG.
        
        Returns:
            bool: True se iniciado com sucesso, False caso contrário.
        """
        # Inicia a leitura de dados do Arduino
        if not self.arduino.is_reading:
            success = self.arduino.start_reading()
            if not success:
                logger.error("Falha ao iniciar leitura do Arduino")
                return False
        
        # Inicia o processamento em thread separada
        if not self.is_processing:
            self.is_processing = True
            self.process_thread = Thread(target=self._process_loop)
            self.process_thread.daemon = True
            self.process_thread.start()
            logger.info("Processamento de sinais EMG iniciado")
            return True
        else:
            logger.warning("Processamento já está em andamento")
            return True
    
    def stop(self):
        """
        Interrompe a leitura e processamento de sinais EMG.
        """
        self.is_processing = False
        if self.process_thread:
            self.process_thread.join(timeout=1.0)
            self.process_thread = None
        
        logger.info("Processamento de sinais EMG interrompido")
    
    def _process_loop(self):
        """
        Loop de processamento contínuo dos sinais EMG.
        Este método é executado em uma thread separada.
        """
        last_window_time = time.time()
        samples_since_last_window = 0
        
        while self.is_processing:
            # Obtém os dados mais recentes do Arduino
            new_data = self.arduino.get_latest_data(n_samples=self.window_step)
            
            if len(new_data) > 0:
                # Adiciona os novos dados ao buffer
                with self.raw_buffer_lock:
                    # Desloca o buffer para abrir espaço para os novos dados
                    self.raw_buffer = np.roll(self.raw_buffer, -len(new_data))
                    # Adiciona os novos dados ao final do buffer
                    self.raw_buffer[-len(new_data):] = new_data
                    samples_since_last_window += len(new_data)
                
                # Verifica se é hora de processar uma nova janela
                if samples_since_last_window >= self.window_step:
                    # Extrai a janela atual
                    with self.raw_buffer_lock:
                        current_window = self.raw_buffer[-self.window_size:].copy()
                    
                    # Processa a janela
                    processed_window = self._process_window(current_window)
                    
                    # Armazena a janela processada
                    with self.processed_windows_lock:
                        self.processed_windows.append(processed_window)
                        # Limita o número de janelas armazenadas
                        if len(self.processed_windows) > 100:
                            self.processed_windows = self.processed_windows[-100:]
                    
                    # Notifica os callbacks
                    for callback in self.window_callbacks:
                        try:
                            callback(processed_window)
                        except Exception as e:
                            logger.error(f"Erro em callback de janela: {str(e)}")
                    
                    # Reinicia o contador
                    samples_since_last_window = 0
                    last_window_time = time.time()
            
            # Pequena pausa para evitar uso excessivo de CPU
            time.sleep(0.01)
    
    def _process_window(self, window):
        """
        Processa uma janela de dados EMG.
        
        Args:
            window (numpy.ndarray): Janela de dados brutos.
        
        Returns:
            dict: Dicionário contendo a janela original e características extraídas.
        """
        # Cria um dicionário para armazenar a janela e suas características
        processed = {
            'raw': window,
            'timestamp': time.time(),
            'features': {}
        }
        
        # Extrai características básicas
        processed['features']['mean'] = np.mean(window)
        processed['features']['std'] = np.std(window)
        processed['features']['min'] = np.min(window)
        processed['features']['max'] = np.max(window)
        processed['features']['rms'] = np.sqrt(np.mean(np.square(window)))
        
        # Calcula a energia do sinal
        processed['features']['energy'] = np.sum(np.square(window))
        
        # Calcula o número de cruzamentos por zero
        zero_crossings = np.where(np.diff(np.signbit(window)))[0]
        processed['features']['zero_crossings'] = len(zero_crossings)
        
        # Calcula a assimetria (skewness)
        if processed['features']['std'] > 0:
            processed['features']['skewness'] = np.mean(((window - processed['features']['mean']) / processed['features']['std']) ** 3)
        else:
            processed['features']['skewness'] = 0
        
        return processed
    
    def get_latest_window(self):
        """
        Obtém a janela processada mais recente.
        
        Returns:
            dict: Janela processada mais recente ou None se não houver janelas.
        """
        with self.processed_windows_lock:
            if len(self.processed_windows) > 0:
                return self.processed_windows[-1]
            else:
                return None
    
    def get_latest_windows(self, n_windows=10):
        """
        Obtém as janelas processadas mais recentes.
        
        Args:
            n_windows (int, optional): Número de janelas a serem retornadas. Padrão é 10.
        
        Returns:
            list: Lista com as n_windows mais recentes.
        """
        with self.processed_windows_lock:
            return self.processed_windows[-n_windows:]
    
    def add_window_callback(self, callback):
        """
        Adiciona um callback para ser notificado quando uma nova janela for processada.
        
        Args:
            callback (callable): Função a ser chamada com a janela processada como argumento.
        """
        self.window_callbacks.append(callback)
    
    def remove_window_callback(self, callback):
        """
        Remove um callback previamente adicionado.
        
        Args:
            callback (callable): Callback a ser removido.
        """
        if callback in self.window_callbacks:
            self.window_callbacks.remove(callback)
    
    def get_raw_data(self, n_samples=1000):
        """
        Obtém os dados brutos mais recentes.
        
        Args:
            n_samples (int, optional): Número de amostras a serem retornadas. Padrão é 1000.
        
        Returns:
            numpy.ndarray: Array com as n_samples mais recentes.
        """
        with self.raw_buffer_lock:
            return self.raw_buffer[-n_samples:].copy()
