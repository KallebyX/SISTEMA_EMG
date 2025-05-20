"""
Gerador de sinais EMG sintéticos baseados em bancos de dados públicos.

Este módulo é responsável pela geração de sinais EMG sintéticos para o modo simulado,
utilizando dados de bancos públicos como Ninapro, EMG-UKA e PhysioNet.
"""

import os
import numpy as np
import pandas as pd
import random
import logging
import time
from threading import Thread, Lock

logger = logging.getLogger(__name__)

class SyntheticGenerator:
    """
    Classe para geração de sinais EMG sintéticos.
    
    Esta classe gera sinais EMG sintéticos baseados em bancos de dados públicos,
    permitindo a simulação de diferentes gestos e movimentos para teste e desenvolvimento.
    """
    
    def __init__(self, data_dir=None, sampling_rate=1000, window_size=200, noise_level=0.05):
        """
        Inicializa o gerador de sinais sintéticos.
        
        Args:
            data_dir (str, optional): Diretório contendo os dados dos bancos públicos.
                Se None, usa o diretório padrão 'data/raw'.
            sampling_rate (int, optional): Taxa de amostragem em Hz. Padrão é 1000 Hz.
            window_size (int, optional): Tamanho da janela de análise em amostras. Padrão é 200.
            noise_level (float, optional): Nível de ruído a ser adicionado (0.0 a 1.0). Padrão é 0.05.
        """
        # Define o diretório de dados
        if data_dir is None:
            # Obtém o diretório do script atual
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Sobe dois níveis para chegar à raiz do projeto
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
            self.data_dir = os.path.join(project_root, 'data', 'raw')
        else:
            self.data_dir = data_dir
        
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.noise_level = noise_level
        
        # Dicionário para armazenar os templates de gestos
        self.gesture_templates = {
            'rest': [],
            'open': [],
            'close': [],
            'pinch': [],
            'point': []
        }
        
        # Estado atual da simulação
        self.current_gesture = 'rest'
        self.transition_progress = 0.0
        self.previous_gesture = 'rest'
        
        # Controle de threads
        self.is_generating = False
        self.generate_thread = None
        self.buffer_lock = Lock()
        
        # Buffer para armazenar dados gerados
        self.buffer = np.zeros(window_size * 10)
        self.buffer_index = 0
        
        # Callbacks para notificação de novos dados
        self.data_callbacks = []
        
        # Carrega os templates de gestos
        self._load_templates()
    
    def _load_templates(self):
        """
        Carrega os templates de gestos a partir dos bancos de dados.
        Se os bancos não estiverem disponíveis, gera templates sintéticos.
        """
        # Verifica se os diretórios dos bancos existem
        ninapro_dir = os.path.join(self.data_dir, 'ninapro')
        emg_uka_dir = os.path.join(self.data_dir, 'emg_uka')
        physionet_dir = os.path.join(self.data_dir, 'physionet')
        
        # Flag para indicar se algum banco foi carregado
        loaded_from_db = False
        
        # Tenta carregar do Ninapro
        if os.path.exists(ninapro_dir) and os.listdir(ninapro_dir):
            try:
                self._load_from_ninapro(ninapro_dir)
                loaded_from_db = True
                logger.info("Templates carregados do banco Ninapro")
            except Exception as e:
                logger.error(f"Erro ao carregar do Ninapro: {str(e)}")
        
        # Tenta carregar do EMG-UKA
        if os.path.exists(emg_uka_dir) and os.listdir(emg_uka_dir):
            try:
                self._load_from_emg_uka(emg_uka_dir)
                loaded_from_db = True
                logger.info("Templates carregados do banco EMG-UKA")
            except Exception as e:
                logger.error(f"Erro ao carregar do EMG-UKA: {str(e)}")
        
        # Tenta carregar do PhysioNet
        if os.path.exists(physionet_dir) and os.listdir(physionet_dir):
            try:
                self._load_from_physionet(physionet_dir)
                loaded_from_db = True
                logger.info("Templates carregados do banco PhysioNet")
            except Exception as e:
                logger.error(f"Erro ao carregar do PhysioNet: {str(e)}")
        
        # Se nenhum banco foi carregado, gera templates sintéticos
        if not loaded_from_db:
            self._generate_synthetic_templates()
            logger.info("Templates sintéticos gerados (bancos de dados não encontrados)")
    
    def _load_from_ninapro(self, ninapro_dir):
        """
        Carrega templates do banco Ninapro.
        
        Args:
            ninapro_dir (str): Diretório contendo os dados do Ninapro.
        """
        # Implementação simplificada - em um sistema real, seria necessário
        # conhecer a estrutura específica dos arquivos do Ninapro
        
        # Mapeamento de gestos do Ninapro para nossos gestos
        gesture_mapping = {
            0: 'rest',
            1: 'open',
            2: 'close',
            3: 'pinch',
            4: 'point'
        }
        
        # Procura por arquivos CSV no diretório
        for filename in os.listdir(ninapro_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(ninapro_dir, filename)
                try:
                    # Carrega o arquivo CSV
                    df = pd.read_csv(file_path)
                    
                    # Verifica se o arquivo tem as colunas esperadas
                    if 'emg' in df.columns and 'gesture' in df.columns:
                        # Agrupa por gesto
                        for gesture_id, group in df.groupby('gesture'):
                            if gesture_id in gesture_mapping:
                                gesture_name = gesture_mapping[gesture_id]
                                
                                # Extrai segmentos de EMG para o gesto
                                emg_data = group['emg'].values
                                
                                # Divide em segmentos do tamanho da janela
                                for i in range(0, len(emg_data) - self.window_size, self.window_size):
                                    segment = emg_data[i:i+self.window_size]
                                    self.gesture_templates[gesture_name].append(segment)
                
                except Exception as e:
                    logger.error(f"Erro ao processar arquivo {filename}: {str(e)}")
    
    def _load_from_emg_uka(self, emg_uka_dir):
        """
        Carrega templates do banco EMG-UKA.
        
        Args:
            emg_uka_dir (str): Diretório contendo os dados do EMG-UKA.
        """
        # Implementação simplificada - em um sistema real, seria necessário
        # conhecer a estrutura específica dos arquivos do EMG-UKA
        
        # Mapeamento de gestos do EMG-UKA para nossos gestos
        gesture_mapping = {
            'rest': 'rest',
            'open_hand': 'open',
            'close_hand': 'close',
            'pinch': 'pinch',
            'pointing': 'point'
        }
        
        # Procura por arquivos CSV no diretório
        for filename in os.listdir(emg_uka_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(emg_uka_dir, filename)
                try:
                    # Carrega o arquivo CSV
                    df = pd.read_csv(file_path)
                    
                    # Verifica se o arquivo tem as colunas esperadas
                    if 'emg_signal' in df.columns and 'gesture' in df.columns:
                        # Agrupa por gesto
                        for gesture_id, group in df.groupby('gesture'):
                            if gesture_id in gesture_mapping:
                                gesture_name = gesture_mapping[gesture_id]
                                
                                # Extrai segmentos de EMG para o gesto
                                emg_data = group['emg_signal'].values
                                
                                # Divide em segmentos do tamanho da janela
                                for i in range(0, len(emg_data) - self.window_size, self.window_size):
                                    segment = emg_data[i:i+self.window_size]
                                    self.gesture_templates[gesture_name].append(segment)
                
                except Exception as e:
                    logger.error(f"Erro ao processar arquivo {filename}: {str(e)}")
    
    def _load_from_physionet(self, physionet_dir):
        """
        Carrega templates do banco PhysioNet.
        
        Args:
            physionet_dir (str): Diretório contendo os dados do PhysioNet.
        """
        # Implementação simplificada - em um sistema real, seria necessário
        # conhecer a estrutura específica dos arquivos do PhysioNet
        
        # Procura por arquivos de dados no diretório
        for filename in os.listdir(physionet_dir):
            if filename.endswith('.dat') or filename.endswith('.txt'):
                file_path = os.path.join(physionet_dir, filename)
                try:
                    # Carrega o arquivo
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Extrai os valores numéricos
                    values = []
                    for line in lines:
                        try:
                            # Tenta converter cada linha para float
                            value = float(line.strip())
                            values.append(value)
                        except ValueError:
                            # Ignora linhas que não podem ser convertidas
                            continue
                    
                    # Se temos dados suficientes, cria templates
                    if len(values) >= self.window_size:
                        # Divide em segmentos do tamanho da janela
                        for i in range(0, len(values) - self.window_size, self.window_size):
                            segment = np.array(values[i:i+self.window_size])
                            
                            # Normaliza o segmento
                            if np.max(segment) != np.min(segment):
                                segment = (segment - np.min(segment)) / (np.max(segment) - np.min(segment))
                            
                            # Distribui aleatoriamente entre os gestos
                            gesture = random.choice(list(self.gesture_templates.keys()))
                            self.gesture_templates[gesture].append(segment)
                
                except Exception as e:
                    logger.error(f"Erro ao processar arquivo {filename}: {str(e)}")
    
    def _generate_synthetic_templates(self):
        """
        Gera templates sintéticos para cada gesto quando não há dados disponíveis.
        """
        # Parâmetros para geração de templates
        n_templates_per_gesture = 10
        
        # Gera templates para cada gesto
        for gesture in self.gesture_templates.keys():
            for _ in range(n_templates_per_gesture):
                # Gera um template sintético baseado no tipo de gesto
                if gesture == 'rest':
                    # Repouso: sinal de baixa amplitude
                    base = np.random.normal(0.1, 0.05, self.window_size)
                    
                elif gesture == 'open':
                    # Abertura: sinal com pico no início e decaimento
                    base = np.zeros(self.window_size)
                    peak_pos = random.randint(10, 30)
                    peak_width = random.randint(20, 40)
                    peak_height = random.uniform(0.6, 0.9)
                    
                    # Cria um pico gaussiano
                    for i in range(self.window_size):
                        base[i] = peak_height * np.exp(-0.5 * ((i - peak_pos) / (peak_width / 2.5)) ** 2)
                    
                    # Adiciona componente de decaimento
                    decay = np.exp(-np.linspace(0, 5, self.window_size))
                    base = base * decay
                    
                elif gesture == 'close':
                    # Fechamento: sinal com aumento gradual e platô
                    base = np.zeros(self.window_size)
                    rise_duration = random.randint(30, 50)
                    plateau_level = random.uniform(0.7, 0.9)
                    
                    # Cria rampa de subida
                    for i in range(rise_duration):
                        base[i] = (i / rise_duration) * plateau_level
                    
                    # Cria platô
                    base[rise_duration:] = plateau_level
                    
                    # Adiciona oscilações
                    oscillation = 0.1 * np.sin(np.linspace(0, 10 * np.pi, self.window_size))
                    base = base + oscillation
                    
                elif gesture == 'pinch':
                    # Pinça: sinal com dois picos próximos
                    base = np.zeros(self.window_size)
                    peak1_pos = random.randint(30, 50)
                    peak2_pos = peak1_pos + random.randint(20, 40)
                    peak_width = random.randint(15, 25)
                    peak_height = random.uniform(0.6, 0.8)
                    
                    # Cria dois picos gaussianos
                    for i in range(self.window_size):
                        if i < len(base):  # Garante que não ultrapasse o tamanho do array
                            base[i] = peak_height * np.exp(-0.5 * ((i - peak1_pos) / (peak_width / 2.5)) ** 2)
                            if peak2_pos < self.window_size:
                                base[i] += peak_height * 0.8 * np.exp(-0.5 * ((i - peak2_pos) / (peak_width / 2.5)) ** 2)
                    
                else:  # 'point'
                    # Apontar: sinal com um pico forte seguido de atividade sustentada
                    base = np.zeros(self.window_size)
                    peak_pos = random.randint(20, 40)
                    peak_width = random.randint(10, 20)
                    peak_height = random.uniform(0.8, 1.0)
                    sustain_level = random.uniform(0.3, 0.5)
                    
                    # Cria um pico gaussiano
                    for i in range(self.window_size):
                        base[i] = peak_height * np.exp(-0.5 * ((i - peak_pos) / (peak_width / 2.5)) ** 2)
                    
                    # Adiciona atividade sustentada após o pico
                    for i in range(peak_pos + peak_width, self.window_size):
                        base[i] = sustain_level
                    
                    # Adiciona oscilações na parte sustentada
                    oscillation = 0.1 * np.sin(np.linspace(0, 15 * np.pi, self.window_size))
                    for i in range(peak_pos + peak_width, self.window_size):
                        if i < len(base):  # Garante que não ultrapasse o tamanho do array
                            base[i] += oscillation[i]
                
                # Adiciona ruído ao template
                noise = np.random.normal(0, 0.05, self.window_size)
                template = base + noise
                
                # Normaliza o template
                if np.max(template) != np.min(template):
                    template = (template - np.min(template)) / (np.max(template) - np.min(template))
                
                # Adiciona o template à lista do gesto correspondente
                self.gesture_templates[gesture].append(template)
    
    def start_generation(self):
        """
        Inicia a geração contínua de sinais EMG sintéticos.
        
        Returns:
            bool: True se a geração foi iniciada com sucesso, False caso contrário.
        """
        if self.is_generating:
            logger.warning("Geração já está em andamento")
            return True
        
        # Verifica se temos templates para todos os gestos
        for gesture, templates in self.gesture_templates.items():
            if not templates:
                logger.error(f"Não há templates disponíveis para o gesto '{gesture}'")
                return False
        
        self.is_generating = True
        self.generate_thread = Thread(target=self._generate_loop)
        self.generate_thread.daemon = True
        self.generate_thread.start()
        
        logger.info("Geração de sinais EMG sintéticos iniciada")
        return True
    
    def stop_generation(self):
        """
        Interrompe a geração contínua de sinais EMG sintéticos.
        """
        if not self.is_generating:
            logger.warning("Geração não está em andamento")
            return
        
        self.is_generating = False
        if self.generate_thread:
            self.generate_thread.join(timeout=1.0)
            self.generate_thread = None
        
        logger.info("Geração de sinais EMG sintéticos interrompida")
    
    def _generate_loop(self):
        """
        Loop de geração contínua de sinais EMG sintéticos.
        Este método é executado em uma thread separada.
        """
        samples_per_iteration = 10  # Número de amostras geradas por iteração
        
        while self.is_generating:
            # Gera novas amostras
            new_samples = self._generate_samples(samples_per_iteration)
            
            # Adiciona as novas amostras ao buffer
            with self.buffer_lock:
                # Desloca o buffer para abrir espaço para as novas amostras
                self.buffer = np.roll(self.buffer, -len(new_samples))
                # Adiciona as novas amostras ao final do buffer
                self.buffer[-len(new_samples):] = new_samples
                self.buffer_index = (self.buffer_index + len(new_samples)) % len(self.buffer)
            
            # Notifica os callbacks
            for callback in self.data_callbacks:
                try:
                    callback(new_samples)
                except Exception as e:
                    logger.error(f"Erro em callback de dados: {str(e)}")
            
            # Pequena pausa para simular a taxa de amostragem
            time.sleep(samples_per_iteration / self.sampling_rate)
    
    def _generate_samples(self, n_samples):
        """
        Gera um número específico de amostras EMG sintéticas.
        
        Args:
            n_samples (int): Número de amostras a serem geradas.
        
        Returns:
            numpy.ndarray: Array com as amostras geradas.
        """
        # Verifica se estamos em transição entre gestos
        if self.transition_progress > 0.0 and self.transition_progress < 1.0:
            # Avança a transição
            self.transition_progress += 0.01
            if self.transition_progress >= 1.0:
                self.transition_progress = 0.0
                self.previous_gesture = self.current_gesture
        
        # Seleciona um template para o gesto atual
        if self.gesture_templates[self.current_gesture]:
            current_template = random.choice(self.gesture_templates[self.current_gesture])
        else:
            # Fallback para repouso se não houver templates para o gesto atual
            current_template = random.choice(self.gesture_templates['rest'])
        
        # Se estamos em transição, mistura com o template do gesto anterior
        if self.transition_progress > 0.0 and self.transition_progress < 1.0:
            if self.gesture_templates[self.previous_gesture]:
                previous_template = random.choice(self.gesture_templates[self.previous_gesture])
            else:
                previous_template = random.choice(self.gesture_templates['rest'])
            
            # Interpola entre os templates
            template = previous_template * (1 - self.transition_progress) + current_template * self.transition_progress
        else:
            template = current_template
        
        # Seleciona um segmento aleatório do template
        if len(template) > n_samples:
            start_idx = random.randint(0, len(template) - n_samples)
            segment = template[start_idx:start_idx + n_samples]
        else:
            # Se o template for menor que n_samples, repete-o
            repeats = int(np.ceil(n_samples / len(template)))
            segment = np.tile(template, repeats)[:n_samples]
        
        # Adiciona ruído
        noise = np.random.normal(0, self.noise_level, n_samples)
        samples = segment + noise
        
        return samples
    
    def set_gesture(self, gesture, transition=True):
        """
        Define o gesto atual para geração de sinais.
        
        Args:
            gesture (str): Nome do gesto ('rest', 'open', 'close', 'pinch', 'point').
            transition (bool, optional): Se True, realiza transição suave entre gestos. Padrão é True.
        
        Returns:
            bool: True se o gesto foi definido com sucesso, False caso contrário.
        """
        if gesture not in self.gesture_templates:
            logger.error(f"Gesto inválido: {gesture}")
            return False
        
        if not self.gesture_templates[gesture]:
            logger.error(f"Não há templates disponíveis para o gesto '{gesture}'")
            return False
        
        if transition and gesture != self.current_gesture:
            self.previous_gesture = self.current_gesture
            self.transition_progress = 0.01
        else:
            self.transition_progress = 0.0
            self.previous_gesture = gesture
        
        self.current_gesture = gesture
        logger.info(f"Gesto atual definido para '{gesture}'")
        return True
    
    def get_latest_data(self, n_samples=100):
        """
        Obtém as amostras mais recentes do buffer.
        
        Args:
            n_samples (int, optional): Número de amostras a serem retornadas. Padrão é 100.
        
        Returns:
            numpy.ndarray: Array com as n_samples mais recentes.
        """
        n_samples = min(n_samples, len(self.buffer))
        
        with self.buffer_lock:
            if self.buffer_index >= n_samples:
                # Retorna as últimas n_samples do buffer
                return self.buffer[self.buffer_index - n_samples:self.buffer_index]
            else:
                # Retorna as amostras do final do buffer e do início
                return np.concatenate([
                    self.buffer[len(self.buffer) - (n_samples - self.buffer_index):],
                    self.buffer[:self.buffer_index]
                ])
    
    def add_data_callback(self, callback):
        """
        Adiciona um callback para ser notificado quando novos dados forem gerados.
        
        Args:
            callback (callable): Função a ser chamada com os novos dados como argumento.
        """
        self.data_callbacks.append(callback)
    
    def remove_data_callback(self, callback):
        """
        Remove um callback previamente adicionado.
        
        Args:
            callback (callable): Callback a ser removido.
        """
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
