"""
Interface de comunicação com Arduino para aquisição de sinais EMG.

Este módulo gerencia a comunicação serial com o Arduino conectado ao sensor MyoWare 2.0,
permitindo a aquisição de sinais EMG em tempo real.
"""

import time
import serial
import serial.tools.list_ports
import numpy as np
from threading import Thread, Lock
import logging

logger = logging.getLogger(__name__)

class ArduinoInterface:
    """
    Classe para gerenciar a comunicação com o Arduino para aquisição de sinais EMG.
    
    Esta classe estabelece uma conexão serial com o Arduino, configura os parâmetros
    de comunicação e gerencia a leitura contínua dos dados do sensor MyoWare 2.0.
    """
    
    def __init__(self, port=None, baudrate=115200, timeout=1.0, buffer_size=1000):
        """
        Inicializa a interface Arduino.
        
        Args:
            port (str, optional): Porta serial do Arduino. Se None, tenta detectar automaticamente.
            baudrate (int, optional): Taxa de transmissão. Padrão é 115200.
            timeout (float, optional): Timeout para operações de leitura em segundos. Padrão é 1.0.
            buffer_size (int, optional): Tamanho do buffer circular para armazenar amostras. Padrão é 1000.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.buffer_size = buffer_size
        
        self.serial_conn = None
        self.is_connected = False
        self.is_reading = False
        self.read_thread = None
        
        # Buffer circular para armazenar as amostras mais recentes
        self.buffer = np.zeros(buffer_size)
        self.buffer_index = 0
        self.buffer_lock = Lock()
        
        # Configurações para comunicação com a prótese
        self.last_command_time = 0
        self.command_timeout = 0.5  # Tempo mínimo entre comandos em segundos
    
    def detect_arduino(self):
        """
        Detecta automaticamente a porta do Arduino.
        
        Returns:
            str: Porta detectada ou None se não encontrada.
        """
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            # Procura por portas que contenham "Arduino" ou "CH340" (chip comum em clones)
            if "Arduino" in port.description or "CH340" in port.description:
                logger.info(f"Arduino detectado na porta {port.device}")
                return port.device
        
        # Se não encontrar especificamente Arduino, tenta a primeira porta disponível
        if ports:
            logger.info(f"Nenhum Arduino detectado especificamente. Usando primeira porta disponível: {ports[0].device}")
            return ports[0].device
        
        logger.warning("Nenhuma porta serial encontrada")
        return None
    
    def connect(self):
        """
        Estabelece conexão com o Arduino.
        
        Returns:
            bool: True se a conexão foi estabelecida com sucesso, False caso contrário.
        """
        if self.is_connected:
            logger.warning("Já conectado ao Arduino")
            return True
        
        if self.port is None:
            self.port = self.detect_arduino()
            if self.port is None:
                logger.error("Não foi possível detectar o Arduino")
                return False
        
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            
            # Aguarda inicialização do Arduino
            time.sleep(2.0)
            
            # Limpa o buffer de entrada
            self.serial_conn.reset_input_buffer()
            
            self.is_connected = True
            logger.info(f"Conectado ao Arduino na porta {self.port}")
            return True
            
        except serial.SerialException as e:
            logger.error(f"Erro ao conectar ao Arduino: {str(e)}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """
        Encerra a conexão com o Arduino.
        """
        if self.is_reading:
            self.stop_reading()
        
        if self.serial_conn and self.is_connected:
            self.serial_conn.close()
            self.is_connected = False
            logger.info("Desconectado do Arduino")
    
    def _read_loop(self):
        """
        Loop de leitura contínua dos dados do Arduino.
        Este método é executado em uma thread separada.
        """
        if not self.is_connected:
            logger.error("Tentativa de leitura sem conexão estabelecida")
            return
        
        logger.info("Iniciando loop de leitura de dados do Arduino")
        
        while self.is_reading:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    
                    # Processa a linha recebida
                    try:
                        # Espera-se que o Arduino envie apenas o valor numérico do sensor
                        value = float(line)
                        
                        # Armazena no buffer circular
                        with self.buffer_lock:
                            self.buffer[self.buffer_index] = value
                            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
                            
                    except ValueError:
                        # Ignora linhas que não podem ser convertidas para float
                        logger.debug(f"Valor ignorado: {line}")
                        continue
                        
            except Exception as e:
                logger.error(f"Erro na leitura do Arduino: {str(e)}")
                self.is_reading = False
                break
                
            # Pequena pausa para evitar uso excessivo de CPU
            time.sleep(0.001)
        
        logger.info("Loop de leitura de dados do Arduino finalizado")
    
    def start_reading(self):
        """
        Inicia a leitura contínua dos dados do Arduino em uma thread separada.
        
        Returns:
            bool: True se a leitura foi iniciada com sucesso, False caso contrário.
        """
        if not self.is_connected:
            success = self.connect()
            if not success:
                return False
        
        if self.is_reading:
            logger.warning("Leitura já está em andamento")
            return True
        
        self.is_reading = True
        self.read_thread = Thread(target=self._read_loop)
        self.read_thread.daemon = True
        self.read_thread.start()
        
        logger.info("Leitura de dados do Arduino iniciada")
        return True
    
    def stop_reading(self):
        """
        Interrompe a leitura contínua dos dados do Arduino.
        """
        if not self.is_reading:
            logger.warning("Leitura não está em andamento")
            return
        
        self.is_reading = False
        if self.read_thread:
            self.read_thread.join(timeout=1.0)
            self.read_thread = None
        
        logger.info("Leitura de dados do Arduino interrompida")
    
    def get_latest_data(self, n_samples=100):
        """
        Obtém as amostras mais recentes do buffer.
        
        Args:
            n_samples (int, optional): Número de amostras a serem retornadas. Padrão é 100.
                Se maior que o tamanho do buffer, retorna o buffer inteiro.
        
        Returns:
            numpy.ndarray: Array com as n_samples mais recentes.
        """
        n_samples = min(n_samples, self.buffer_size)
        
        with self.buffer_lock:
            if self.buffer_index >= n_samples:
                # Retorna as últimas n_samples do buffer
                return self.buffer[self.buffer_index - n_samples:self.buffer_index]
            else:
                # Retorna as amostras do final do buffer e do início
                return np.concatenate([
                    self.buffer[self.buffer_size - (n_samples - self.buffer_index):],
                    self.buffer[:self.buffer_index]
                ])
    
    def send_command(self, command):
        """
        Envia um comando para o Arduino.
        
        Args:
            command (str): Comando a ser enviado.
        
        Returns:
            bool: True se o comando foi enviado com sucesso, False caso contrário.
        """
        if not self.is_connected:
            logger.error("Tentativa de envio de comando sem conexão estabelecida")
            return False
        
        # Verifica timeout entre comandos para evitar sobrecarga
        current_time = time.time()
        if current_time - self.last_command_time < self.command_timeout:
            logger.warning(f"Comando ignorado devido ao timeout ({self.command_timeout}s)")
            return False
        
        try:
            # Adiciona quebra de linha ao final do comando
            command_bytes = (command + '\n').encode('utf-8')
            self.serial_conn.write(command_bytes)
            self.serial_conn.flush()
            
            self.last_command_time = current_time
            logger.info(f"Comando enviado: {command}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao enviar comando: {str(e)}")
            return False
    
    def send_motor_command(self, action):
        """
        Envia comando específico para controle do motor da prótese.
        
        Args:
            action (str): Ação do motor: "OPEN", "CLOSE" ou "STOP".
        
        Returns:
            bool: True se o comando foi enviado com sucesso, False caso contrário.
        """
        if action.upper() not in ["OPEN", "CLOSE", "STOP"]:
            logger.error(f"Comando de motor inválido: {action}")
            return False
        
        command = f"MOTOR_{action.upper()}"
        return self.send_command(command)
