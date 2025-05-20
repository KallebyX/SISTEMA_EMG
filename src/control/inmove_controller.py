"""
Módulo de controle para prótese INMOVE.

Este módulo contém implementações para controle da prótese INMOVE
através de comandos seriais baseados em predições de gestos.
"""

import time
import logging
import threading

logger = logging.getLogger(__name__)

class INMOVEController:
    """
    Classe para controle da prótese INMOVE.
    
    Esta classe gerencia o envio de comandos para a prótese INMOVE
    com base nas predições de gestos realizadas pelo modelo de ML.
    """
    
    def __init__(self, arduino_interface, confidence_threshold=0.8, 
                 command_timeout=0.5, safety_timeout=5.0):
        """
        Inicializa o controlador da prótese INMOVE.
        
        Args:
            arduino_interface: Interface Arduino para comunicação serial.
            confidence_threshold (float, optional): Limiar de confiança para aceitação de comandos.
                Padrão é 0.8.
            command_timeout (float, optional): Tempo mínimo entre comandos em segundos.
                Padrão é 0.5.
            safety_timeout (float, optional): Tempo máximo para manter um comando ativo sem
                novas confirmações. Padrão é 5.0.
        """
        self.arduino = arduino_interface
        self.confidence_threshold = confidence_threshold
        self.command_timeout = command_timeout
        self.safety_timeout = safety_timeout
        
        # Estado atual do controlador
        self.current_command = None
        self.current_confidence = 0.0
        self.last_command_time = 0.0
        self.is_active = False
        
        # Thread de segurança para timeout
        self.safety_thread = None
        self.safety_lock = threading.Lock()
    
    def start(self):
        """
        Inicia o controlador da prótese.
        
        Returns:
            bool: True se o controlador foi iniciado com sucesso, False caso contrário.
        """
        if self.is_active:
            logger.warning("Controlador já está ativo")
            return True
        
        # Verifica se a conexão com o Arduino está estabelecida
        if not self.arduino.is_connected:
            success = self.arduino.connect()
            if not success:
                logger.error("Falha ao conectar ao Arduino")
                return False
        
        # Inicia a thread de segurança
        self.is_active = True
        self.safety_thread = threading.Thread(target=self._safety_monitor)
        self.safety_thread.daemon = True
        self.safety_thread.start()
        
        logger.info("Controlador da prótese INMOVE iniciado")
        return True
    
    def stop(self):
        """
        Para o controlador da prótese.
        """
        if not self.is_active:
            logger.warning("Controlador não está ativo")
            return
        
        # Para a thread de segurança
        self.is_active = False
        if self.safety_thread:
            self.safety_thread.join(timeout=1.0)
            self.safety_thread = None
        
        # Envia comando de parada para a prótese
        self.send_command("STOP")
        
        logger.info("Controlador da prótese INMOVE parado")
    
    def _safety_monitor(self):
        """
        Thread de monitoramento de segurança.
        
        Esta thread verifica se um comando está ativo por muito tempo sem
        novas confirmações e envia um comando de parada se necessário.
        """
        while self.is_active:
            with self.safety_lock:
                if self.current_command and self.current_command != "STOP":
                    # Verifica se o comando está ativo por muito tempo
                    if time.time() - self.last_command_time > self.safety_timeout:
                        logger.warning(f"Timeout de segurança atingido para comando {self.current_command}")
                        self.send_command("STOP")
            
            # Pequena pausa para evitar uso excessivo de CPU
            time.sleep(0.1)
    
    def process_prediction(self, prediction_data):
        """
        Processa uma predição e envia o comando correspondente para a prótese.
        
        Args:
            prediction_data (dict): Dicionário com dados da predição.
                Deve conter pelo menos as chaves 'prediction' e 'confidence'.
        
        Returns:
            bool: True se um comando foi enviado, False caso contrário.
        """
        if not self.is_active:
            logger.warning("Controlador não está ativo")
            return False
        
        # Extrai predição e confiança
        prediction = prediction_data.get('prediction')
        confidence = prediction_data.get('confidence', 0.0)
        
        # Verifica se a predição é válida
        if prediction is None:
            # Se temos um comando ativo e a predição é None, para a prótese
            if self.current_command and self.current_command != "STOP":
                logger.info("Predição inválida, parando prótese")
                return self.send_command("STOP")
            return False
        
        # Mapeia a predição para um comando
        command = self._map_prediction_to_command(prediction)
        
        # Verifica se o comando é válido
        if not command:
            logger.warning(f"Não foi possível mapear predição {prediction} para um comando")
            return False
        
        # Verifica se a confiança é suficiente
        if confidence < self.confidence_threshold:
            logger.debug(f"Confiança insuficiente: {confidence:.4f} < {self.confidence_threshold:.4f}")
            return False
        
        # Verifica se é o mesmo comando atual e se o timeout já passou
        if command == self.current_command:
            if time.time() - self.last_command_time < self.command_timeout:
                logger.debug(f"Timeout entre comandos não atingido: {time.time() - self.last_command_time:.2f}s < {self.command_timeout:.2f}s")
                return False
        
        # Envia o comando
        return self.send_command(command)
    
    def send_command(self, command):
        """
        Envia um comando para a prótese.
        
        Args:
            command (str): Comando a ser enviado ('OPEN', 'CLOSE', 'STOP').
        
        Returns:
            bool: True se o comando foi enviado com sucesso, False caso contrário.
        """
        if not self.arduino.is_connected:
            logger.error("Arduino não está conectado")
            return False
        
        with self.safety_lock:
            # Envia o comando para o Arduino
            success = self.arduino.send_motor_command(command)
            
            if success:
                self.current_command = command
                self.last_command_time = time.time()
                logger.info(f"Comando enviado para a prótese: {command}")
            else:
                logger.error(f"Falha ao enviar comando {command} para a prótese")
            
            return success
    
    def _map_prediction_to_command(self, prediction):
        """
        Mapeia uma predição para um comando da prótese.
        
        Args:
            prediction: Predição do modelo (classe ou índice).
        
        Returns:
            str: Comando correspondente ('OPEN', 'CLOSE', 'STOP') ou None se não houver mapeamento.
        """
        # Mapeamento básico de predições para comandos
        # Este mapeamento deve ser ajustado conforme as classes do modelo
        mapping = {
            0: "STOP",    # Repouso
            1: "OPEN",    # Mão aberta
            2: "CLOSE",   # Mão fechada
            3: "STOP",    # Pinça (não suportado diretamente, usa STOP)
            4: "STOP",    # Apontar (não suportado diretamente, usa STOP)
            
            # Mapeamento por string (caso a predição seja uma string)
            "rest": "STOP",
            "open": "OPEN",
            "close": "CLOSE",
            "pinch": "STOP",
            "point": "STOP"
        }
        
        # Tenta obter o comando do mapeamento
        command = mapping.get(prediction)
        
        if command is None:
            logger.warning(f"Predição desconhecida: {prediction}")
        
        return command
    
    def set_confidence_threshold(self, threshold):
        """
        Define o limiar de confiança para aceitação de comandos.
        
        Args:
            threshold (float): Novo limiar de confiança (0.0 a 1.0).
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Limiar de confiança definido para {self.confidence_threshold}")
    
    def set_command_timeout(self, timeout):
        """
        Define o tempo mínimo entre comandos.
        
        Args:
            timeout (float): Novo timeout em segundos.
        """
        self.command_timeout = max(0.0, timeout)
        logger.info(f"Timeout entre comandos definido para {self.command_timeout} segundos")
    
    def set_safety_timeout(self, timeout):
        """
        Define o tempo máximo para manter um comando ativo sem novas confirmações.
        
        Args:
            timeout (float): Novo timeout em segundos.
        """
        self.safety_timeout = max(1.0, timeout)
        logger.info(f"Timeout de segurança definido para {self.safety_timeout} segundos")
