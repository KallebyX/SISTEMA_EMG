"""
Módulo de controle para prótese virtual.

Este módulo contém implementações para controle da prótese virtual
através de comandos baseados em predições de gestos.
"""

import time
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)

class VirtualController:
    """
    Classe para controle da prótese virtual.
    
    Esta classe gerencia o estado da prótese virtual com base nas
    predições de gestos realizadas pelo modelo de ML.
    """
    
    def __init__(self, confidence_threshold=0.7, smoothing_window=3):
        """
        Inicializa o controlador da prótese virtual.
        
        Args:
            confidence_threshold (float, optional): Limiar de confiança para aceitação de comandos.
                Padrão é 0.7.
            smoothing_window (int, optional): Tamanho da janela para suavização de movimentos.
                Padrão é 3.
        """
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        
        # Estado atual da prótese virtual
        self.current_gesture = "rest"
        self.target_gesture = "rest"
        self.transition_progress = 0.0
        self.transition_speed = 0.1  # Velocidade de transição entre gestos
        
        # Posições dos dedos (0.0 = fechado, 1.0 = aberto)
        self.finger_positions = {
            "thumb": 0.0,
            "index": 0.0,
            "middle": 0.0,
            "ring": 0.0,
            "pinky": 0.0
        }
        
        # Posições alvo para cada gesto
        self.gesture_positions = {
            "rest": {
                "thumb": 0.3,
                "index": 0.3,
                "middle": 0.3,
                "ring": 0.3,
                "pinky": 0.3
            },
            "open": {
                "thumb": 1.0,
                "index": 1.0,
                "middle": 1.0,
                "ring": 1.0,
                "pinky": 1.0
            },
            "close": {
                "thumb": 0.0,
                "index": 0.0,
                "middle": 0.0,
                "ring": 0.0,
                "pinky": 0.0
            },
            "pinch": {
                "thumb": 0.7,
                "index": 0.7,
                "middle": 0.0,
                "ring": 0.0,
                "pinky": 0.0
            },
            "point": {
                "thumb": 0.0,
                "index": 1.0,
                "middle": 0.0,
                "ring": 0.0,
                "pinky": 0.0
            }
        }
        
        # Histórico de predições para suavização
        self.prediction_history = []
        
        # Estado do controlador
        self.is_active = False
        self.update_thread = None
        self.update_lock = threading.Lock()
        
        # Callbacks para notificação de atualizações
        self.update_callbacks = []
    
    def start(self):
        """
        Inicia o controlador da prótese virtual.
        
        Returns:
            bool: True se o controlador foi iniciado com sucesso, False caso contrário.
        """
        if self.is_active:
            logger.warning("Controlador já está ativo")
            return True
        
        # Inicia a thread de atualização
        self.is_active = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info("Controlador da prótese virtual iniciado")
        return True
    
    def stop(self):
        """
        Para o controlador da prótese virtual.
        """
        if not self.is_active:
            logger.warning("Controlador não está ativo")
            return
        
        # Para a thread de atualização
        self.is_active = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
            self.update_thread = None
        
        logger.info("Controlador da prótese virtual parado")
    
    def _update_loop(self):
        """
        Thread de atualização contínua da prótese virtual.
        
        Esta thread atualiza as posições dos dedos com base no gesto alvo
        e na velocidade de transição.
        """
        update_interval = 0.03  # ~30 FPS
        
        while self.is_active:
            with self.update_lock:
                # Atualiza a transição entre gestos
                if self.current_gesture != self.target_gesture:
                    self.transition_progress += self.transition_speed
                    
                    if self.transition_progress >= 1.0:
                        self.transition_progress = 0.0
                        self.current_gesture = self.target_gesture
                
                # Atualiza as posições dos dedos
                self._update_finger_positions()
                
                # Notifica os callbacks
                self._notify_update_callbacks()
            
            # Pequena pausa para controlar a taxa de atualização
            time.sleep(update_interval)
    
    def _update_finger_positions(self):
        """
        Atualiza as posições dos dedos com base no gesto atual e no gesto alvo.
        """
        if self.current_gesture == self.target_gesture:
            # Se não estamos em transição, usa diretamente as posições do gesto atual
            target_positions = self.gesture_positions[self.current_gesture]
            
            for finger, target in target_positions.items():
                self.finger_positions[finger] = target
        else:
            # Se estamos em transição, interpola entre os gestos
            current_positions = self.gesture_positions[self.current_gesture]
            target_positions = self.gesture_positions[self.target_gesture]
            
            for finger in self.finger_positions:
                current = current_positions[finger]
                target = target_positions[finger]
                
                # Interpolação linear
                self.finger_positions[finger] = current + self.transition_progress * (target - current)
    
    def process_prediction(self, prediction_data):
        """
        Processa uma predição e atualiza o gesto alvo da prótese virtual.
        
        Args:
            prediction_data (dict): Dicionário com dados da predição.
                Deve conter pelo menos as chaves 'prediction' e 'confidence'.
        
        Returns:
            bool: True se o gesto alvo foi atualizado, False caso contrário.
        """
        if not self.is_active:
            logger.warning("Controlador não está ativo")
            return False
        
        # Extrai predição e confiança
        prediction = prediction_data.get('prediction')
        confidence = prediction_data.get('confidence', 0.0)
        
        # Verifica se a predição é válida
        if prediction is None:
            return False
        
        # Adiciona a predição ao histórico
        self.prediction_history.append((prediction, confidence))
        if len(self.prediction_history) > self.smoothing_window:
            self.prediction_history.pop(0)
        
        # Realiza suavização das predições
        smoothed_prediction, smoothed_confidence = self._smooth_predictions()
        
        # Verifica se a confiança é suficiente
        if smoothed_confidence < self.confidence_threshold:
            logger.debug(f"Confiança insuficiente: {smoothed_confidence:.4f} < {self.confidence_threshold:.4f}")
            return False
        
        # Mapeia a predição para um gesto
        gesture = self._map_prediction_to_gesture(smoothed_prediction)
        
        # Verifica se o gesto é válido
        if not gesture:
            logger.warning(f"Não foi possível mapear predição {smoothed_prediction} para um gesto")
            return False
        
        # Atualiza o gesto alvo
        with self.update_lock:
            if gesture != self.target_gesture:
                self.target_gesture = gesture
                logger.info(f"Gesto alvo atualizado para: {gesture}")
                return True
        
        return False
    
    def _smooth_predictions(self):
        """
        Realiza suavização das predições usando o histórico.
        
        Returns:
            tuple: (predição suavizada, confiança suavizada)
        """
        if not self.prediction_history:
            return None, 0.0
        
        # Conta as ocorrências de cada classe
        predictions = [p for p, _ in self.prediction_history]
        confidences = [c for _, c in self.prediction_history]
        
        # Se todas as predições são iguais, retorna essa predição
        if len(set(predictions)) == 1:
            return predictions[0], np.mean(confidences)
        
        # Caso contrário, faz uma votação ponderada pela confiança
        prediction_counts = {}
        for pred, conf in self.prediction_history:
            if pred not in prediction_counts:
                prediction_counts[pred] = 0
            prediction_counts[pred] += conf
        
        # Encontra a classe com maior contagem ponderada
        smoothed_prediction = max(prediction_counts, key=prediction_counts.get)
        
        # Calcula a confiança média para essa classe
        class_confidences = [conf for pred, conf in self.prediction_history if pred == smoothed_prediction]
        smoothed_confidence = np.mean(class_confidences) if class_confidences else 0.0
        
        return smoothed_prediction, smoothed_confidence
    
    def _map_prediction_to_gesture(self, prediction):
        """
        Mapeia uma predição para um gesto da prótese virtual.
        
        Args:
            prediction: Predição do modelo (classe ou índice).
        
        Returns:
            str: Gesto correspondente ('rest', 'open', 'close', 'pinch', 'point') ou None se não houver mapeamento.
        """
        # Mapeamento básico de predições para gestos
        # Este mapeamento deve ser ajustado conforme as classes do modelo
        mapping = {
            0: "rest",    # Repouso
            1: "open",    # Mão aberta
            2: "close",   # Mão fechada
            3: "pinch",   # Pinça
            4: "point",   # Apontar
            
            # Mapeamento por string (caso a predição seja uma string)
            "rest": "rest",
            "open": "open",
            "close": "close",
            "pinch": "pinch",
            "point": "point"
        }
        
        # Tenta obter o gesto do mapeamento
        gesture = mapping.get(prediction)
        
        if gesture is None:
            logger.warning(f"Predição desconhecida: {prediction}")
        
        return gesture
    
    def set_gesture(self, gesture):
        """
        Define diretamente o gesto da prótese virtual.
        
        Args:
            gesture (str): Gesto a ser definido ('rest', 'open', 'close', 'pinch', 'point').
        
        Returns:
            bool: True se o gesto foi definido com sucesso, False caso contrário.
        """
        if gesture not in self.gesture_positions:
            logger.error(f"Gesto inválido: {gesture}")
            return False
        
        with self.update_lock:
            if gesture != self.target_gesture:
                self.target_gesture = gesture
                logger.info(f"Gesto definido para: {gesture}")
                return True
        
        return False
    
    def get_finger_positions(self):
        """
        Obtém as posições atuais dos dedos.
        
        Returns:
            dict: Dicionário com as posições dos dedos.
        """
        with self.update_lock:
            return self.finger_positions.copy()
    
    def get_current_gesture(self):
        """
        Obtém o gesto atual da prótese virtual.
        
        Returns:
            str: Gesto atual.
        """
        with self.update_lock:
            return self.current_gesture
    
    def get_target_gesture(self):
        """
        Obtém o gesto alvo da prótese virtual.
        
        Returns:
            str: Gesto alvo.
        """
        with self.update_lock:
            return self.target_gesture
    
    def set_confidence_threshold(self, threshold):
        """
        Define o limiar de confiança para aceitação de gestos.
        
        Args:
            threshold (float): Novo limiar de confiança (0.0 a 1.0).
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Limiar de confiança definido para {self.confidence_threshold}")
    
    def set_smoothing_window(self, window_size):
        """
        Define o tamanho da janela para suavização de gestos.
        
        Args:
            window_size (int): Novo tamanho da janela.
        """
        if window_size < 1:
            logger.error("Tamanho da janela deve ser pelo menos 1")
            return
        
        self.smoothing_window = window_size
        
        # Ajusta o histórico de predições
        if len(self.prediction_history) > self.smoothing_window:
            self.prediction_history = self.prediction_history[-self.smoothing_window:]
        
        logger.info(f"Janela de suavização definida para {self.smoothing_window}")
    
    def set_transition_speed(self, speed):
        """
        Define a velocidade de transição entre gestos.
        
        Args:
            speed (float): Nova velocidade de transição (0.0 a 1.0).
        """
        self.transition_speed = max(0.01, min(1.0, speed))
        logger.info(f"Velocidade de transição definida para {self.transition_speed}")
    
    def add_update_callback(self, callback):
        """
        Adiciona um callback para ser notificado quando a prótese virtual for atualizada.
        
        Args:
            callback (callable): Função a ser chamada com o estado atual da prótese como argumento.
        """
        self.update_callbacks.append(callback)
    
    def remove_update_callback(self, callback):
        """
        Remove um callback previamente adicionado.
        
        Args:
            callback (callable): Callback a ser removido.
        """
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)
    
    def _notify_update_callbacks(self):
        """
        Notifica os callbacks sobre o estado atual da prótese virtual.
        """
        state = {
            'current_gesture': self.current_gesture,
            'target_gesture': self.target_gesture,
            'transition_progress': self.transition_progress,
            'finger_positions': self.finger_positions.copy(),
            'timestamp': time.time()
        }
        
        for callback in self.update_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Erro em callback de atualização: {str(e)}")
