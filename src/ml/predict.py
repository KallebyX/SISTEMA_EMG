"""
Módulo para predição em tempo real usando modelos treinados.

Este módulo contém implementações para realizar predições em tempo real
usando modelos de aprendizado de máquina treinados para classificação
de gestos a partir de sinais EMG.
"""

import numpy as np
import joblib
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)

class ModelPredictor:
    """
    Classe para realizar predições em tempo real usando modelos treinados.
    """
    
    def __init__(self, model=None, scaler=None, confidence_threshold=0.7, 
                 smoothing_window=5, timeout=1.0):
        """
        Inicializa o preditor de modelos.
        
        Args:
            model: Modelo treinado para predição.
            scaler: Scaler para normalização dos dados.
            confidence_threshold (float, optional): Limiar de confiança para aceitação da predição.
                Padrão é 0.7.
            smoothing_window (int, optional): Tamanho da janela para suavização das predições.
                Padrão é 5.
            timeout (float, optional): Tempo máximo (em segundos) para manter uma predição sem
                novas confirmações. Padrão é 1.0.
        """
        self.model = model
        self.scaler = scaler
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.timeout = timeout
        
        # Histórico de predições para suavização
        self.prediction_history = deque(maxlen=smoothing_window)
        
        # Estado atual da predição
        self.current_prediction = None
        self.current_confidence = 0.0
        self.last_prediction_time = 0.0
        
        # Callbacks para notificação de novas predições
        self.prediction_callbacks = []
    
    def load_model(self, model_path, scaler_path=None):
        """
        Carrega um modelo treinado e o scaler associado.
        
        Args:
            model_path (str): Caminho para o arquivo do modelo.
            scaler_path (str, optional): Caminho para o arquivo do scaler.
                Se None, tenta inferir a partir do caminho do modelo.
        
        Returns:
            bool: True se o modelo foi carregado com sucesso, False caso contrário.
        """
        try:
            # Determina o tipo de modelo com base na extensão do arquivo
            if model_path.endswith('.h5'):
                # Modelo Keras (CNN ou LSTM)
                from tensorflow.keras.models import load_model
                self.model = load_model(model_path)
                logger.info(f"Modelo Keras carregado de {model_path}")
            else:
                # Verifica se o modelo tem método de carregamento personalizado
                if hasattr(self.model, 'load'):
                    success = self.model.load(model_path)
                    if not success:
                        logger.error(f"Falha ao carregar modelo de {model_path}")
                        return False
                else:
                    # Modelo scikit-learn ou joblib
                    self.model = joblib.load(model_path)
                    logger.info(f"Modelo carregado de {model_path}")
            
            # Carrega o scaler se fornecido
            if scaler_path:
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler carregado de {scaler_path}")
            elif self.scaler is None:
                # Tenta inferir o caminho do scaler
                try:
                    inferred_scaler_path = model_path.rsplit('_', 1)[0] + '_scaler.pkl'
                    self.scaler = joblib.load(inferred_scaler_path)
                    logger.info(f"Scaler carregado de {inferred_scaler_path}")
                except:
                    logger.warning("Não foi possível carregar o scaler. Os dados não serão normalizados.")
            
            return True
        
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            return False
    
    def predict(self, features):
        """
        Realiza uma predição com o modelo carregado.
        
        Args:
            features (numpy.ndarray): Vetor ou matriz de características.
        
        Returns:
            tuple: (classe predita, confiança)
        """
        if self.model is None:
            logger.error("Modelo não carregado")
            return None, 0.0
        
        try:
            # Garante que features seja um array 2D
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Aplica o scaler se disponível
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # Verifica o tipo de modelo e realiza a predição
            if hasattr(self.model, 'predict_proba'):
                # Modelo com probabilidades
                probas = self.model.predict_proba(features)
                prediction = np.argmax(probas, axis=1)[0]
                confidence = probas[0, prediction]
            elif hasattr(self.model, 'predict'):
                # Modelo sem probabilidades
                prediction = self.model.predict(features)[0]
                confidence = 1.0  # Não temos confiança real
            else:
                logger.error("Modelo não suporta predição")
                return None, 0.0
            
            return prediction, confidence
        
        except Exception as e:
            logger.error(f"Erro ao realizar predição: {str(e)}")
            return None, 0.0
    
    def predict_window(self, window_data):
        """
        Processa uma janela de dados e realiza predição.
        
        Args:
            window_data (dict): Dicionário contendo a janela de dados e metadados.
                Deve conter pelo menos a chave 'features' com as características extraídas.
        
        Returns:
            dict: Dicionário com a predição e metadados.
        """
        if 'features' not in window_data:
            logger.error("Janela de dados não contém características")
            return None
        
        # Extrai as características
        features = np.array(list(window_data['features'].values())).reshape(1, -1)
        
        # Realiza a predição
        prediction, confidence = self.predict(features)
        
        # Atualiza o histórico de predições
        self.prediction_history.append((prediction, confidence))
        
        # Realiza suavização das predições
        smoothed_prediction, smoothed_confidence = self._smooth_predictions()
        
        # Verifica se a predição atende ao limiar de confiança
        if smoothed_confidence >= self.confidence_threshold:
            # Atualiza o estado atual
            self.current_prediction = smoothed_prediction
            self.current_confidence = smoothed_confidence
            self.last_prediction_time = time.time()
            
            # Notifica os callbacks
            prediction_data = {
                'prediction': smoothed_prediction,
                'confidence': smoothed_confidence,
                'raw_prediction': prediction,
                'raw_confidence': confidence,
                'timestamp': time.time(),
                'features': window_data['features']
            }
            
            for callback in self.prediction_callbacks:
                try:
                    callback(prediction_data)
                except Exception as e:
                    logger.error(f"Erro em callback de predição: {str(e)}")
            
            return prediction_data
        
        # Verifica timeout da predição atual
        elif self.current_prediction is not None:
            if time.time() - self.last_prediction_time > self.timeout:
                # Reseta a predição atual após timeout
                old_prediction = self.current_prediction
                self.current_prediction = None
                self.current_confidence = 0.0
                
                # Notifica os callbacks sobre o timeout
                timeout_data = {
                    'prediction': None,
                    'confidence': 0.0,
                    'raw_prediction': prediction,
                    'raw_confidence': confidence,
                    'timestamp': time.time(),
                    'features': window_data['features'],
                    'timeout': True,
                    'previous_prediction': old_prediction
                }
                
                for callback in self.prediction_callbacks:
                    try:
                        callback(timeout_data)
                    except Exception as e:
                        logger.error(f"Erro em callback de timeout: {str(e)}")
                
                return timeout_data
        
        # Retorna dados da predição mesmo abaixo do limiar
        return {
            'prediction': None,
            'confidence': smoothed_confidence,
            'raw_prediction': prediction,
            'raw_confidence': confidence,
            'timestamp': time.time(),
            'features': window_data['features'],
            'below_threshold': True
        }
    
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
    
    def add_prediction_callback(self, callback):
        """
        Adiciona um callback para ser notificado quando uma nova predição for realizada.
        
        Args:
            callback (callable): Função a ser chamada com os dados da predição como argumento.
        """
        self.prediction_callbacks.append(callback)
    
    def remove_prediction_callback(self, callback):
        """
        Remove um callback previamente adicionado.
        
        Args:
            callback (callable): Callback a ser removido.
        """
        if callback in self.prediction_callbacks:
            self.prediction_callbacks.remove(callback)
    
    def set_confidence_threshold(self, threshold):
        """
        Define o limiar de confiança para aceitação da predição.
        
        Args:
            threshold (float): Novo limiar de confiança (0.0 a 1.0).
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Limiar de confiança definido para {self.confidence_threshold}")
    
    def set_smoothing_window(self, window_size):
        """
        Define o tamanho da janela para suavização das predições.
        
        Args:
            window_size (int): Novo tamanho da janela.
        """
        if window_size < 1:
            logger.error("Tamanho da janela deve ser pelo menos 1")
            return
        
        # Cria uma nova deque com o novo tamanho
        new_history = deque(self.prediction_history, maxlen=window_size)
        self.prediction_history = new_history
        self.smoothing_window = window_size
        
        logger.info(f"Janela de suavização definida para {self.smoothing_window}")
    
    def set_timeout(self, timeout):
        """
        Define o tempo máximo para manter uma predição sem novas confirmações.
        
        Args:
            timeout (float): Novo timeout em segundos.
        """
        self.timeout = max(0.0, timeout)
        logger.info(f"Timeout definido para {self.timeout} segundos")
    
    def reset(self):
        """
        Reinicia o estado do preditor.
        """
        self.prediction_history.clear()
        self.current_prediction = None
        self.current_confidence = 0.0
        self.last_prediction_time = 0.0
        
        logger.info("Estado do preditor reiniciado")
