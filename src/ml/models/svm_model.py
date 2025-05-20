"""
Implementação do modelo SVM para classificação de sinais EMG.

Este módulo contém a implementação do modelo SVM (Support Vector Machine)
para classificação de gestos a partir de características extraídas de sinais EMG.
"""

import numpy as np
import joblib
import logging
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class SVMModel:
    """
    Classe para implementação do modelo SVM para classificação de sinais EMG.
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', probability=True):
        """
        Inicializa o modelo SVM.
        
        Args:
            kernel (str, optional): Kernel do SVM ('linear', 'poly', 'rbf', 'sigmoid'). Padrão é 'rbf'.
            C (float, optional): Parâmetro de regularização. Padrão é 1.0.
            gamma (str or float, optional): Coeficiente do kernel. Padrão é 'scale'.
            probability (bool, optional): Se True, habilita estimativas de probabilidade. Padrão é True.
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        
        # Cria o pipeline com scaler e SVM
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                probability=self.probability
            ))
        ])
        
        self.classes = None
        self.is_trained = False
    
    def train(self, X, y):
        """
        Treina o modelo SVM.
        
        Args:
            X (numpy.ndarray): Matriz de características de treinamento.
            y (numpy.ndarray): Vetor de classes de treinamento.
        
        Returns:
            float: Acurácia do modelo no conjunto de treinamento.
        """
        logger.info(f"Treinando modelo SVM com {X.shape[0]} amostras...")
        
        # Armazena as classes
        self.classes = np.unique(y)
        
        # Treina o modelo
        self.pipeline.fit(X, y)
        
        # Calcula a acurácia no conjunto de treinamento
        accuracy = self.pipeline.score(X, y)
        
        self.is_trained = True
        logger.info(f"Modelo SVM treinado com acurácia de {accuracy:.4f}")
        
        return accuracy
    
    def predict(self, X):
        """
        Realiza predição com o modelo SVM.
        
        Args:
            X (numpy.ndarray): Matriz de características para predição.
        
        Returns:
            numpy.ndarray: Vetor de classes preditas.
        """
        if not self.is_trained:
            logger.error("Modelo SVM não foi treinado")
            return None
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Realiza predição com probabilidades.
        
        Args:
            X (numpy.ndarray): Matriz de características para predição.
        
        Returns:
            numpy.ndarray: Matriz de probabilidades para cada classe.
        """
        if not self.is_trained:
            logger.error("Modelo SVM não foi treinado")
            return None
        
        if not self.probability:
            logger.error("Modelo SVM não foi configurado para estimar probabilidades")
            return None
        
        return self.pipeline.predict_proba(X)
    
    def save(self, file_path):
        """
        Salva o modelo em um arquivo.
        
        Args:
            file_path (str): Caminho para o arquivo.
        
        Returns:
            bool: True se o modelo foi salvo com sucesso, False caso contrário.
        """
        if not self.is_trained:
            logger.error("Modelo SVM não foi treinado")
            return False
        
        try:
            joblib.dump(self.pipeline, file_path)
            logger.info(f"Modelo SVM salvo em {file_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar modelo SVM: {str(e)}")
            return False
    
    def load(self, file_path):
        """
        Carrega o modelo de um arquivo.
        
        Args:
            file_path (str): Caminho para o arquivo.
        
        Returns:
            bool: True se o modelo foi carregado com sucesso, False caso contrário.
        """
        try:
            self.pipeline = joblib.load(file_path)
            self.is_trained = True
            
            # Extrai os parâmetros do modelo carregado
            svm = self.pipeline.named_steps['svm']
            self.kernel = svm.kernel
            self.C = svm.C
            self.gamma = svm.gamma
            self.probability = svm.probability
            self.classes = svm.classes_
            
            logger.info(f"Modelo SVM carregado de {file_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo SVM: {str(e)}")
            return False
    
    def get_params(self):
        """
        Obtém os parâmetros do modelo.
        
        Returns:
            dict: Dicionário com os parâmetros do modelo.
        """
        return {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'probability': self.probability,
            'classes': self.classes.tolist() if self.classes is not None else None,
            'is_trained': self.is_trained
        }
