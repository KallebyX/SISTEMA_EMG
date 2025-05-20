"""
Implementação do modelo MLP para classificação de sinais EMG.

Este módulo contém a implementação do modelo MLP (Multi-Layer Perceptron)
para classificação de gestos a partir de características extraídas de sinais EMG.
"""

import numpy as np
import joblib
import logging
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class MLPModel:
    """
    Classe para implementação do modelo MLP para classificação de sinais EMG.
    """
    
    def __init__(self, hidden_layer_sizes=(100, 50), activation='relu', 
                 solver='adam', alpha=0.0001, learning_rate='adaptive',
                 max_iter=200, early_stopping=True):
        """
        Inicializa o modelo MLP.
        
        Args:
            hidden_layer_sizes (tuple, optional): Tamanho das camadas ocultas. Padrão é (100, 50).
            activation (str, optional): Função de ativação ('identity', 'logistic', 'tanh', 'relu'). Padrão é 'relu'.
            solver (str, optional): Algoritmo de otimização ('lbfgs', 'sgd', 'adam'). Padrão é 'adam'.
            alpha (float, optional): Termo de regularização L2. Padrão é 0.0001.
            learning_rate (str, optional): Estratégia de taxa de aprendizado. Padrão é 'adaptive'.
            max_iter (int, optional): Número máximo de iterações. Padrão é 200.
            early_stopping (bool, optional): Se True, usa early stopping. Padrão é True.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        
        # Cria o pipeline com scaler e MLP
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                early_stopping=self.early_stopping,
                random_state=42
            ))
        ])
        
        self.classes = None
        self.is_trained = False
    
    def train(self, X, y):
        """
        Treina o modelo MLP.
        
        Args:
            X (numpy.ndarray): Matriz de características de treinamento.
            y (numpy.ndarray): Vetor de classes de treinamento.
        
        Returns:
            float: Acurácia do modelo no conjunto de treinamento.
        """
        logger.info(f"Treinando modelo MLP com {X.shape[0]} amostras...")
        
        # Armazena as classes
        self.classes = np.unique(y)
        
        # Treina o modelo
        self.pipeline.fit(X, y)
        
        # Calcula a acurácia no conjunto de treinamento
        accuracy = self.pipeline.score(X, y)
        
        self.is_trained = True
        logger.info(f"Modelo MLP treinado com acurácia de {accuracy:.4f}")
        
        return accuracy
    
    def predict(self, X):
        """
        Realiza predição com o modelo MLP.
        
        Args:
            X (numpy.ndarray): Matriz de características para predição.
        
        Returns:
            numpy.ndarray: Vetor de classes preditas.
        """
        if not self.is_trained:
            logger.error("Modelo MLP não foi treinado")
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
            logger.error("Modelo MLP não foi treinado")
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
            logger.error("Modelo MLP não foi treinado")
            return False
        
        try:
            joblib.dump(self.pipeline, file_path)
            logger.info(f"Modelo MLP salvo em {file_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar modelo MLP: {str(e)}")
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
            mlp = self.pipeline.named_steps['mlp']
            self.hidden_layer_sizes = mlp.hidden_layer_sizes
            self.activation = mlp.activation
            self.solver = mlp.solver
            self.alpha = mlp.alpha
            self.learning_rate = mlp.learning_rate
            self.max_iter = mlp.max_iter
            self.early_stopping = mlp.early_stopping
            self.classes = mlp.classes_
            
            logger.info(f"Modelo MLP carregado de {file_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo MLP: {str(e)}")
            return False
    
    def get_params(self):
        """
        Obtém os parâmetros do modelo.
        
        Returns:
            dict: Dicionário com os parâmetros do modelo.
        """
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'solver': self.solver,
            'alpha': self.alpha,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'early_stopping': self.early_stopping,
            'classes': self.classes.tolist() if self.classes is not None else None,
            'is_trained': self.is_trained
        }
