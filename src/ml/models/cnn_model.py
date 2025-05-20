"""
Implementação do modelo CNN para classificação de sinais EMG.

Este módulo contém a implementação do modelo CNN (Convolutional Neural Network)
para classificação de gestos a partir de sinais EMG.
"""

import numpy as np
import os
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

logger = logging.getLogger(__name__)

class CNNModel:
    """
    Classe para implementação do modelo CNN para classificação de sinais EMG.
    """
    
    def __init__(self, input_shape=(200, 1), num_classes=5, filters=(32, 64), 
                 kernel_size=3, pool_size=2, dense_units=(128, 64), dropout_rate=0.3):
        """
        Inicializa o modelo CNN.
        
        Args:
            input_shape (tuple, optional): Forma dos dados de entrada (amostras, canais). Padrão é (200, 1).
            num_classes (int, optional): Número de classes. Padrão é 5.
            filters (tuple, optional): Número de filtros em cada camada convolucional. Padrão é (32, 64).
            kernel_size (int, optional): Tamanho do kernel das camadas convolucionais. Padrão é 3.
            pool_size (int, optional): Tamanho do pooling. Padrão é 2.
            dense_units (tuple, optional): Número de unidades nas camadas densas. Padrão é (128, 64).
            dropout_rate (float, optional): Taxa de dropout. Padrão é 0.3.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.history = None
        self.classes = None
        self.is_trained = False
        
        # Cria o modelo
        self._build_model()
    
    def _build_model(self):
        """
        Constrói a arquitetura do modelo CNN.
        """
        self.model = Sequential()
        
        # Primeira camada convolucional
        self.model.add(Conv1D(filters=self.filters[0], 
                             kernel_size=self.kernel_size, 
                             activation='relu', 
                             input_shape=self.input_shape))
        self.model.add(MaxPooling1D(pool_size=self.pool_size))
        
        # Segunda camada convolucional
        self.model.add(Conv1D(filters=self.filters[1], 
                             kernel_size=self.kernel_size, 
                             activation='relu'))
        self.model.add(MaxPooling1D(pool_size=self.pool_size))
        
        # Flatten
        self.model.add(Flatten())
        
        # Camadas densas
        self.model.add(Dense(self.dense_units[0], activation='relu'))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Dense(self.dense_units[1], activation='relu'))
        self.model.add(Dropout(self.dropout_rate))
        
        # Camada de saída
        self.model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compila o modelo
        self.model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
        
        logger.info("Modelo CNN construído")
    
    def train(self, X, y, validation_split=0.2, batch_size=32, epochs=50, patience=10):
        """
        Treina o modelo CNN.
        
        Args:
            X (numpy.ndarray): Matriz de características de treinamento.
            y (numpy.ndarray): Vetor de classes de treinamento.
            validation_split (float, optional): Fração dos dados para validação. Padrão é 0.2.
            batch_size (int, optional): Tamanho do batch. Padrão é 32.
            epochs (int, optional): Número máximo de épocas. Padrão é 50.
            patience (int, optional): Paciência para early stopping. Padrão é 10.
        
        Returns:
            dict: Histórico de treinamento.
        """
        logger.info(f"Treinando modelo CNN com {X.shape[0]} amostras...")
        
        # Armazena as classes
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        
        # Converte as classes para one-hot encoding
        y_categorical = to_categorical(y, num_classes=self.num_classes)
        
        # Reshape X para o formato esperado pelo CNN (amostras, timesteps, features)
        if len(X.shape) == 2:
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        else:
            X_reshaped = X
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ModelCheckpoint('temp_cnn_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        # Treina o modelo
        self.history = self.model.fit(
            X_reshaped, y_categorical,
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Avalia o modelo no conjunto de treinamento
        _, accuracy = self.model.evaluate(X_reshaped, y_categorical, verbose=0)
        logger.info(f"Modelo CNN treinado com acurácia de {accuracy:.4f}")
        
        # Remove o arquivo temporário
        if os.path.exists('temp_cnn_model.h5'):
            os.remove('temp_cnn_model.h5')
        
        return self.history.history
    
    def predict(self, X):
        """
        Realiza predição com o modelo CNN.
        
        Args:
            X (numpy.ndarray): Matriz de características para predição.
        
        Returns:
            numpy.ndarray: Vetor de classes preditas.
        """
        if not self.is_trained:
            logger.error("Modelo CNN não foi treinado")
            return None
        
        # Reshape X para o formato esperado pelo CNN (amostras, timesteps, features)
        if len(X.shape) == 2:
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        else:
            X_reshaped = X
        
        # Realiza a predição
        y_pred_proba = self.model.predict(X_reshaped)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Realiza predição com probabilidades.
        
        Args:
            X (numpy.ndarray): Matriz de características para predição.
        
        Returns:
            numpy.ndarray: Matriz de probabilidades para cada classe.
        """
        if not self.is_trained:
            logger.error("Modelo CNN não foi treinado")
            return None
        
        # Reshape X para o formato esperado pelo CNN (amostras, timesteps, features)
        if len(X.shape) == 2:
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        else:
            X_reshaped = X
        
        return self.model.predict(X_reshaped)
    
    def save(self, file_path):
        """
        Salva o modelo em um arquivo.
        
        Args:
            file_path (str): Caminho para o arquivo.
        
        Returns:
            bool: True se o modelo foi salvo com sucesso, False caso contrário.
        """
        if not self.is_trained:
            logger.error("Modelo CNN não foi treinado")
            return False
        
        try:
            # Salva o modelo Keras
            self.model.save(file_path)
            
            # Salva metadados adicionais
            metadata_path = file_path + '.metadata'
            np.savez(metadata_path, 
                    classes=self.classes,
                    input_shape=self.input_shape,
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    pool_size=self.pool_size,
                    dense_units=self.dense_units,
                    dropout_rate=self.dropout_rate)
            
            logger.info(f"Modelo CNN salvo em {file_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar modelo CNN: {str(e)}")
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
            # Carrega o modelo Keras
            self.model = load_model(file_path)
            
            # Carrega metadados adicionais
            metadata_path = file_path + '.metadata'
            if os.path.exists(metadata_path):
                metadata = np.load(metadata_path, allow_pickle=True)
                self.classes = metadata['classes']
                self.input_shape = tuple(metadata['input_shape'])
                self.filters = tuple(metadata['filters'])
                self.kernel_size = int(metadata['kernel_size'])
                self.pool_size = int(metadata['pool_size'])
                self.dense_units = tuple(metadata['dense_units'])
                self.dropout_rate = float(metadata['dropout_rate'])
            else:
                logger.warning(f"Arquivo de metadados não encontrado: {metadata_path}")
                # Infere o número de classes a partir da camada de saída
                self.num_classes = self.model.layers[-1].output_shape[-1]
                self.classes = np.arange(self.num_classes)
            
            self.is_trained = True
            logger.info(f"Modelo CNN carregado de {file_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo CNN: {str(e)}")
            return False
    
    def get_params(self):
        """
        Obtém os parâmetros do modelo.
        
        Returns:
            dict: Dicionário com os parâmetros do modelo.
        """
        return {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'pool_size': self.pool_size,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'classes': self.classes.tolist() if self.classes is not None else None,
            'is_trained': self.is_trained
        }
