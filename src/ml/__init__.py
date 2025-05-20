"""
Módulo de aprendizado de máquina para classificação de sinais EMG.

Este módulo contém implementações de diferentes algoritmos de aprendizado de máquina
para classificação de gestos a partir de sinais EMG.
"""

from .models.svm_model import SVMModel
from .models.mlp_model import MLPModel
from .models.cnn_model import CNNModel
from .models.lstm_model import LSTMModel
from .train import ModelTrainer
from .predict import ModelPredictor
from .evaluation import ModelEvaluator

__all__ = ['SVMModel', 'MLPModel', 'CNNModel', 'LSTMModel', 
           'ModelTrainer', 'ModelPredictor', 'ModelEvaluator']
