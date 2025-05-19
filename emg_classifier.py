#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Biomove - Sistema de Machine Learning para EMG
=============================================

Este script implementa algoritmos de machine learning para
classificação de movimentos a partir de sinais EMG.

Desenvolvido por: Biomove
Data: Maio 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob
from datetime import datetime
import pickle
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

class EMGClassifier:
    def __init__(self, data_dir="dados_treinamento", window_size=100, overlap=0.5, 
                 models_dir="modelos_treinados"):
        """
        Inicializa o classificador de sinais EMG.
        
        Args:
            data_dir (str): Diretório com os dados de treinamento
            window_size (int): Tamanho da janela para extração de características
            overlap (float): Sobreposição entre janelas (0-1)
            models_dir (str): Diretório para salvar os modelos treinados
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.overlap = overlap
        self.models_dir = models_dir
        self.features = None
        self.labels = None
        self.scaler = StandardScaler()
        self.models = {}
        
        # Cria diretório para modelos se não existir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
    
    def load_data(self):
        """
        Carrega os dados de treinamento do diretório especificado.
        
        Returns:
            tuple: (dados_carregados, rótulos_únicos)
        """
        all_data = []
        all_labels = []
        
        # Busca todos os arquivos JSON no diretório de dados
        files = glob.glob(os.path.join(self.data_dir, "*.json"))
        
        if not files:
            print(f"Nenhum arquivo de dados encontrado em {self.data_dir}")
            return False, []
        
        print(f"Carregando {len(files)} arquivos de dados...")
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                label = data['label']
                raw_data = [entry['filtered'] for entry in data['data']]
                
                all_data.append({
                    'label': label,
                    'data': raw_data
                })
                
                if label not in all_labels:
                    all_labels.append(label)
                
                print(f"Carregado: {file_path} - {len(raw_data)} amostras para '{label}'")
                
            except Exception as e:
                print(f"Erro ao carregar {file_path}: {e}")
        
        print(f"Dados carregados com sucesso. Rótulos encontrados: {all_labels}")
        return all_data, all_labels
    
    def extract_features(self, data):
        """
        Extrai características dos sinais EMG usando janelas deslizantes.
        
        Args:
            data (list): Lista de dicionários com dados e rótulos
            
        Returns:
            bool: True se a extração foi bem-sucedida
        """
        features = []
        labels = []
        
        for entry in data:
            signal = np.array(entry['data'])
            label = entry['label']
            
            # Calcula o passo entre janelas com base na sobreposição
            step = int(self.window_size * (1 - self.overlap))
            
            # Extrai janelas do sinal
            for i in range(0, len(signal) - self.window_size, step):
                window = signal[i:i+self.window_size]
                
                # Extrai características da janela
                feature_vector = self._compute_features(window)
                
                features.append(feature_vector)
                labels.append(label)
        
        if not features:
            print("Nenhuma característica extraída. Verifique os dados de entrada.")
            return False
        
        self.features = np.array(features)
        self.labels = np.array(labels)
        
        print(f"Características extraídas: {self.features.shape}, Rótulos: {self.labels.shape}")
        return True
    
    def _compute_features(self, window):
        """
        Calcula características para uma janela de sinal EMG.
        
        Args:
            window (numpy.ndarray): Janela de sinal EMG
            
        Returns:
            numpy.ndarray: Vetor de características
        """
        # Características no domínio do tempo
        mean = np.mean(window)
        std = np.std(window)
        rms = np.sqrt(np.mean(np.square(window)))
        max_val = np.max(window)
        min_val = np.min(window)
        range_val = max_val - min_val
        
        # Características estatísticas
        skewness = np.mean(((window - mean) / std)**3) if std > 0 else 0
        kurtosis = np.mean(((window - mean) / std)**4) if std > 0 else 0
        
        # Cruzamentos por zero (simplificado)
        zero_crossings = np.sum(np.diff(np.signbit(window).astype(int)) != 0)
        
        # Energia do sinal
        energy = np.sum(np.square(window)) / len(window)
        
        # Retorna vetor de características
        return np.array([mean, std, rms, max_val, min_val, range_val, 
                         skewness, kurtosis, zero_crossings, energy])
    
    def prepare_data_for_training(self):
        """
        Prepara os dados para treinamento (normalização e divisão).
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.features is None or self.labels is None:
            print("Dados não extraídos. Execute extract_features primeiro.")
            return None
        
        # Divide os dados em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42, stratify=self.labels
        )
        
        # Normaliza os dados
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"Dados preparados: X_train: {X_train.shape}, X_test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def train_svm(self, X_train, y_train):
        """
        Treina um modelo SVM para classificação.
        
        Args:
            X_train (numpy.ndarray): Dados de treinamento
            y_train (numpy.ndarray): Rótulos de treinamento
            
        Returns:
            sklearn.svm.SVC: Modelo SVM treinado
        """
        print("Treinando modelo SVM...")
        
        # Define o pipeline com busca em grade
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        }
        
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)
        
        print(f"Melhores parâmetros SVM: {grid_search.best_params_}")
        print(f"Melhor score: {grid_search.best_score_:.4f}")
        
        self.models['svm'] = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def train_mlp(self, X_train, y_train):
        """
        Treina um modelo MLP para classificação.
        
        Args:
            X_train (numpy.ndarray): Dados de treinamento
            y_train (numpy.ndarray): Rótulos de treinamento
            
        Returns:
            sklearn.neural_network.MLPClassifier: Modelo MLP treinado
        """
        print("Treinando modelo MLP...")
        
        # Define o pipeline com busca em grade
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        
        mlp = MLPClassifier(max_iter=1000, random_state=42)
        grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)
        
        print(f"Melhores parâmetros MLP: {grid_search.best_params_}")
        print(f"Melhor score: {grid_search.best_score_:.4f}")
        
        self.models['mlp'] = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def train_cnn(self, X_train, y_train, X_test, y_test):
        """
        Treina um modelo CNN para classificação.
        
        Args:
            X_train (numpy.ndarray): Dados de treinamento
            y_train (numpy.ndarray): Rótulos de treinamento
            X_test (numpy.ndarray): Dados de teste
            y_test (numpy.ndarray): Rótulos de teste
            
        Returns:
            tensorflow.keras.Model: Modelo CNN treinado
        """
        print("Treinando modelo CNN...")
        
        # Converte rótulos para one-hot encoding
        unique_labels = np.unique(y_train)
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        
        y_train_idx = np.array([label_to_idx[label] for label in y_train])
        y_test_idx = np.array([label_to_idx[label] for label in y_test])
        
        y_train_onehot = tf.keras.utils.to_categorical(y_train_idx)
        y_test_onehot = tf.keras.utils.to_categorical(y_test_idx)
        
        # Reshape dos dados para formato CNN (amostras, timesteps, features)
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Cria o modelo CNN
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(100, activation='relu'),
            Dropout(0.5),
            Dense(len(unique_labels), activation='softmax')
        ])
        
        # Compila o modelo
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks para early stopping e checkpoint
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, 'cnn_model_checkpoint.h5'),
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Treina o modelo
        history = model.fit(
            X_train_reshaped, y_train_onehot,
            epochs=50,
            batch_size=32,
            validation_data=(X_test_reshaped, y_test_onehot),
            callbacks=callbacks,
            verbose=1
        )
        
        # Avalia o modelo
        _, accuracy = model.evaluate(X_test_reshaped, y_test_onehot)
        print(f"Acurácia do CNN: {accuracy:.4f}")
        
        # Salva o modelo e o mapeamento de rótulos
        self.models['cnn'] = {
            'model': model,
            'label_mapping': label_to_idx
        }
        
        return model, history
    
    def evaluate_models(self, X_test, y_test):
        """
        Avalia os modelos treinados.
        
        Args:
            X_test (numpy.ndarray): Dados de teste
            y_test (numpy.ndarray): Rótulos de teste
            
        Returns:
            dict: Resultados da avaliação
        """
        results = {}
        
        for name, model in self.models.items():
            if name == 'cnn':
                # Avaliação específica para CNN
                X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                label_mapping = model['label_mapping']
                y_test_idx = np.array([label_mapping[label] for label in y_test])
                y_test_onehot = tf.keras.utils.to_categorical(y_test_idx)
                
                cnn_model = model['model']
                y_pred_prob = cnn_model.predict(X_test_reshaped)
                y_pred_idx = np.argmax(y_pred_prob, axis=1)
                
                # Converte índices de volta para rótulos originais
                idx_to_label = {i: label for label, i in label_mapping.items()}
                y_pred = np.array([idx_to_label[idx] for idx in y_pred_idx])
                
                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
            else:
                # Avaliação para modelos scikit-learn
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': report
            }
            
            print(f"\nAvaliação do modelo {name.upper()}:")
            print(f"Acurácia: {accuracy:.4f}")
            print("Relatório de classificação:")
            print(classification_report(y_test, y_pred))
        
        return results
    
    def save_models(self):
        """
        Salva os modelos treinados.
        
        Returns:
            dict: Caminhos dos modelos salvos
        """
        saved_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in self.models.items():
            if name == 'cnn':
                # Salva modelo CNN
                model_path = os.path.join(self.models_dir, f"cnn_model_{timestamp}.h5")
                mapping_path = os.path.join(self.models_dir, f"cnn_mapping_{timestamp}.json")
                
                model['model'].save(model_path)
                
                with open(mapping_path, 'w') as f:
                    json.dump(model['label_mapping'], f)
                
                # Converte para TFLite
                tflite_path = os.path.join(self.models_dir, f"cnn_model_{timestamp}.tflite")
                converter = tf.lite.TFLiteConverter.from_keras_model(model['model'])
                tflite_model = converter.convert()
                
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                
                saved_paths[name] = {
                    'model': model_path,
                    'mapping': mapping_path,
                    'tflite': tflite_path
                }
                
            else:
                # Salva modelos scikit-learn
                model_path = os.path.join(self.models_dir, f"{name}_model_{timestamp}.pkl")
                scaler_path = os.path.join(self.models_dir, f"{name}_scaler_{timestamp}.pkl")
                
                # Salva o modelo
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Salva o scaler
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                
                saved_paths[name] = {
                    'model': model_path,
                    'scaler': scaler_path
                }
        
        print(f"Modelos salvos em {self.models_dir}")
        return saved_paths
    
    def plot_results(self, results):
        """
        Plota os resultados da avaliação.
        
        Args:
            results (dict): Resultados da avaliação
            
        Returns:
            matplotlib.figure.Figure: Figura com os gráficos
        """
        # Configura o estilo dos gráficos
        plt.style.use('ggplot')
        
        # Cria uma figura com subplots
        fig, axes = plt.subplots(len(results), 2, figsize=(15, 5 * len(results)))
        
        for i, (name, result) in enumerate(results.items()):
            # Ajusta para caso de apenas um modelo
            if len(results) == 1:
                ax1, ax2 = axes
            else:
                ax1, ax2 = axes[i]
            
            # Plota matriz de confusão
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title(f'Matriz de Confusão - {name.upper()}')
            ax1.set_xlabel('Predito')
            ax1.set_ylabel('Real')
            
            # Plota acurácia por classe
            report = result['classification_report']
            classes = [c for c in report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
            accuracies = [report[c]['precision'] for c in classes]
            
            ax2.bar(classes, accuracies)
            ax2.set_title(f'Precisão por Classe - {name.upper()}')
            ax2.set_xlabel('Classe')
            ax2.set_ylabel('Precisão')
            ax2.set_ylim(0, 1)
            
            for j, v in enumerate(accuracies):
                ax2.text(j, v + 0.05, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        return fig
    
    def run_training_pipeline(self):
        """
        Executa o pipeline completo de treinamento.
        
        Returns:
            dict: Resultados e caminhos dos modelos
        """
        # Carrega os dados
        data, labels = self.load_data()
        if not data:
            return None
        
        # Extrai características
        if not self.extract_features(data):
            return None
        
        # Prepara os dados
        train_test_data = self.prepare_data_for_training()
        if train_test_data is None:
            return None
        
        X_train, X_test, y_train, y_test = train_test_data
        
        # Treina os modelos
        self.train_svm(X_train, y_train)
        self.train_mlp(X_train, y_train)
        self.train_cnn(X_train, y_train, X_test, y_test)
        
        # Avalia os modelos
        results = self.evaluate_models(X_test, y_test)
        
        # Plota os resultados
        fig = self.plot_results(results)
        
        # Salva a figura
        fig_path = os.path.join(self.models_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(fig_path)
        
        # Salva os modelos
        model_paths = self.save_models()
        
        return {
            'results': results,
            'model_paths': model_paths,
            'figure_path': fig_path
        }


def main():
    """Função principal."""
    # Cria diretórios se não existirem
    data_dir = "dados_treinamento"
    models_dir = "modelos_treinados"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Diretório {data_dir} criado")
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Diretório {models_dir} criado")
    
    # Verifica se existem dados de treinamento
    files = glob.glob(os.path.join(data_dir, "*.json"))
    
    if not files:
        print(f"Nenhum arquivo de dados encontrado em {data_dir}")
        print("Gere dados de treinamento primeiro usando o emg_processor.py")
        return
    
    # Cria e executa o classificador
    classifier = EMGClassifier(data_dir=data_dir, models_dir=models_dir)
    results = classifier.run_training_pipeline()
    
    if results:
        print("\nTreinamento concluído com sucesso!")
        print(f"Resultados salvos em {models_dir}")
        
        # Mostra o melhor modelo
        best_model = max(results['results'].items(), key=lambda x: x[1]['accuracy'])
        print(f"\nMelhor modelo: {best_model[0].upper()} com acurácia de {best_model[1]['accuracy']:.4f}")
    else:
        print("\nErro durante o treinamento.")


if __name__ == "__main__":
    main()
