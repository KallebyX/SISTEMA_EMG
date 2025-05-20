"""
Módulo para treinamento de modelos de aprendizado de máquina.

Este módulo contém implementações para treinamento, validação e avaliação
de diferentes modelos de aprendizado de máquina para classificação de gestos
a partir de sinais EMG.
"""

import os
import numpy as np
import pandas as pd
import logging
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Classe para treinamento e avaliação de modelos de aprendizado de máquina.
    """
    
    def __init__(self, models=None, test_size=0.2, random_state=42):
        """
        Inicializa o treinador de modelos.
        
        Args:
            models (dict, optional): Dicionário de modelos a serem treinados.
                Chaves são nomes dos modelos e valores são instâncias dos modelos.
            test_size (float, optional): Fração dos dados para teste. Padrão é 0.2.
            random_state (int, optional): Semente aleatória para reprodutibilidade. Padrão é 42.
        """
        self.models = models or {}
        self.test_size = test_size
        self.random_state = random_state
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.scaler = StandardScaler()
        self.results = {}
        self.best_model_name = None
    
    def add_model(self, name, model):
        """
        Adiciona um modelo ao treinador.
        
        Args:
            name (str): Nome do modelo.
            model: Instância do modelo.
        """
        self.models[name] = model
        logger.info(f"Modelo '{name}' adicionado ao treinador")
    
    def load_data(self, file_path, feature_cols=None, target_col='gesture', normalize=True):
        """
        Carrega dados de um arquivo CSV.
        
        Args:
            file_path (str): Caminho para o arquivo CSV.
            feature_cols (list, optional): Lista de colunas de características.
                Se None, usa todas as colunas exceto a coluna alvo.
            target_col (str, optional): Nome da coluna alvo. Padrão é 'gesture'.
            normalize (bool, optional): Se True, normaliza os dados. Padrão é True.
        
        Returns:
            bool: True se os dados foram carregados com sucesso, False caso contrário.
        """
        try:
            # Carrega o arquivo CSV
            df = pd.read_csv(file_path)
            
            # Verifica se a coluna alvo existe
            if target_col not in df.columns:
                logger.error(f"Coluna alvo '{target_col}' não encontrada no arquivo")
                return False
            
            # Define as colunas de características
            if feature_cols is None:
                feature_cols = [col for col in df.columns if col != target_col]
            
            # Verifica se todas as colunas de características existem
            for col in feature_cols:
                if col not in df.columns:
                    logger.error(f"Coluna de característica '{col}' não encontrada no arquivo")
                    return False
            
            # Extrai características e alvo
            X = df[feature_cols].values
            y = df[target_col].values
            
            # Divide os dados em conjuntos de treinamento e teste
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            
            # Normaliza os dados se solicitado
            if normalize:
                self.X_train = self.scaler.fit_transform(self.X_train)
                self.X_test = self.scaler.transform(self.X_test)
            
            logger.info(f"Dados carregados de {file_path}: {X.shape[0]} amostras, {X.shape[1]} características")
            logger.info(f"Classes: {np.unique(y)}")
            logger.info(f"Distribuição de classes: {np.bincount(y)}")
            
            return True
        
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            return False
    
    def set_data(self, X, y, normalize=True):
        """
        Define os dados de treinamento e teste.
        
        Args:
            X (numpy.ndarray): Matriz de características.
            y (numpy.ndarray): Vetor de classes.
            normalize (bool, optional): Se True, normaliza os dados. Padrão é True.
        """
        # Divide os dados em conjuntos de treinamento e teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Normaliza os dados se solicitado
        if normalize:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        
        logger.info(f"Dados definidos: {X.shape[0]} amostras, {X.shape[1]} características")
        logger.info(f"Classes: {np.unique(y)}")
        logger.info(f"Distribuição de classes: {np.bincount(y)}")
    
    def train_all(self, cross_validation=True, n_folds=5):
        """
        Treina todos os modelos adicionados.
        
        Args:
            cross_validation (bool, optional): Se True, realiza validação cruzada. Padrão é True.
            n_folds (int, optional): Número de folds para validação cruzada. Padrão é 5.
        
        Returns:
            dict: Dicionário com resultados do treinamento.
        """
        if self.X_train is None or self.y_train is None:
            logger.error("Dados de treinamento não definidos")
            return None
        
        if not self.models:
            logger.error("Nenhum modelo adicionado ao treinador")
            return None
        
        logger.info(f"Iniciando treinamento de {len(self.models)} modelos...")
        
        self.results = {}
        best_accuracy = 0.0
        
        for name, model in self.models.items():
            logger.info(f"Treinando modelo '{name}'...")
            start_time = time.time()
            
            try:
                # Treina o modelo
                if hasattr(model, 'train'):
                    # Interface personalizada
                    accuracy = model.train(self.X_train, self.y_train)
                else:
                    # Interface scikit-learn
                    model.fit(self.X_train, self.y_train)
                    accuracy = model.score(self.X_train, self.y_train)
                
                # Avalia no conjunto de teste
                if hasattr(model, 'predict'):
                    y_pred = model.predict(self.X_test)
                else:
                    y_pred = model.predict(self.X_test)
                
                test_accuracy = accuracy_score(self.y_test, y_pred)
                
                # Realiza validação cruzada se solicitado
                cv_scores = None
                if cross_validation and hasattr(model, 'fit') and hasattr(model, 'predict'):
                    cv_scores = cross_val_score(model, np.vstack((self.X_train, self.X_test)),
                                              np.hstack((self.y_train, self.y_test)),
                                              cv=n_folds)
                
                # Calcula o tempo de treinamento
                train_time = time.time() - start_time
                
                # Armazena os resultados
                self.results[name] = {
                    'train_accuracy': accuracy,
                    'test_accuracy': test_accuracy,
                    'cv_scores': cv_scores,
                    'train_time': train_time,
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                    'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
                }
                
                logger.info(f"Modelo '{name}' treinado em {train_time:.2f}s")
                logger.info(f"Acurácia de treinamento: {accuracy:.4f}")
                logger.info(f"Acurácia de teste: {test_accuracy:.4f}")
                
                if cv_scores is not None:
                    logger.info(f"Acurácia de validação cruzada: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
                # Atualiza o melhor modelo
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    self.best_model_name = name
            
            except Exception as e:
                logger.error(f"Erro ao treinar modelo '{name}': {str(e)}")
                self.results[name] = {'error': str(e)}
        
        if self.best_model_name:
            logger.info(f"Melhor modelo: '{self.best_model_name}' com acurácia de teste de {best_accuracy:.4f}")
        
        return self.results
    
    def save_models(self, output_dir):
        """
        Salva todos os modelos treinados.
        
        Args:
            output_dir (str): Diretório de saída.
        
        Returns:
            dict: Dicionário com caminhos dos modelos salvos.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_paths = {}
        
        for name, model in self.models.items():
            try:
                # Define o caminho do arquivo
                file_path = os.path.join(output_dir, f"{name}_model")
                
                # Adiciona extensão apropriada
                if name in ['cnn', 'lstm']:
                    file_path += '.h5'
                else:
                    file_path += '.pkl'
                
                # Salva o modelo
                if hasattr(model, 'save'):
                    # Interface personalizada
                    success = model.save(file_path)
                    if success:
                        saved_paths[name] = file_path
                else:
                    # Interface scikit-learn
                    joblib.dump(model, file_path)
                    saved_paths[name] = file_path
                
                logger.info(f"Modelo '{name}' salvo em {file_path}")
            
            except Exception as e:
                logger.error(f"Erro ao salvar modelo '{name}': {str(e)}")
        
        # Salva o scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        saved_paths['scaler'] = scaler_path
        logger.info(f"Scaler salvo em {scaler_path}")
        
        return saved_paths
    
    def plot_results(self, output_dir=None):
        """
        Plota os resultados do treinamento.
        
        Args:
            output_dir (str, optional): Diretório para salvar os gráficos.
                Se None, apenas exibe os gráficos.
        
        Returns:
            dict: Dicionário com caminhos dos gráficos salvos.
        """
        if not self.results:
            logger.error("Nenhum resultado disponível para plotar")
            return None
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_paths = {}
        
        # Plota acurácia de treinamento e teste
        plt.figure(figsize=(10, 6))
        models = []
        train_acc = []
        test_acc = []
        
        for name, result in self.results.items():
            if 'error' not in result:
                models.append(name)
                train_acc.append(result['train_accuracy'])
                test_acc.append(result['test_accuracy'])
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, train_acc, width, label='Treinamento')
        plt.bar(x + width/2, test_acc, width, label='Teste')
        
        plt.xlabel('Modelo')
        plt.ylabel('Acurácia')
        plt.title('Acurácia de Treinamento e Teste')
        plt.xticks(x, models)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if output_dir:
            accuracy_path = os.path.join(output_dir, 'accuracy_comparison.png')
            plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
            saved_paths['accuracy'] = accuracy_path
            logger.info(f"Gráfico de acurácia salvo em {accuracy_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Plota tempo de treinamento
        plt.figure(figsize=(10, 6))
        train_time = []
        
        for name, result in self.results.items():
            if 'error' not in result:
                train_time.append(result['train_time'])
        
        plt.bar(models, train_time, color='green')
        plt.xlabel('Modelo')
        plt.ylabel('Tempo (s)')
        plt.title('Tempo de Treinamento')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if output_dir:
            time_path = os.path.join(output_dir, 'training_time.png')
            plt.savefig(time_path, dpi=300, bbox_inches='tight')
            saved_paths['time'] = time_path
            logger.info(f"Gráfico de tempo salvo em {time_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Plota matriz de confusão para o melhor modelo
        if self.best_model_name and 'confusion_matrix' in self.results[self.best_model_name]:
            plt.figure(figsize=(8, 6))
            cm = self.results[self.best_model_name]['confusion_matrix']
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Matriz de Confusão - {self.best_model_name}')
            plt.colorbar()
            
            classes = np.unique(np.concatenate((self.y_train, self.y_test)))
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            
            fmt = 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.ylabel('Classe Real')
            plt.xlabel('Classe Predita')
            plt.tight_layout()
            
            if output_dir:
                cm_path = os.path.join(output_dir, f'{self.best_model_name}_confusion_matrix.png')
                plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                saved_paths['confusion_matrix'] = cm_path
                logger.info(f"Matriz de confusão salva em {cm_path}")
            else:
                plt.show()
            
            plt.close()
        
        return saved_paths
    
    def get_best_model(self):
        """
        Obtém o melhor modelo treinado.
        
        Returns:
            tuple: (nome do modelo, instância do modelo)
        """
        if not self.best_model_name:
            logger.error("Nenhum modelo foi identificado como o melhor")
            return None, None
        
        return self.best_model_name, self.models[self.best_model_name]
    
    def get_results_summary(self):
        """
        Obtém um resumo dos resultados do treinamento.
        
        Returns:
            dict: Resumo dos resultados.
        """
        if not self.results:
            logger.error("Nenhum resultado disponível")
            return None
        
        summary = {}
        
        for name, result in self.results.items():
            if 'error' in result:
                summary[name] = {'status': 'error', 'error': result['error']}
            else:
                summary[name] = {
                    'status': 'success',
                    'train_accuracy': result['train_accuracy'],
                    'test_accuracy': result['test_accuracy']
                }
                
                if result['cv_scores'] is not None:
                    summary[name]['cv_accuracy'] = result['cv_scores'].mean()
                    summary[name]['cv_std'] = result['cv_scores'].std()
                
                summary[name]['train_time'] = result['train_time']
        
        if self.best_model_name:
            summary['best_model'] = self.best_model_name
        
        return summary
