"""
Módulo para avaliação de modelos de aprendizado de máquina.

Este módulo contém implementações para avaliação detalhada de modelos
de aprendizado de máquina para classificação de gestos a partir de sinais EMG.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import learning_curve

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Classe para avaliação detalhada de modelos de aprendizado de máquina.
    """
    
    def __init__(self, model=None, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Inicializa o avaliador de modelos.
        
        Args:
            model: Modelo a ser avaliado.
            X_train (numpy.ndarray, optional): Dados de treinamento.
            y_train (numpy.ndarray, optional): Classes de treinamento.
            X_test (numpy.ndarray, optional): Dados de teste.
            y_test (numpy.ndarray, optional): Classes de teste.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics = {}
    
    def set_data(self, X_train, y_train, X_test, y_test):
        """
        Define os dados para avaliação.
        
        Args:
            X_train (numpy.ndarray): Dados de treinamento.
            y_train (numpy.ndarray): Classes de treinamento.
            X_test (numpy.ndarray): Dados de teste.
            y_test (numpy.ndarray): Classes de teste.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def set_model(self, model):
        """
        Define o modelo a ser avaliado.
        
        Args:
            model: Modelo a ser avaliado.
        """
        self.model = model
    
    def evaluate(self):
        """
        Realiza a avaliação completa do modelo.
        
        Returns:
            dict: Dicionário com métricas de avaliação.
        """
        if self.model is None:
            logger.error("Modelo não definido")
            return None
        
        if self.X_test is None or self.y_test is None:
            logger.error("Dados de teste não definidos")
            return None
        
        # Realiza predições
        if hasattr(self.model, 'predict'):
            self.y_pred = self.model.predict(self.X_test)
        else:
            logger.error("Modelo não suporta método predict")
            return None
        
        # Tenta obter probabilidades se disponíveis
        if hasattr(self.model, 'predict_proba'):
            self.y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Calcula métricas básicas
        self.metrics['accuracy'] = accuracy_score(self.y_test, self.y_pred)
        self.metrics['precision'] = precision_score(self.y_test, self.y_pred, average='weighted')
        self.metrics['recall'] = recall_score(self.y_test, self.y_pred, average='weighted')
        self.metrics['f1'] = f1_score(self.y_test, self.y_pred, average='weighted')
        
        # Calcula matriz de confusão
        self.metrics['confusion_matrix'] = confusion_matrix(self.y_test, self.y_pred)
        
        # Calcula relatório de classificação
        self.metrics['classification_report'] = classification_report(self.y_test, self.y_pred, output_dict=True)
        
        logger.info(f"Avaliação concluída: acurácia={self.metrics['accuracy']:.4f}, f1={self.metrics['f1']:.4f}")
        
        return self.metrics
    
    def plot_confusion_matrix(self, output_path=None, class_names=None):
        """
        Plota a matriz de confusão.
        
        Args:
            output_path (str, optional): Caminho para salvar o gráfico.
                Se None, apenas exibe o gráfico.
            class_names (list, optional): Nomes das classes.
                Se None, usa índices numéricos.
        
        Returns:
            str: Caminho do arquivo salvo ou None se apenas exibido.
        """
        if 'confusion_matrix' not in self.metrics:
            logger.error("Matriz de confusão não disponível. Execute evaluate() primeiro.")
            return None
        
        plt.figure(figsize=(10, 8))
        cm = self.metrics['confusion_matrix']
        
        if class_names is None:
            class_names = np.unique(np.concatenate((self.y_train, self.y_test)))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Matriz de Confusão')
        plt.ylabel('Classe Real')
        plt.xlabel('Classe Predita')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Matriz de confusão salva em {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            plt.close()
            return None
    
    def plot_roc_curve(self, output_path=None):
        """
        Plota a curva ROC para classificação multiclasse (one-vs-rest).
        
        Args:
            output_path (str, optional): Caminho para salvar o gráfico.
                Se None, apenas exibe o gráfico.
        
        Returns:
            str: Caminho do arquivo salvo ou None se apenas exibido.
        """
        if self.y_pred_proba is None:
            logger.error("Probabilidades não disponíveis. O modelo não suporta predict_proba.")
            return None
        
        plt.figure(figsize=(10, 8))
        
        # Obtém classes únicas
        classes = np.unique(np.concatenate((self.y_train, self.y_test)))
        n_classes = len(classes)
        
        # Calcula curva ROC para cada classe (one-vs-rest)
        for i, class_id in enumerate(classes):
            # Converte para problema binário (classe atual vs resto)
            y_test_bin = (self.y_test == class_id).astype(int)
            y_score = self.y_pred_proba[:, i]
            
            # Calcula curva ROC
            fpr, tpr, _ = roc_curve(y_test_bin, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Plota curva ROC
            plt.plot(fpr, tpr, lw=2,
                    label=f'Classe {class_id} (AUC = {roc_auc:.2f})')
        
        # Plota linha diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva ROC Multiclasse (One-vs-Rest)')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Curva ROC salva em {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            plt.close()
            return None
    
    def plot_learning_curve(self, output_path=None, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
        """
        Plota a curva de aprendizado do modelo.
        
        Args:
            output_path (str, optional): Caminho para salvar o gráfico.
                Se None, apenas exibe o gráfico.
            cv (int, optional): Número de folds para validação cruzada. Padrão é 5.
            train_sizes (numpy.ndarray, optional): Tamanhos relativos do conjunto de treinamento.
                Padrão é np.linspace(0.1, 1.0, 5).
        
        Returns:
            str: Caminho do arquivo salvo ou None se apenas exibido.
        """
        if self.model is None:
            logger.error("Modelo não definido")
            return None
        
        if self.X_train is None or self.y_train is None:
            logger.error("Dados de treinamento não definidos")
            return None
        
        plt.figure(figsize=(10, 6))
        
        # Calcula curva de aprendizado
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X_train, self.y_train, cv=cv, n_jobs=-1,
            train_sizes=train_sizes, scoring='accuracy'
        )
        
        # Calcula médias e desvios padrão
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plota curvas
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Treinamento')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
        
        plt.plot(train_sizes, test_mean, 'o-', color='g', label='Validação')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
        
        plt.xlabel('Tamanho do Conjunto de Treinamento')
        plt.ylabel('Acurácia')
        plt.title('Curva de Aprendizado')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Curva de aprendizado salva em {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            plt.close()
            return None
    
    def plot_feature_importance(self, output_path=None, feature_names=None, top_n=10):
        """
        Plota a importância das características para modelos que suportam essa funcionalidade.
        
        Args:
            output_path (str, optional): Caminho para salvar o gráfico.
                Se None, apenas exibe o gráfico.
            feature_names (list, optional): Nomes das características.
                Se None, usa índices numéricos.
            top_n (int, optional): Número de características mais importantes a serem exibidas.
                Padrão é 10.
        
        Returns:
            str: Caminho do arquivo salvo ou None se apenas exibido.
        """
        if self.model is None:
            logger.error("Modelo não definido")
            return None
        
        # Verifica se o modelo suporta importância de características
        if not hasattr(self.model, 'feature_importances_'):
            # Tenta acessar através de um pipeline
            if hasattr(self.model, 'named_steps'):
                for step_name, step in self.model.named_steps.items():
                    if hasattr(step, 'feature_importances_'):
                        importances = step.feature_importances_
                        break
                else:
                    logger.error("Modelo não suporta importância de características")
                    return None
            else:
                logger.error("Modelo não suporta importância de características")
                return None
        else:
            importances = self.model.feature_importances_
        
        # Define nomes das características
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        # Cria DataFrame para ordenação
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Ordena por importância e seleciona as top_n
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Plota gráfico de barras
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Top {top_n} Características Mais Importantes')
        plt.xlabel('Importância')
        plt.ylabel('Característica')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Importância de características salva em {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            plt.close()
            return None
    
    def generate_report(self, output_dir, class_names=None):
        """
        Gera um relatório completo de avaliação com gráficos e métricas.
        
        Args:
            output_dir (str): Diretório para salvar o relatório.
            class_names (list, optional): Nomes das classes.
                Se None, usa índices numéricos.
        
        Returns:
            dict: Dicionário com caminhos dos arquivos gerados.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Avalia o modelo se ainda não foi avaliado
        if not self.metrics:
            self.evaluate()
        
        # Gera gráficos
        plots = {}
        plots['confusion_matrix'] = self.plot_confusion_matrix(
            os.path.join(output_dir, 'confusion_matrix.png'),
            class_names=class_names
        )
        
        if self.y_pred_proba is not None:
            plots['roc_curve'] = self.plot_roc_curve(
                os.path.join(output_dir, 'roc_curve.png')
            )
        
        plots['learning_curve'] = self.plot_learning_curve(
            os.path.join(output_dir, 'learning_curve.png')
        )
        
        # Tenta gerar gráfico de importância de características
        try:
            plots['feature_importance'] = self.plot_feature_importance(
                os.path.join(output_dir, 'feature_importance.png')
            )
        except:
            logger.warning("Não foi possível gerar gráfico de importância de características")
        
        # Gera relatório em texto
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("=== RELATÓRIO DE AVALIAÇÃO DO MODELO ===\n\n")
            
            f.write("--- Métricas Gerais ---\n")
            f.write(f"Acurácia: {self.metrics['accuracy']:.4f}\n")
            f.write(f"Precisão: {self.metrics['precision']:.4f}\n")
            f.write(f"Recall: {self.metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {self.metrics['f1']:.4f}\n\n")
            
            f.write("--- Relatório de Classificação ---\n")
            # Converte o dicionário para formato de texto
            for class_id, metrics in self.metrics['classification_report'].items():
                if class_id in ['accuracy', 'macro avg', 'weighted avg']:
                    f.write(f"\n{class_id}:\n")
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {metric_name}: {value:.4f}\n")
                else:
                    class_name = class_names[int(class_id)] if class_names and int(class_id) < len(class_names) else class_id
                    f.write(f"\nClasse {class_name}:\n")
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {metric_name}: {value:.4f}\n")
        
        plots['report'] = report_path
        logger.info(f"Relatório de avaliação gerado em {output_dir}")
        
        return plots
