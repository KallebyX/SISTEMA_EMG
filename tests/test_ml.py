"""
Testes unitários para o módulo de aprendizado de máquina.

Este script contém testes unitários para validar as funcionalidades
do módulo de aprendizado de máquina do SISTEMA_EMG.
"""

import os
import sys
import unittest
import numpy as np
import tempfile
import shutil

# Adiciona o diretório raiz ao path para importação dos módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.models import svm_model, mlp_model
from src.ml import train, predict, evaluation

class TestSVMModel(unittest.TestCase):
    """Testes para o modelo SVM."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        # Cria dados de teste
        np.random.seed(42)
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 3, 100)
        self.X_test = np.random.rand(20, 10)
        self.y_test = np.random.randint(0, 3, 20)
        
        # Inicializa o modelo
        self.model = svm_model.SVMModel()
    
    def test_train(self):
        """Testa o treinamento do modelo."""
        # Treina o modelo
        accuracy = self.model.train(self.X_train, self.y_train)
        
        # Verifica se a acurácia é um número entre 0 e 1
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_predict(self):
        """Testa a predição do modelo."""
        # Treina o modelo
        self.model.train(self.X_train, self.y_train)
        
        # Realiza predições
        predictions = self.model.predict(self.X_test)
        
        # Verifica se o número de predições está correto
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Verifica se as predições são classes válidas
        for pred in predictions:
            self.assertIn(pred, np.unique(self.y_train))
    
    def test_predict_proba(self):
        """Testa a predição de probabilidades."""
        # Treina o modelo
        self.model.train(self.X_train, self.y_train)
        
        # Realiza predições de probabilidade
        probas = self.model.predict_proba(self.X_test)
        
        # Verifica se o formato está correto
        self.assertEqual(probas.shape, (len(self.X_test), len(np.unique(self.y_train))))
        
        # Verifica se as probabilidades somam aproximadamente 1 para cada amostra
        for proba in probas:
            self.assertAlmostEqual(np.sum(proba), 1.0, places=5)
    
    def test_save_load(self):
        """Testa o salvamento e carregamento do modelo."""
        # Cria um diretório temporário
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "svm_model.pkl")
        
        try:
            # Treina o modelo
            self.model.train(self.X_train, self.y_train)
            
            # Salva o modelo
            success = self.model.save(model_path)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(model_path))
            
            # Cria um novo modelo
            new_model = svm_model.SVMModel()
            
            # Carrega o modelo
            success = new_model.load(model_path)
            self.assertTrue(success)
            
            # Verifica se as predições são iguais
            pred1 = self.model.predict(self.X_test)
            pred2 = new_model.predict(self.X_test)
            np.testing.assert_array_equal(pred1, pred2)
        
        finally:
            # Remove o diretório temporário
            shutil.rmtree(temp_dir)

class TestMLPModel(unittest.TestCase):
    """Testes para o modelo MLP."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        # Cria dados de teste
        np.random.seed(42)
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 3, 100)
        self.X_test = np.random.rand(20, 10)
        self.y_test = np.random.randint(0, 3, 20)
        
        # Inicializa o modelo
        self.model = mlp_model.MLPModel(hidden_layer_sizes=(10,), max_iter=100)
    
    def test_train(self):
        """Testa o treinamento do modelo."""
        # Treina o modelo
        accuracy = self.model.train(self.X_train, self.y_train)
        
        # Verifica se a acurácia é um número entre 0 e 1
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_predict(self):
        """Testa a predição do modelo."""
        # Treina o modelo
        self.model.train(self.X_train, self.y_train)
        
        # Realiza predições
        predictions = self.model.predict(self.X_test)
        
        # Verifica se o número de predições está correto
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Verifica se as predições são classes válidas
        for pred in predictions:
            self.assertIn(pred, np.unique(self.y_train))
    
    def test_predict_proba(self):
        """Testa a predição de probabilidades."""
        # Treina o modelo
        self.model.train(self.X_train, self.y_train)
        
        # Realiza predições de probabilidade
        probas = self.model.predict_proba(self.X_test)
        
        # Verifica se o formato está correto
        self.assertEqual(probas.shape, (len(self.X_test), len(np.unique(self.y_train))))
        
        # Verifica se as probabilidades somam aproximadamente 1 para cada amostra
        for proba in probas:
            self.assertAlmostEqual(np.sum(proba), 1.0, places=5)
    
    def test_save_load(self):
        """Testa o salvamento e carregamento do modelo."""
        # Cria um diretório temporário
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "mlp_model.pkl")
        
        try:
            # Treina o modelo
            self.model.train(self.X_train, self.y_train)
            
            # Salva o modelo
            success = self.model.save(model_path)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(model_path))
            
            # Cria um novo modelo
            new_model = mlp_model.MLPModel()
            
            # Carrega o modelo
            success = new_model.load(model_path)
            self.assertTrue(success)
            
            # Verifica se as predições são iguais
            pred1 = self.model.predict(self.X_test)
            pred2 = new_model.predict(self.X_test)
            np.testing.assert_array_equal(pred1, pred2)
        
        finally:
            # Remove o diretório temporário
            shutil.rmtree(temp_dir)

class TestModelTrainer(unittest.TestCase):
    """Testes para o treinador de modelos."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        # Cria dados de teste
        np.random.seed(42)
        self.X = np.random.rand(100, 10)
        self.y = np.random.randint(0, 3, 100)
        
        # Inicializa o treinador
        self.trainer = train.ModelTrainer()
        
        # Adiciona modelos
        self.trainer.add_model("svm", svm_model.SVMModel())
        self.trainer.add_model("mlp", mlp_model.MLPModel(hidden_layer_sizes=(10,), max_iter=100))
    
    def test_set_data(self):
        """Testa a definição de dados."""
        # Define os dados
        self.trainer.set_data(self.X, self.y)
        
        # Verifica se os dados foram definidos corretamente
        self.assertIsNotNone(self.trainer.X_train)
        self.assertIsNotNone(self.trainer.y_train)
        self.assertIsNotNone(self.trainer.X_test)
        self.assertIsNotNone(self.trainer.y_test)
    
    def test_train_all(self):
        """Testa o treinamento de todos os modelos."""
        # Define os dados
        self.trainer.set_data(self.X, self.y)
        
        # Treina todos os modelos
        results = self.trainer.train_all(cross_validation=False)
        
        # Verifica se os resultados estão corretos
        self.assertIsInstance(results, dict)
        self.assertIn("svm", results)
        self.assertIn("mlp", results)
        
        # Verifica se as métricas estão presentes
        for model_name, result in results.items():
            self.assertIn("train_accuracy", result)
            self.assertIn("test_accuracy", result)
            self.assertIn("train_time", result)
    
    def test_save_models(self):
        """Testa o salvamento de modelos."""
        # Cria um diretório temporário
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Define os dados
            self.trainer.set_data(self.X, self.y)
            
            # Treina todos os modelos
            self.trainer.train_all(cross_validation=False)
            
            # Salva os modelos
            saved_paths = self.trainer.save_models(temp_dir)
            
            # Verifica se os caminhos foram retornados
            self.assertIsInstance(saved_paths, dict)
            self.assertIn("svm", saved_paths)
            self.assertIn("mlp", saved_paths)
            
            # Verifica se os arquivos foram criados
            for model_name, path in saved_paths.items():
                self.assertTrue(os.path.exists(path))
        
        finally:
            # Remove o diretório temporário
            shutil.rmtree(temp_dir)
    
    def test_get_best_model(self):
        """Testa a obtenção do melhor modelo."""
        # Define os dados
        self.trainer.set_data(self.X, self.y)
        
        # Treina todos os modelos
        self.trainer.train_all(cross_validation=False)
        
        # Obtém o melhor modelo
        best_name, best_model = self.trainer.get_best_model()
        
        # Verifica se o melhor modelo foi retornado
        self.assertIsNotNone(best_name)
        self.assertIsNotNone(best_model)
        self.assertIn(best_name, ["svm", "mlp"])

class TestModelPredictor(unittest.TestCase):
    """Testes para o preditor de modelos."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        # Cria dados de teste
        np.random.seed(42)
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 3, 100)
        self.X_test = np.random.rand(20, 10)
        
        # Treina um modelo
        self.model = svm_model.SVMModel()
        self.model.train(self.X_train, self.y_train)
        
        # Inicializa o preditor
        self.predictor = predict.ModelPredictor(model=self.model)
    
    def test_predict(self):
        """Testa a predição."""
        # Realiza uma predição
        prediction, confidence = self.predictor.predict(self.X_test[0])
        
        # Verifica se a predição é válida
        self.assertIsNotNone(prediction)
        self.assertIn(prediction, np.unique(self.y_train))
        
        # Verifica se a confiança é um número entre 0 e 1
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_predict_window(self):
        """Testa a predição de uma janela."""
        # Cria uma janela de dados
        window_data = {
            'features': {f'feature_{i}': self.X_test[0, i] for i in range(self.X_test.shape[1])}
        }
        
        # Realiza a predição da janela
        result = self.predictor.predict_window(window_data)
        
        # Verifica se o resultado é válido
        self.assertIsInstance(result, dict)
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertIn('timestamp', result)
    
    def test_callback(self):
        """Testa o callback de predição."""
        # Variável para armazenar o resultado do callback
        callback_result = None
        
        # Define o callback
        def callback(data):
            nonlocal callback_result
            callback_result = data
        
        # Adiciona o callback
        self.predictor.add_prediction_callback(callback)
        
        # Cria uma janela de dados
        window_data = {
            'features': {f'feature_{i}': self.X_test[0, i] for i in range(self.X_test.shape[1])}
        }
        
        # Realiza a predição da janela
        self.predictor.predict_window(window_data)
        
        # Verifica se o callback foi chamado
        self.assertIsNotNone(callback_result)
        self.assertIn('prediction', callback_result)
        self.assertIn('confidence', callback_result)
        
        # Remove o callback
        self.predictor.remove_prediction_callback(callback)

class TestModelEvaluator(unittest.TestCase):
    """Testes para o avaliador de modelos."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        # Cria dados de teste
        np.random.seed(42)
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 3, 100)
        self.X_test = np.random.rand(20, 10)
        self.y_test = np.random.randint(0, 3, 20)
        
        # Treina um modelo
        self.model = svm_model.SVMModel()
        self.model.train(self.X_train, self.y_train)
        
        # Inicializa o avaliador
        self.evaluator = evaluation.ModelEvaluator(
            model=self.model,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test
        )
    
    def test_evaluate(self):
        """Testa a avaliação do modelo."""
        # Avalia o modelo
        metrics = self.evaluator.evaluate()
        
        # Verifica se as métricas estão presentes
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertIn('classification_report', metrics)
    
    def test_generate_report(self):
        """Testa a geração de relatório."""
        # Cria um diretório temporário
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Gera o relatório
            plots = self.evaluator.generate_report(temp_dir)
            
            # Verifica se os arquivos foram criados
            self.assertIsInstance(plots, dict)
            self.assertIn('report', plots)
            self.assertTrue(os.path.exists(plots['report']))
            
            # Verifica se pelo menos um gráfico foi gerado
            self.assertGreaterEqual(len(plots), 1)
        
        finally:
            # Remove o diretório temporário
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    unittest.main()
