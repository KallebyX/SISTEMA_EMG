"""
Testes unitários para o módulo de controle.

Este script contém testes unitários para validar as funcionalidades
do módulo de controle do SISTEMA_EMG.
"""

import os
import sys
import unittest
import time
import threading
from unittest.mock import MagicMock, patch

# Adiciona o diretório raiz ao path para importação dos módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.control import inmove_controller, virtual_controller

class MockArduinoInterface:
    """Mock da interface Arduino para testes."""
    
    def __init__(self, connected=True):
        self.is_connected = connected
        self.commands = []
    
    def connect(self):
        self.is_connected = True
        return True
    
    def disconnect(self):
        self.is_connected = False
        return True
    
    def send_motor_command(self, command):
        self.commands.append(command)
        return True

class TestINMOVEController(unittest.TestCase):
    """Testes para o controlador da prótese INMOVE."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        # Cria um mock da interface Arduino
        self.arduino = MockArduinoInterface()
        
        # Inicializa o controlador
        self.controller = inmove_controller.INMOVEController(
            arduino_interface=self.arduino,
            confidence_threshold=0.8,
            command_timeout=0.1,
            safety_timeout=0.5
        )
    
    def test_start_stop(self):
        """Testa o início e parada do controlador."""
        # Inicia o controlador
        success = self.controller.start()
        self.assertTrue(success)
        self.assertTrue(self.controller.is_active)
        
        # Verifica se a thread de segurança foi iniciada
        self.assertIsNotNone(self.controller.safety_thread)
        
        # Para o controlador
        self.controller.stop()
        self.assertFalse(self.controller.is_active)
        
        # Verifica se um comando de parada foi enviado
        self.assertIn("STOP", self.arduino.commands)
    
    def test_process_prediction_valid(self):
        """Testa o processamento de uma predição válida."""
        # Inicia o controlador
        self.controller.start()
        
        # Cria uma predição válida
        prediction_data = {
            'prediction': 1,  # Mão aberta
            'confidence': 0.9  # Alta confiança
        }
        
        # Processa a predição
        success = self.controller.process_prediction(prediction_data)
        
        # Verifica se o comando foi enviado
        self.assertTrue(success)
        self.assertEqual(self.controller.current_command, "OPEN")
        self.assertIn("OPEN", self.arduino.commands)
        
        # Para o controlador
        self.controller.stop()
    
    def test_process_prediction_low_confidence(self):
        """Testa o processamento de uma predição com baixa confiança."""
        # Inicia o controlador
        self.controller.start()
        
        # Cria uma predição com baixa confiança
        prediction_data = {
            'prediction': 1,  # Mão aberta
            'confidence': 0.5  # Baixa confiança
        }
        
        # Processa a predição
        success = self.controller.process_prediction(prediction_data)
        
        # Verifica se o comando não foi enviado
        self.assertFalse(success)
        self.assertNotEqual(self.controller.current_command, "OPEN")
        self.assertNotIn("OPEN", self.arduino.commands)
        
        # Para o controlador
        self.controller.stop()
    
    def test_process_prediction_invalid(self):
        """Testa o processamento de uma predição inválida."""
        # Inicia o controlador
        self.controller.start()
        
        # Cria uma predição inválida
        prediction_data = {
            'prediction': None,
            'confidence': 0.0
        }
        
        # Processa a predição
        success = self.controller.process_prediction(prediction_data)
        
        # Verifica se o comando não foi enviado
        self.assertFalse(success)
        
        # Para o controlador
        self.controller.stop()
    
    def test_send_command(self):
        """Testa o envio de um comando."""
        # Inicia o controlador
        self.controller.start()
        
        # Envia um comando
        success = self.controller.send_command("CLOSE")
        
        # Verifica se o comando foi enviado
        self.assertTrue(success)
        self.assertEqual(self.controller.current_command, "CLOSE")
        self.assertIn("CLOSE", self.arduino.commands)
        
        # Para o controlador
        self.controller.stop()
    
    def test_safety_timeout(self):
        """Testa o timeout de segurança."""
        # Reduz o timeout para o teste
        self.controller.safety_timeout = 0.2
        
        # Inicia o controlador
        self.controller.start()
        
        # Envia um comando
        self.controller.send_command("CLOSE")
        
        # Espera o timeout
        time.sleep(0.3)
        
        # Verifica se um comando de parada foi enviado
        self.assertIn("STOP", self.arduino.commands)
        
        # Para o controlador
        self.controller.stop()
    
    def test_map_prediction_to_command(self):
        """Testa o mapeamento de predições para comandos."""
        # Testa diferentes predições
        self.assertEqual(self.controller._map_prediction_to_command(0), "STOP")
        self.assertEqual(self.controller._map_prediction_to_command(1), "OPEN")
        self.assertEqual(self.controller._map_prediction_to_command(2), "CLOSE")
        self.assertEqual(self.controller._map_prediction_to_command(3), "STOP")
        self.assertEqual(self.controller._map_prediction_to_command(4), "STOP")
        
        # Testa predições por string
        self.assertEqual(self.controller._map_prediction_to_command("rest"), "STOP")
        self.assertEqual(self.controller._map_prediction_to_command("open"), "OPEN")
        self.assertEqual(self.controller._map_prediction_to_command("close"), "CLOSE")
        
        # Testa predição inválida
        self.assertIsNone(self.controller._map_prediction_to_command(99))

class TestVirtualController(unittest.TestCase):
    """Testes para o controlador da prótese virtual."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        # Inicializa o controlador
        self.controller = virtual_controller.VirtualController(
            confidence_threshold=0.7,
            smoothing_window=3
        )
    
    def test_start_stop(self):
        """Testa o início e parada do controlador."""
        # Inicia o controlador
        success = self.controller.start()
        self.assertTrue(success)
        self.assertTrue(self.controller.is_active)
        
        # Verifica se a thread de atualização foi iniciada
        self.assertIsNotNone(self.controller.update_thread)
        
        # Para o controlador
        self.controller.stop()
        self.assertFalse(self.controller.is_active)
    
    def test_process_prediction_valid(self):
        """Testa o processamento de uma predição válida."""
        # Inicia o controlador
        self.controller.start()
        
        # Cria uma predição válida
        prediction_data = {
            'prediction': 1,  # Mão aberta
            'confidence': 0.8  # Alta confiança
        }
        
        # Processa a predição
        success = self.controller.process_prediction(prediction_data)
        
        # Verifica se o gesto foi atualizado
        self.assertTrue(success)
        self.assertEqual(self.controller.target_gesture, "open")
        
        # Para o controlador
        self.controller.stop()
    
    def test_process_prediction_low_confidence(self):
        """Testa o processamento de uma predição com baixa confiança."""
        # Inicia o controlador
        self.controller.start()
        
        # Cria uma predição com baixa confiança
        prediction_data = {
            'prediction': 1,  # Mão aberta
            'confidence': 0.5  # Baixa confiança
        }
        
        # Processa a predição
        success = self.controller.process_prediction(prediction_data)
        
        # Verifica se o gesto não foi atualizado
        self.assertFalse(success)
        self.assertNotEqual(self.controller.target_gesture, "open")
        
        # Para o controlador
        self.controller.stop()
    
    def test_process_prediction_invalid(self):
        """Testa o processamento de uma predição inválida."""
        # Inicia o controlador
        self.controller.start()
        
        # Cria uma predição inválida
        prediction_data = {
            'prediction': None,
            'confidence': 0.0
        }
        
        # Processa a predição
        success = self.controller.process_prediction(prediction_data)
        
        # Verifica se o gesto não foi atualizado
        self.assertFalse(success)
        
        # Para o controlador
        self.controller.stop()
    
    def test_set_gesture(self):
        """Testa a definição direta de um gesto."""
        # Inicia o controlador
        self.controller.start()
        
        # Define um gesto
        success = self.controller.set_gesture("pinch")
        
        # Verifica se o gesto foi atualizado
        self.assertTrue(success)
        self.assertEqual(self.controller.target_gesture, "pinch")
        
        # Para o controlador
        self.controller.stop()
    
    def test_get_finger_positions(self):
        """Testa a obtenção das posições dos dedos."""
        # Inicia o controlador
        self.controller.start()
        
        # Obtém as posições dos dedos
        positions = self.controller.get_finger_positions()
        
        # Verifica se as posições estão corretas
        self.assertIsInstance(positions, dict)
        self.assertIn("thumb", positions)
        self.assertIn("index", positions)
        self.assertIn("middle", positions)
        self.assertIn("ring", positions)
        self.assertIn("pinky", positions)
        
        # Para o controlador
        self.controller.stop()
    
    def test_update_callback(self):
        """Testa o callback de atualização."""
        # Variável para armazenar o resultado do callback
        callback_result = None
        
        # Define o callback
        def callback(state):
            nonlocal callback_result
            callback_result = state
        
        # Inicia o controlador
        self.controller.start()
        
        # Adiciona o callback
        self.controller.add_update_callback(callback)
        
        # Define um gesto para acionar o callback
        self.controller.set_gesture("open")
        
        # Espera um pouco para o callback ser chamado
        time.sleep(0.1)
        
        # Verifica se o callback foi chamado
        self.assertIsNotNone(callback_result)
        self.assertIn("finger_positions", callback_result)
        self.assertEqual(callback_result["target_gesture"], "open")
        
        # Remove o callback
        self.controller.remove_update_callback(callback)
        
        # Para o controlador
        self.controller.stop()
    
    def test_map_prediction_to_gesture(self):
        """Testa o mapeamento de predições para gestos."""
        # Testa diferentes predições
        self.assertEqual(self.controller._map_prediction_to_gesture(0), "rest")
        self.assertEqual(self.controller._map_prediction_to_gesture(1), "open")
        self.assertEqual(self.controller._map_prediction_to_gesture(2), "close")
        self.assertEqual(self.controller._map_prediction_to_gesture(3), "pinch")
        self.assertEqual(self.controller._map_prediction_to_gesture(4), "point")
        
        # Testa predições por string
        self.assertEqual(self.controller._map_prediction_to_gesture("rest"), "rest")
        self.assertEqual(self.controller._map_prediction_to_gesture("open"), "open")
        self.assertEqual(self.controller._map_prediction_to_gesture("close"), "close")
        
        # Testa predição inválida
        self.assertIsNone(self.controller._map_prediction_to_gesture(99))

if __name__ == '__main__':
    unittest.main()
