"""
Componente de visualização da prótese virtual para a interface gráfica.

Este módulo implementa a visualização interativa da prótese virtual
usando DearPyGui para renderização.
"""

import dearpygui.dearpygui as dpg
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

class HandVisualizer:
    """
    Classe para visualização da prótese virtual.
    
    Esta classe gerencia a renderização da prótese virtual na interface gráfica,
    permitindo a visualização dos movimentos dos dedos em tempo real.
    """
    
    def __init__(self, parent_tag, width=400, height=400):
        """
        Inicializa o visualizador da prótese virtual.
        
        Args:
            parent_tag (str): Tag do elemento pai no DearPyGui.
            width (int, optional): Largura da área de visualização. Padrão é 400.
            height (int, optional): Altura da área de visualização. Padrão é 400.
        """
        self.parent_tag = parent_tag
        self.width = width
        self.height = height
        
        # Cores
        self.hand_color = [220, 190, 170, 255]  # Cor da mão
        self.outline_color = [120, 100, 90, 255]  # Cor do contorno
        self.joint_color = [100, 100, 100, 255]  # Cor das articulações
        
        # Estado atual dos dedos (0.0 = fechado, 1.0 = aberto)
        self.finger_positions = {
            "thumb": 0.3,
            "index": 0.3,
            "middle": 0.3,
            "ring": 0.3,
            "pinky": 0.3
        }
        
        # Configurações de geometria da mão
        self.palm_center = [width // 2, height // 2 + 50]
        self.palm_size = [width // 3, height // 3]
        
        # Configurações dos dedos
        self.finger_config = {
            "thumb": {
                "base": [-0.4, -0.2],  # Posição relativa à palma
                "length": [width // 8, width // 10],  # Comprimento das falanges
                "angle_base": 120,  # Ângulo base em graus
                "angle_range": [0, 70]  # Faixa de ângulo em graus
            },
            "index": {
                "base": [-0.2, -0.5],
                "length": [width // 7, width // 9, width // 12],
                "angle_base": 80,
                "angle_range": [0, 90]
            },
            "middle": {
                "base": [0.0, -0.5],
                "length": [width // 6, width // 8, width // 11],
                "angle_base": 90,
                "angle_range": [0, 90]
            },
            "ring": {
                "base": [0.2, -0.5],
                "length": [width // 7, width // 9, width // 12],
                "angle_base": 100,
                "angle_range": [0, 90]
            },
            "pinky": {
                "base": [0.4, -0.45],
                "length": [width // 8, width // 10, width // 14],
                "angle_base": 110,
                "angle_range": [0, 90]
            }
        }
        
        # Cria o drawlist
        self.setup_drawlist()
    
    def setup_drawlist(self):
        """
        Configura o drawlist para renderização da mão.
        """
        # Cria o drawlist se não existir
        if not dpg.does_item_exist(self.parent_tag):
            logger.error(f"Elemento pai '{self.parent_tag}' não existe")
            return
        
        # Limpa o drawlist
        dpg.delete_item(self.parent_tag, children_only=True)
        
        # Configura o drawlist
        with dpg.drawlist(parent=self.parent_tag, width=self.width, height=self.height):
            # Desenha a palma da mão
            dpg.draw_rectangle(
                [self.palm_center[0] - self.palm_size[0] // 2, self.palm_center[1] - self.palm_size[1] // 2],
                [self.palm_center[0] + self.palm_size[0] // 2, self.palm_center[1] + self.palm_size[1] // 2],
                color=self.hand_color,
                fill=self.hand_color,
                tag=f"{self.parent_tag}_palm"
            )
            
            # Desenha o contorno da palma
            dpg.draw_rectangle(
                [self.palm_center[0] - self.palm_size[0] // 2, self.palm_center[1] - self.palm_size[1] // 2],
                [self.palm_center[0] + self.palm_size[0] // 2, self.palm_center[1] + self.palm_size[1] // 2],
                color=self.outline_color,
                fill=None,
                thickness=2,
                tag=f"{self.parent_tag}_palm_outline"
            )
            
            # Desenha os dedos
            for finger in self.finger_positions:
                self._draw_finger(finger)
    
    def _draw_finger(self, finger_name):
        """
        Desenha um dedo específico.
        
        Args:
            finger_name (str): Nome do dedo a ser desenhado.
        """
        if finger_name not in self.finger_config:
            logger.error(f"Configuração para o dedo '{finger_name}' não encontrada")
            return
        
        # Obtém a configuração do dedo
        config = self.finger_config[finger_name]
        position = self.finger_positions[finger_name]
        
        # Calcula a posição base do dedo
        base_x = self.palm_center[0] + config["base"][0] * self.palm_size[0]
        base_y = self.palm_center[1] + config["base"][1] * self.palm_size[1]
        base_pos = [base_x, base_y]
        
        # Calcula o ângulo do dedo
        angle_base = config["angle_base"]
        angle_range = config["angle_range"]
        angle = angle_base - position * (angle_range[1] - angle_range[0])
        
        # Converte para radianos
        angle_rad = math.radians(angle)
        
        # Desenha as falanges
        prev_pos = base_pos
        prev_angle = angle_rad
        
        for i, length in enumerate(config["length"]):
            # Calcula a posição da próxima articulação
            next_x = prev_pos[0] + length * math.cos(prev_angle)
            next_y = prev_pos[1] + length * math.sin(prev_angle)
            next_pos = [next_x, next_y]
            
            # Desenha a falange
            dpg.draw_line(
                prev_pos, next_pos,
                color=self.hand_color,
                thickness=10 - i * 2,  # Diminui a espessura para as falanges distais
                tag=f"{self.parent_tag}_{finger_name}_phalanx_{i}"
            )
            
            # Desenha o contorno da falange
            dpg.draw_line(
                prev_pos, next_pos,
                color=self.outline_color,
                thickness=10 - i * 2 + 2,  # Contorno ligeiramente maior
                tag=f"{self.parent_tag}_{finger_name}_phalanx_{i}_outline",
                before=f"{self.parent_tag}_{finger_name}_phalanx_{i}"
            )
            
            # Desenha a articulação
            dpg.draw_circle(
                prev_pos,
                5 - i,  # Diminui o tamanho para as articulações distais
                color=self.joint_color,
                fill=self.joint_color,
                tag=f"{self.parent_tag}_{finger_name}_joint_{i}"
            )
            
            # Atualiza para a próxima falange
            prev_pos = next_pos
            
            # Ajusta o ângulo para a próxima falange (curvatura dos dedos)
            # Quanto mais fechado o dedo, maior a curvatura
            curve_factor = (1.0 - position) * 0.3  # Fator de curvatura
            prev_angle += curve_factor
    
    def update_finger_positions(self, positions):
        """
        Atualiza as posições dos dedos.
        
        Args:
            positions (dict): Dicionário com as posições dos dedos.
                Chaves são nomes dos dedos e valores são posições (0.0 a 1.0).
        """
        # Atualiza as posições
        for finger, position in positions.items():
            if finger in self.finger_positions:
                self.finger_positions[finger] = position
        
        # Redesenha os dedos
        for finger in self.finger_positions:
            self._update_finger(finger)
    
    def _update_finger(self, finger_name):
        """
        Atualiza a visualização de um dedo específico.
        
        Args:
            finger_name (str): Nome do dedo a ser atualizado.
        """
        # Remove o dedo atual
        for i in range(len(self.finger_config[finger_name]["length"])):
            dpg.delete_item(f"{self.parent_tag}_{finger_name}_phalanx_{i}")
            dpg.delete_item(f"{self.parent_tag}_{finger_name}_phalanx_{i}_outline")
            dpg.delete_item(f"{self.parent_tag}_{finger_name}_joint_{i}")
        
        # Desenha o dedo novamente
        self._draw_finger(finger_name)
    
    def set_gesture(self, gesture):
        """
        Define o gesto da prótese virtual.
        
        Args:
            gesture (str): Gesto a ser definido ("rest", "open", "close", "pinch", "point").
        """
        # Mapeia gestos para posições dos dedos
        gesture_positions = {
            "rest": {
                "thumb": 0.3,
                "index": 0.3,
                "middle": 0.3,
                "ring": 0.3,
                "pinky": 0.3
            },
            "open": {
                "thumb": 1.0,
                "index": 1.0,
                "middle": 1.0,
                "ring": 1.0,
                "pinky": 1.0
            },
            "close": {
                "thumb": 0.0,
                "index": 0.0,
                "middle": 0.0,
                "ring": 0.0,
                "pinky": 0.0
            },
            "pinch": {
                "thumb": 0.7,
                "index": 0.7,
                "middle": 0.0,
                "ring": 0.0,
                "pinky": 0.0
            },
            "point": {
                "thumb": 0.0,
                "index": 1.0,
                "middle": 0.0,
                "ring": 0.0,
                "pinky": 0.0
            }
        }
        
        # Verifica se o gesto é válido
        if gesture not in gesture_positions:
            logger.error(f"Gesto inválido: {gesture}")
            return
        
        # Atualiza as posições dos dedos
        self.update_finger_positions(gesture_positions[gesture])
        
        logger.info(f"Gesto definido para: {gesture}")
    
    def animate_gesture(self, gesture, duration=0.5, callback=None):
        """
        Anima a transição para um gesto específico.
        
        Args:
            gesture (str): Gesto alvo ("rest", "open", "close", "pinch", "point").
            duration (float, optional): Duração da animação em segundos. Padrão é 0.5.
            callback (callable, optional): Função a ser chamada ao final da animação.
        """
        # Mapeia gestos para posições dos dedos
        gesture_positions = {
            "rest": {
                "thumb": 0.3,
                "index": 0.3,
                "middle": 0.3,
                "ring": 0.3,
                "pinky": 0.3
            },
            "open": {
                "thumb": 1.0,
                "index": 1.0,
                "middle": 1.0,
                "ring": 1.0,
                "pinky": 1.0
            },
            "close": {
                "thumb": 0.0,
                "index": 0.0,
                "middle": 0.0,
                "ring": 0.0,
                "pinky": 0.0
            },
            "pinch": {
                "thumb": 0.7,
                "index": 0.7,
                "middle": 0.0,
                "ring": 0.0,
                "pinky": 0.0
            },
            "point": {
                "thumb": 0.0,
                "index": 1.0,
                "middle": 0.0,
                "ring": 0.0,
                "pinky": 0.0
            }
        }
        
        # Verifica se o gesto é válido
        if gesture not in gesture_positions:
            logger.error(f"Gesto inválido: {gesture}")
            return
        
        # Posições iniciais e finais
        initial_positions = self.finger_positions.copy()
        target_positions = gesture_positions[gesture]
        
        # Cria uma animação
        with dpg.item_handler_registry() as handler:
            dpg.add_item_visible_handler(callback=lambda: self._animate_step(
                initial_positions, target_positions, duration, callback
            ))
        
        # Associa o handler ao drawlist
        dpg.bind_item_handler_registry(self.parent_tag, handler)
    
    def _animate_step(self, initial_positions, target_positions, duration, callback):
        """
        Executa um passo da animação.
        
        Args:
            initial_positions (dict): Posições iniciais dos dedos.
            target_positions (dict): Posições alvo dos dedos.
            duration (float): Duração total da animação em segundos.
            callback (callable): Função a ser chamada ao final da animação.
        """
        # TODO: Implementar a animação por passos
        pass
