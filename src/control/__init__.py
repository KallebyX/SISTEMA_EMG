"""
Módulo de controle para o SISTEMA_EMG.

Este módulo contém implementações para controle da prótese, seja real (INMOVE)
ou virtual, através de comandos baseados em predições de gestos.
"""

from .inmove_controller import INMOVEController
from .virtual_controller import VirtualController

__all__ = ['INMOVEController', 'VirtualController']
