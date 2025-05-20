"""
Módulo de aquisição de sinais EMG.

Este módulo é responsável pela aquisição de sinais EMG, seja através de hardware real
(Arduino + MyoWare 2.0) ou através de geração sintética baseada em bancos de dados públicos.
"""

from .emg_reader import EMGReader
from .arduino_interface import ArduinoInterface
from .synthetic_generator import SyntheticGenerator

__all__ = ['EMGReader', 'ArduinoInterface', 'SyntheticGenerator']
