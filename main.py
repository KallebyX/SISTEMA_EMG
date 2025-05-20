"""
Script principal para execução do SISTEMA_EMG.

Este script inicializa o sistema e fornece a interface principal
para interação com todas as funcionalidades.
"""

import os
import sys
import logging
import argparse
from src.gui import main_window

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sistema_emg.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Analisa os argumentos de linha de comando.
    
    Returns:
        argparse.Namespace: Argumentos analisados.
    """
    parser = argparse.ArgumentParser(description='SISTEMA_EMG - Sistema para aquisição, processamento e classificação de sinais EMG')
    
    parser.add_argument('--mode', choices=['physical', 'simulated'], default='simulated',
                        help='Modo de operação: físico (com Arduino) ou simulado')
    
    parser.add_argument('--port', type=str, default=None,
                        help='Porta serial do Arduino (apenas para modo físico)')
    
    parser.add_argument('--dataset', type=str, default=None,
                        help='Caminho para o dataset a ser usado no modo simulado')
    
    parser.add_argument('--model', type=str, default=None,
                        help='Caminho para o modelo pré-treinado a ser carregado')
    
    parser.add_argument('--no-gui', action='store_true',
                        help='Executa o sistema sem interface gráfica (modo console)')
    
    return parser.parse_args()

def main():
    """
    Função principal do sistema.
    """
    # Analisa os argumentos
    args = parse_arguments()
    
    logger.info(f"Iniciando SISTEMA_EMG no modo {args.mode}")
    
    # Verifica se o diretório de modelos existe
    if not os.path.exists('models'):
        os.makedirs('models')
        logger.info("Diretório de modelos criado")
    
    # Verifica se o diretório de dados existe
    if not os.path.exists('data'):
        os.makedirs('data')
        logger.info("Diretório de dados criado")
    
    # Executa o sistema com ou sem GUI
    if args.no_gui:
        # Modo console (para scripts e automação)
        logger.info("Executando no modo console (sem GUI)")
        # TODO: Implementar modo console
    else:
        # Modo GUI
        logger.info("Iniciando interface gráfica")
        app = main_window.MainWindow(
            mode=args.mode,
            port=args.port,
            dataset_path=args.dataset,
            model_path=args.model
        )
        app.run()
    
    logger.info("SISTEMA_EMG encerrado")

if __name__ == "__main__":
    main()
