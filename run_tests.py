#!/usr/bin/env python3
"""
Script para executar todos os testes automatizados do SISTEMA_EMG.

Este script executa todos os testes unitários e de integração
para validar a robustez e integridade do sistema.
"""

import os
import sys
import unittest
import argparse
import logging
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tests/test_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def discover_and_run_tests(test_dir="tests", pattern="test_*.py", verbosity=2):
    """
    Descobre e executa todos os testes no diretório especificado.
    
    Args:
        test_dir (str): Diretório contendo os testes.
        pattern (str): Padrão para descobrir arquivos de teste.
        verbosity (int): Nível de detalhamento dos resultados.
    
    Returns:
        unittest.TestResult: Resultado dos testes.
    """
    logger.info(f"Descobrindo testes em {test_dir} com padrão {pattern}")
    
    # Descobre os testes
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(test_dir, pattern=pattern)
    
    # Executa os testes
    logger.info("Iniciando execução dos testes")
    start_time = datetime.now()
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(test_suite)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Registra os resultados
    logger.info(f"Execução dos testes concluída em {duration}")
    logger.info(f"Testes executados: {result.testsRun}")
    logger.info(f"Testes com sucesso: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"Falhas: {len(result.failures)}")
    logger.info(f"Erros: {len(result.errors)}")
    
    # Registra detalhes das falhas e erros
    if result.failures:
        logger.error("Detalhes das falhas:")
        for i, (test, traceback) in enumerate(result.failures):
            logger.error(f"Falha {i+1}: {test}")
            logger.error(traceback)
    
    if result.errors:
        logger.error("Detalhes dos erros:")
        for i, (test, traceback) in enumerate(result.errors):
            logger.error(f"Erro {i+1}: {test}")
            logger.error(traceback)
    
    return result

def run_specific_test_module(module_name, verbosity=2):
    """
    Executa um módulo de teste específico.
    
    Args:
        module_name (str): Nome do módulo de teste (sem a extensão .py).
        verbosity (int): Nível de detalhamento dos resultados.
    
    Returns:
        unittest.TestResult: Resultado dos testes.
    """
    logger.info(f"Executando módulo de teste específico: {module_name}")
    
    # Importa o módulo
    try:
        module = __import__(f"tests.{module_name}", fromlist=["*"])
    except ImportError as e:
        logger.error(f"Erro ao importar módulo {module_name}: {str(e)}")
        return None
    
    # Descobre os testes no módulo
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromModule(module)
    
    # Executa os testes
    logger.info(f"Iniciando execução dos testes do módulo {module_name}")
    start_time = datetime.now()
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(test_suite)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Registra os resultados
    logger.info(f"Execução dos testes do módulo {module_name} concluída em {duration}")
    logger.info(f"Testes executados: {result.testsRun}")
    logger.info(f"Testes com sucesso: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"Falhas: {len(result.failures)}")
    logger.info(f"Erros: {len(result.errors)}")
    
    return result

def generate_test_report(result, output_file="tests/test_report.html"):
    """
    Gera um relatório HTML dos resultados dos testes.
    
    Args:
        result (unittest.TestResult): Resultado dos testes.
        output_file (str): Caminho para o arquivo de saída.
    """
    logger.info(f"Gerando relatório de testes em {output_file}")
    
    # Cria o diretório de saída se não existir
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Gera o relatório HTML
    with open(output_file, "w") as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Relatório de Testes - SISTEMA_EMG</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .summary {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .success {
                    color: #28a745;
                }
                .failure {
                    color: #dc3545;
                }
                .error {
                    color: #dc3545;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                th, td {
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .details {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-top: 10px;
                    white-space: pre-wrap;
                    font-family: monospace;
                    font-size: 14px;
                    overflow-x: auto;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Relatório de Testes - SISTEMA_EMG</h1>
                <p>Gerado em: """ + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + """</p>
                
                <div class="summary">
                    <h2>Resumo</h2>
                    <p>Total de testes: """ + str(result.testsRun) + """</p>
                    <p class="success">Testes com sucesso: """ + str(result.testsRun - len(result.failures) - len(result.errors)) + """</p>
                    <p class="failure">Falhas: """ + str(len(result.failures)) + """</p>
                    <p class="error">Erros: """ + str(len(result.errors)) + """</p>
                </div>
        """)
        
        # Adiciona detalhes das falhas
        if result.failures:
            f.write("""
                <h2 class="failure">Falhas</h2>
                <table>
                    <tr>
                        <th>#</th>
                        <th>Teste</th>
                        <th>Detalhes</th>
                    </tr>
            """)
            
            for i, (test, traceback) in enumerate(result.failures):
                f.write(f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{test}</td>
                        <td><details>
                            <summary>Ver detalhes</summary>
                            <div class="details">{traceback}</div>
                        </details></td>
                    </tr>
                """)
            
            f.write("</table>")
        
        # Adiciona detalhes dos erros
        if result.errors:
            f.write("""
                <h2 class="error">Erros</h2>
                <table>
                    <tr>
                        <th>#</th>
                        <th>Teste</th>
                        <th>Detalhes</th>
                    </tr>
            """)
            
            for i, (test, traceback) in enumerate(result.errors):
                f.write(f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{test}</td>
                        <td><details>
                            <summary>Ver detalhes</summary>
                            <div class="details">{traceback}</div>
                        </details></td>
                    </tr>
                """)
            
            f.write("</table>")
        
        # Finaliza o HTML
        f.write("""
            </div>
        </body>
        </html>
        """)
    
    logger.info(f"Relatório de testes gerado em {output_file}")

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Executa testes automatizados do SISTEMA_EMG")
    parser.add_argument("--module", help="Nome do módulo de teste específico a ser executado (sem a extensão .py)")
    parser.add_argument("--pattern", default="test_*.py", help="Padrão para descobrir arquivos de teste")
    parser.add_argument("--verbosity", type=int, default=2, help="Nível de detalhamento dos resultados")
    parser.add_argument("--report", default="tests/test_report.html", help="Caminho para o arquivo de relatório HTML")
    
    args = parser.parse_args()
    
    # Executa os testes
    if args.module:
        result = run_specific_test_module(args.module, verbosity=args.verbosity)
    else:
        result = discover_and_run_tests(pattern=args.pattern, verbosity=args.verbosity)
    
    # Gera o relatório
    if result:
        generate_test_report(result, output_file=args.report)
    
    # Retorna código de saída
    if result and (result.failures or result.errors):
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
