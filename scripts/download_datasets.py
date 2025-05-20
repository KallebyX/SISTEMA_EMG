"""
Script para download e processamento de bancos públicos de EMG.

Este script baixa e processa dados de bancos públicos de EMG para uso no modo simulado.
"""

import os
import sys
import requests
import zipfile
import tarfile
import gzip
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Diretórios
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Diretórios específicos para cada banco
NINAPRO_DIR = os.path.join(RAW_DIR, 'ninapro')
EMG_UKA_DIR = os.path.join(RAW_DIR, 'emg_uka')
PHYSIONET_DIR = os.path.join(RAW_DIR, 'physionet')

# URLs dos bancos de dados
NINAPRO_URL = "http://ninapro.hevs.ch/system/files/DB2/DB2_s1.zip"  # Exemplo: sujeito 1 do DB2
EMG_UKA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00481/EMG_data_for_gestures-master.zip"
PHYSIONET_URL = "https://physionet.org/files/emgdb/1.0.0/emgdb.zip"

def ensure_dirs():
    """Garante que todos os diretórios necessários existam."""
    for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, NINAPRO_DIR, EMG_UKA_DIR, PHYSIONET_DIR]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Diretório garantido: {directory}")

def download_file(url, destination):
    """
    Baixa um arquivo de uma URL para um destino específico.
    
    Args:
        url (str): URL do arquivo a ser baixado.
        destination (str): Caminho de destino para salvar o arquivo.
    
    Returns:
        bool: True se o download foi bem-sucedido, False caso contrário.
    """
    try:
        # Verifica se o arquivo já existe
        if os.path.exists(destination):
            logger.info(f"Arquivo já existe: {destination}")
            return True
        
        # Cria o diretório de destino se não existir
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Baixa o arquivo
        logger.info(f"Baixando {url} para {destination}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Obtém o tamanho total do arquivo
        total_size = int(response.headers.get('content-length', 0))
        
        # Baixa o arquivo com barra de progresso
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        
        logger.info(f"Download concluído: {destination}")
        return True
    
    except Exception as e:
        logger.error(f"Erro ao baixar {url}: {str(e)}")
        return False

def extract_file(file_path, extract_dir):
    """
    Extrai um arquivo compactado.
    
    Args:
        file_path (str): Caminho do arquivo a ser extraído.
        extract_dir (str): Diretório de destino para extração.
    
    Returns:
        bool: True se a extração foi bem-sucedida, False caso contrário.
    """
    try:
        # Cria o diretório de extração se não existir
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extrai o arquivo de acordo com sua extensão
        logger.info(f"Extraindo {file_path} para {extract_dir}")
        
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        
        elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
        
        elif file_path.endswith('.tar'):
            with tarfile.open(file_path, 'r') as tar_ref:
                tar_ref.extractall(extract_dir)
        
        elif file_path.endswith('.gz') and not file_path.endswith('.tar.gz'):
            with gzip.open(file_path, 'rb') as f_in:
                with open(file_path[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        else:
            logger.error(f"Formato de arquivo não suportado: {file_path}")
            return False
        
        logger.info(f"Extração concluída: {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Erro ao extrair {file_path}: {str(e)}")
        return False

def process_ninapro():
    """
    Processa os dados do banco Ninapro.
    
    Returns:
        bool: True se o processamento foi bem-sucedido, False caso contrário.
    """
    try:
        # Baixa os dados
        ninapro_zip = os.path.join(RAW_DIR, 'ninapro_db2_s1.zip')
        if not download_file(NINAPRO_URL, ninapro_zip):
            return False
        
        # Extrai os dados
        if not extract_file(ninapro_zip, NINAPRO_DIR):
            return False
        
        # Processa os dados
        logger.info("Processando dados do Ninapro")
        
        # Procura por arquivos de dados
        emg_files = []
        gesture_files = []
        
        for root, _, files in os.walk(NINAPRO_DIR):
            for file in files:
                if file.endswith('.mat'):
                    if 'emg' in file.lower():
                        emg_files.append(os.path.join(root, file))
                    elif 'gesture' in file.lower() or 'restimulus' in file.lower():
                        gesture_files.append(os.path.join(root, file))
        
        if not emg_files or not gesture_files:
            logger.error("Arquivos de dados do Ninapro não encontrados")
            return False
        
        # Processa os arquivos
        processed_data = []
        
        for emg_file in emg_files:
            # Encontra o arquivo de gestos correspondente
            base_name = os.path.basename(emg_file).split('_')[0]
            gesture_file = None
            
            for g_file in gesture_files:
                if base_name in g_file:
                    gesture_file = g_file
                    break
            
            if not gesture_file:
                logger.warning(f"Arquivo de gestos não encontrado para {emg_file}")
                continue
            
            # Carrega os dados
            try:
                from scipy.io import loadmat
                emg_data = loadmat(emg_file)
                gesture_data = loadmat(gesture_file)
                
                # Extrai os arrays
                emg = emg_data['emg']
                gestures = gesture_data['restimulus'] if 'restimulus' in gesture_data else gesture_data['gesture']
                
                # Mapeia os gestos para nosso formato
                gesture_map = {
                    0: 0,  # Repouso
                    1: 1,  # Mão aberta
                    2: 2,  # Mão fechada
                    3: 3,  # Pinça
                    8: 4,  # Apontar
                }
                
                # Processa os dados
                for i in range(len(gestures)):
                    gesture = gestures[i][0]
                    if gesture in gesture_map:
                        # Seleciona apenas os gestos que nos interessam
                        processed_data.append({
                            'emg': emg[i],
                            'gesture': gesture_map[gesture],
                            'source': 'ninapro'
                        })
            
            except Exception as e:
                logger.error(f"Erro ao processar {emg_file}: {str(e)}")
                continue
        
        # Salva os dados processados
        if processed_data:
            ninapro_processed = os.path.join(PROCESSED_DIR, 'ninapro_processed.csv')
            
            # Converte para DataFrame
            df_rows = []
            for item in processed_data:
                row = {'gesture': item['gesture'], 'source': item['source']}
                for i, val in enumerate(item['emg']):
                    row[f'emg_{i}'] = val
                df_rows.append(row)
            
            df = pd.DataFrame(df_rows)
            df.to_csv(ninapro_processed, index=False)
            
            logger.info(f"Dados do Ninapro processados e salvos em {ninapro_processed}")
            return True
        else:
            logger.error("Nenhum dado processado do Ninapro")
            return False
    
    except Exception as e:
        logger.error(f"Erro ao processar dados do Ninapro: {str(e)}")
        return False

def process_emg_uka():
    """
    Processa os dados do banco EMG-UKA.
    
    Returns:
        bool: True se o processamento foi bem-sucedido, False caso contrário.
    """
    try:
        # Baixa os dados
        emg_uka_zip = os.path.join(RAW_DIR, 'emg_uka.zip')
        if not download_file(EMG_UKA_URL, emg_uka_zip):
            return False
        
        # Extrai os dados
        if not extract_file(emg_uka_zip, EMG_UKA_DIR):
            return False
        
        # Processa os dados
        logger.info("Processando dados do EMG-UKA")
        
        # Procura por arquivos de dados
        data_files = []
        
        for root, _, files in os.walk(EMG_UKA_DIR):
            for file in files:
                if file.endswith('.csv'):
                    data_files.append(os.path.join(root, file))
        
        if not data_files:
            logger.error("Arquivos de dados do EMG-UKA não encontrados")
            return False
        
        # Processa os arquivos
        processed_data = []
        
        for data_file in data_files:
            try:
                # Carrega o arquivo CSV
                df = pd.read_csv(data_file)
                
                # Verifica se o arquivo tem as colunas esperadas
                if 'class' not in df.columns:
                    logger.warning(f"Arquivo {data_file} não tem coluna 'class'")
                    continue
                
                # Mapeia as classes para nosso formato
                gesture_map = {
                    0: 0,  # Repouso
                    1: 1,  # Mão aberta
                    2: 2,  # Mão fechada
                    3: 3,  # Pinça
                    4: 4,  # Apontar
                }
                
                # Filtra apenas as classes que nos interessam
                df = df[df['class'].isin(gesture_map.keys())].copy()
                
                if df.empty:
                    logger.warning(f"Nenhum gesto relevante encontrado em {data_file}")
                    continue
                
                # Mapeia as classes
                df['gesture'] = df['class'].map(gesture_map)
                
                # Identifica as colunas de EMG
                emg_cols = [col for col in df.columns if col.startswith('sensor')]
                
                if not emg_cols:
                    logger.warning(f"Nenhuma coluna de sensor encontrada em {data_file}")
                    continue
                
                # Renomeia as colunas
                rename_dict = {col: f'emg_{i}' for i, col in enumerate(emg_cols)}
                df = df.rename(columns=rename_dict)
                
                # Adiciona a fonte
                df['source'] = 'emg_uka'
                
                # Seleciona apenas as colunas relevantes
                cols_to_keep = ['gesture', 'source'] + [f'emg_{i}' for i in range(len(emg_cols))]
                df = df[cols_to_keep]
                
                # Adiciona aos dados processados
                processed_data.append(df)
            
            except Exception as e:
                logger.error(f"Erro ao processar {data_file}: {str(e)}")
                continue
        
        # Combina todos os dados processados
        if processed_data:
            combined_df = pd.concat(processed_data, ignore_index=True)
            
            # Salva os dados processados
            emg_uka_processed = os.path.join(PROCESSED_DIR, 'emg_uka_processed.csv')
            combined_df.to_csv(emg_uka_processed, index=False)
            
            logger.info(f"Dados do EMG-UKA processados e salvos em {emg_uka_processed}")
            return True
        else:
            logger.error("Nenhum dado processado do EMG-UKA")
            return False
    
    except Exception as e:
        logger.error(f"Erro ao processar dados do EMG-UKA: {str(e)}")
        return False

def process_physionet():
    """
    Processa os dados do banco PhysioNet.
    
    Returns:
        bool: True se o processamento foi bem-sucedido, False caso contrário.
    """
    try:
        # Baixa os dados
        physionet_zip = os.path.join(RAW_DIR, 'physionet_emg.zip')
        if not download_file(PHYSIONET_URL, physionet_zip):
            return False
        
        # Extrai os dados
        if not extract_file(physionet_zip, PHYSIONET_DIR):
            return False
        
        # Processa os dados
        logger.info("Processando dados do PhysioNet")
        
        # Procura por arquivos de dados
        data_files = []
        
        for root, _, files in os.walk(PHYSIONET_DIR):
            for file in files:
                if file.endswith('.dat') or file.endswith('.txt'):
                    data_files.append(os.path.join(root, file))
        
        if not data_files:
            logger.error("Arquivos de dados do PhysioNet não encontrados")
            return False
        
        # Processa os arquivos
        processed_data = []
        
        for data_file in data_files:
            try:
                # Carrega o arquivo
                with open(data_file, 'r') as f:
                    lines = f.readlines()
                
                # Extrai os valores numéricos
                values = []
                for line in lines:
                    try:
                        # Tenta converter cada linha para float
                        value = float(line.strip())
                        values.append(value)
                    except ValueError:
                        # Ignora linhas que não podem ser convertidas
                        continue
                
                if not values:
                    logger.warning(f"Nenhum valor numérico encontrado em {data_file}")
                    continue
                
                # Converte para array numpy
                emg_signal = np.array(values)
                
                # Normaliza o sinal
                if np.max(emg_signal) != np.min(emg_signal):
                    emg_signal = (emg_signal - np.min(emg_signal)) / (np.max(emg_signal) - np.min(emg_signal))
                
                # Divide em segmentos
                segment_size = 200  # Tamanho da janela
                
                for i in range(0, len(emg_signal) - segment_size, segment_size):
                    segment = emg_signal[i:i+segment_size]
                    
                    # Determina o gesto com base em características do sinal
                    # Esta é uma heurística simples e pode não ser precisa
                    mean = np.mean(segment)
                    std = np.std(segment)
                    
                    if mean < 0.2 and std < 0.1:
                        gesture = 0  # Repouso
                    elif mean > 0.6:
                        gesture = 1  # Mão aberta
                    elif std > 0.3:
                        gesture = 2  # Mão fechada
                    else:
                        # Alterna entre pinça e apontar para os demais casos
                        gesture = 3 if (i // segment_size) % 2 == 0 else 4
                    
                    # Cria um dicionário com os dados
                    data_dict = {
                        'gesture': gesture,
                        'source': 'physionet'
                    }
                    
                    # Adiciona os valores do segmento
                    for j, val in enumerate(segment):
                        data_dict[f'emg_{j}'] = val
                    
                    processed_data.append(data_dict)
            
            except Exception as e:
                logger.error(f"Erro ao processar {data_file}: {str(e)}")
                continue
        
        # Salva os dados processados
        if processed_data:
            # Converte para DataFrame
            df = pd.DataFrame(processed_data)
            
            # Salva os dados processados
            physionet_processed = os.path.join(PROCESSED_DIR, 'physionet_processed.csv')
            df.to_csv(physionet_processed, index=False)
            
            logger.info(f"Dados do PhysioNet processados e salvos em {physionet_processed}")
            return True
        else:
            logger.error("Nenhum dado processado do PhysioNet")
            return False
    
    except Exception as e:
        logger.error(f"Erro ao processar dados do PhysioNet: {str(e)}")
        return False

def combine_datasets():
    """
    Combina todos os datasets processados em um único arquivo.
    
    Returns:
        bool: True se a combinação foi bem-sucedida, False caso contrário.
    """
    try:
        # Verifica se os arquivos processados existem
        ninapro_processed = os.path.join(PROCESSED_DIR, 'ninapro_processed.csv')
        emg_uka_processed = os.path.join(PROCESSED_DIR, 'emg_uka_processed.csv')
        physionet_processed = os.path.join(PROCESSED_DIR, 'physionet_processed.csv')
        
        dfs = []
        
        if os.path.exists(ninapro_processed):
            ninapro_df = pd.read_csv(ninapro_processed)
            dfs.append(ninapro_df)
            logger.info(f"Carregado {len(ninapro_df)} amostras do Ninapro")
        
        if os.path.exists(emg_uka_processed):
            emg_uka_df = pd.read_csv(emg_uka_processed)
            dfs.append(emg_uka_df)
            logger.info(f"Carregado {len(emg_uka_df)} amostras do EMG-UKA")
        
        if os.path.exists(physionet_processed):
            physionet_df = pd.read_csv(physionet_processed)
            dfs.append(physionet_df)
            logger.info(f"Carregado {len(physionet_df)} amostras do PhysioNet")
        
        if not dfs:
            logger.error("Nenhum dataset processado encontrado")
            return False
        
        # Combina os datasets
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Verifica se todos os datasets têm o mesmo número de colunas EMG
        emg_cols = [col for col in combined_df.columns if col.startswith('emg_')]
        
        if len(emg_cols) == 0:
            logger.error("Nenhuma coluna EMG encontrada nos datasets")
            return False
        
        # Garante que todas as linhas tenham o mesmo número de colunas EMG
        max_emg_idx = max([int(col.split('_')[1]) for col in emg_cols])
        
        for i in range(max_emg_idx + 1):
            col_name = f'emg_{i}'
            if col_name not in combined_df.columns:
                combined_df[col_name] = 0.0
        
        # Salva o dataset combinado
        combined_path = os.path.join(PROCESSED_DIR, 'emg_dataset.csv')
        combined_df.to_csv(combined_path, index=False)
        
        logger.info(f"Dataset combinado salvo em {combined_path} com {len(combined_df)} amostras")
        return True
    
    except Exception as e:
        logger.error(f"Erro ao combinar datasets: {str(e)}")
        return False

def main():
    """Função principal."""
    logger.info("Iniciando download e processamento de bancos públicos de EMG")
    
    # Garante que os diretórios existam
    ensure_dirs()
    
    # Processa cada banco de dados
    ninapro_success = process_ninapro()
    emg_uka_success = process_emg_uka()
    physionet_success = process_physionet()
    
    # Combina os datasets
    if ninapro_success or emg_uka_success or physionet_success:
        combine_success = combine_datasets()
        if combine_success:
            logger.info("Processamento concluído com sucesso")
            return 0
    
    logger.error("Processamento concluído com erros")
    return 1

if __name__ == "__main__":
    sys.exit(main())
