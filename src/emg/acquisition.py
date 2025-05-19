#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Biomove - Sistema de Processamento EMG
======================================

Este script realiza a comunicação serial com o Arduino,
processa os sinais EMG com filtros digitais e visualiza
os resultados em tempo real.

Desenvolvido por: Biomove
Data: Maio 2025
"""

import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
import time
import argparse
import json
import os
from datetime import datetime

class EMGProcessor:
    def __init__(self, port='/dev/ttyACM0', baud_rate=115200, window_size=1000, 
                 sample_rate=500, notch_freq=60, highpass_freq=20, lowpass_freq=450):
        """
        Inicializa o processador de sinais EMG.
        
        Args:
            port (str): Porta serial do Arduino
            baud_rate (int): Taxa de comunicação serial
            window_size (int): Tamanho da janela de visualização
            sample_rate (int): Taxa de amostragem em Hz
            notch_freq (int): Frequência do filtro notch (rejeita-faixa) em Hz
            highpass_freq (int): Frequência de corte do filtro passa-alta em Hz
            lowpass_freq (int): Frequência de corte do filtro passa-baixa em Hz
        """
        # Parâmetros de configuração
        self.port = port
        self.baud_rate = baud_rate
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.notch_freq = notch_freq
        self.highpass_freq = highpass_freq
        self.lowpass_freq = lowpass_freq
        
        # Buffers de dados
        self.raw_data = np.zeros(window_size)
        self.filtered_data = np.zeros(window_size)
        self.baseline = 0
        self.threshold = 0
        
        # Estado do sistema
        self.is_recording = False
        self.recording_data = []
        self.recording_label = ""
        self.calibration_mode = False
        
        # Configuração dos filtros
        self.configure_filters()
        
        # Inicialização da comunicação serial
        try:
            self.ser = serial.Serial(port, baud_rate, timeout=1)
            print(f"Conexão serial estabelecida em {port}")
            time.sleep(2)  # Aguarda estabilização da conexão
        except serial.SerialException as e:
            print(f"Erro ao abrir porta serial {port}: {e}")
            raise
    
    def configure_filters(self):
        """Configura os filtros digitais para processamento do sinal EMG."""
        # Filtro Notch (rejeita-faixa) para remover ruído da rede elétrica
        notch_b, notch_a = signal.iirnotch(
            w0=self.notch_freq/(self.sample_rate/2), 
            Q=30.0
        )
        self.notch_filter = (notch_b, notch_a)
        
        # Filtro passa-alta para remover offset DC e artefatos de movimento
        highpass_b, highpass_a = signal.butter(
            N=4, 
            Wn=self.highpass_freq/(self.sample_rate/2),
            btype='highpass'
        )
        self.highpass_filter = (highpass_b, highpass_a)
        
        # Filtro passa-baixa para suavizar o sinal
        lowpass_b, lowpass_a = signal.butter(
            N=4, 
            Wn=self.lowpass_freq/(self.sample_rate/2),
            btype='lowpass'
        )
        self.lowpass_filter = (lowpass_b, lowpass_a)
    
    def apply_filters(self, data):
        """
        Aplica os filtros digitais ao sinal EMG.
        
        Args:
            data (numpy.ndarray): Sinal EMG bruto
            
        Returns:
            numpy.ndarray: Sinal EMG filtrado
        """
        # Aplica filtro notch
        notch_filtered = signal.lfilter(self.notch_filter[0], self.notch_filter[1], data)
        
        # Aplica filtro passa-alta
        highpass_filtered = signal.lfilter(self.highpass_filter[0], self.highpass_filter[1], notch_filtered)
        
        # Aplica filtro passa-baixa
        lowpass_filtered = signal.lfilter(self.lowpass_filter[0], self.lowpass_filter[1], highpass_filtered)
        
        return lowpass_filtered
    
    def convert_to_db(self, data):
        """
        Converte o sinal EMG para decibéis (dB).
        
        Args:
            data (numpy.ndarray): Sinal EMG
            
        Returns:
            numpy.ndarray: Sinal EMG em dB
        """
        # Evita log de zero ou valores negativos
        data_abs = np.abs(data)
        data_abs[data_abs < 1] = 1
        
        # Converte para dB (20 * log10(amplitude))
        return 20 * np.log10(data_abs)
    
    def read_serial_data(self):
        """
        Lê dados da porta serial.
        
        Returns:
            tuple: (raw_value, smoothed_value, baseline, threshold) ou None se não houver dados
        """
        if self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                
                # Verifica se é uma mensagem de controle
                if line.startswith("CALIBRATION_START"):
                    print("Iniciando calibração...")
                    self.calibration_mode = True
                    return None
                elif line.startswith("CALIBRATION_END"):
                    parts = line.split(',')
                    if len(parts) > 1:
                        self.baseline = int(parts[1])
                        print(f"Calibração concluída. Baseline: {self.baseline}")
                    self.calibration_mode = False
                    return None
                elif line.startswith("BIOMOVE_EMG_INIT"):
                    print("Sistema EMG inicializado")
                    return None
                
                # Processa dados normais
                parts = line.split(',')
                if len(parts) >= 4:
                    raw_value = int(parts[0])
                    smoothed_value = int(parts[1])
                    baseline = int(parts[2])
                    threshold = int(parts[3])
                    return (raw_value, smoothed_value, baseline, threshold)
            except (ValueError, UnicodeDecodeError) as e:
                print(f"Erro ao processar dados: {e}")
        
        return None
    
    def update_data(self):
        """Atualiza os buffers de dados com novas leituras."""
        data = self.read_serial_data()
        if data:
            raw_value, smoothed_value, baseline, threshold = data
            
            # Atualiza buffers
            self.raw_data = np.roll(self.raw_data, -1)
            self.raw_data[-1] = raw_value
            
            # Aplica filtros
            self.filtered_data = self.apply_filters(self.raw_data)
            
            # Atualiza baseline e threshold
            self.baseline = baseline
            self.threshold = threshold
            
            # Registra dados se estiver gravando
            if self.is_recording:
                self.recording_data.append({
                    'raw': raw_value,
                    'filtered': self.filtered_data[-1],
                    'timestamp': time.time()
                })
            
            return True
        return False
    
    def start_recording(self, label):
        """
        Inicia a gravação de dados para treinamento.
        
        Args:
            label (str): Rótulo do movimento sendo gravado
        """
        self.is_recording = True
        self.recording_label = label
        self.recording_data = []
        print(f"Iniciando gravação para '{label}'...")
    
    def stop_recording(self):
        """
        Para a gravação e salva os dados.
        
        Returns:
            str: Caminho do arquivo salvo
        """
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        # Cria diretório para dados se não existir
        data_dir = "dados_treinamento"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Gera nome de arquivo com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_dir}/{self.recording_label}_{timestamp}.json"
        
        # Salva dados em formato JSON
        with open(filename, 'w') as f:
            json.dump({
                'label': self.recording_label,
                'sample_rate': self.sample_rate,
                'data': self.recording_data
            }, f)
        
        print(f"Gravação finalizada. Dados salvos em {filename}")
        return filename
    
    def visualize_realtime(self):
        """Configura e inicia a visualização em tempo real."""
        # Configuração do gráfico
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Biomove - Monitoramento EMG em Tempo Real', fontsize=16)
        
        # Linha de tempo para o eixo x
        time_axis = np.linspace(-self.window_size/self.sample_rate, 0, self.window_size)
        
        # Gráfico do sinal bruto
        line_raw, = ax1.plot(time_axis, self.raw_data, 'c-', linewidth=1.5)
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Sinal EMG Bruto')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico do sinal filtrado
        line_filtered, = ax2.plot(time_axis, self.filtered_data, 'g-', linewidth=1.5)
        line_threshold, = ax2.plot(time_axis, [self.threshold]*self.window_size, 'r--', linewidth=1)
        ax2.set_xlabel('Tempo (s)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Sinal EMG Filtrado')
        ax2.grid(True, alpha=0.3)
        
        # Texto para status
        status_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                              color='white', fontsize=10)
        
        def init():
            """Inicialização da animação."""
            line_raw.set_ydata(self.raw_data)
            line_filtered.set_ydata(self.filtered_data)
            line_threshold.set_ydata([self.threshold]*self.window_size)
            return line_raw, line_filtered, line_threshold, status_text
        
        def update(frame):
            """Atualização da animação."""
            self.update_data()
            
            # Atualiza os dados nos gráficos
            line_raw.set_ydata(self.raw_data)
            line_filtered.set_ydata(self.filtered_data)
            line_threshold.set_ydata([self.threshold]*self.window_size)
            
            # Ajusta os limites dos eixos y
            ax1.set_ylim(np.min(self.raw_data)-50, np.max(self.raw_data)+50)
            ax2.set_ylim(np.min(self.filtered_data)-50, np.max(self.filtered_data)+50)
            
            # Atualiza texto de status
            status = "Calibrando..." if self.calibration_mode else ""
            status += " Gravando: " + self.recording_label if self.is_recording else ""
            status_text.set_text(status)
            
            return line_raw, line_filtered, line_threshold, status_text
        
        # Inicia a animação
        ani = FuncAnimation(fig, update, frames=None, init_func=init, 
                           blit=True, interval=20)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    def close(self):
        """Fecha a conexão serial."""
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
            print("Conexão serial fechada")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description='Biomove EMG Processor')
    parser.add_argument('--port', type=str, default='/dev/ttyACM0',
                        help='Porta serial do Arduino')
    parser.add_argument('--baud', type=int, default=115200,
                        help='Taxa de comunicação serial')
    parser.add_argument('--sample-rate', type=int, default=500,
                        help='Taxa de amostragem em Hz')
    parser.add_argument('--window-size', type=int, default=1000,
                        help='Tamanho da janela de visualização')
    
    args = parser.parse_args()
    
    try:
        processor = EMGProcessor(
            port=args.port,
            baud_rate=args.baud,
            sample_rate=args.sample_rate,
            window_size=args.window_size
        )
        
        # Exemplo de uso interativo
        print("\nBiomove EMG Processor")
        print("=====================")
        print("Comandos disponíveis:")
        print("  r <label> - Iniciar gravação com rótulo")
        print("  s - Parar gravação")
        print("  v - Visualizar em tempo real")
        print("  q - Sair")
        
        while True:
            cmd = input("\nComando: ").strip()
            
            if cmd.startswith('r '):
                label = cmd[2:].strip()
                if label:
                    processor.start_recording(label)
                else:
                    print("Erro: Forneça um rótulo para a gravação")
            
            elif cmd == 's':
                processor.stop_recording()
            
            elif cmd == 'v':
                processor.visualize_realtime()
            
            elif cmd == 'q':
                break
            
            else:
                print("Comando desconhecido")
    
    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo usuário")
    except Exception as e:
        print(f"Erro: {e}")
    finally:
        if 'processor' in locals():
            processor.close()


if __name__ == "__main__":
    main()
