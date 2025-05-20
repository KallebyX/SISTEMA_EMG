#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Biomove - Sistema de Controle da Prótese
=======================================

Este script implementa o controle da prótese baseado nos
sinais EMG processados e classificados.

Desenvolvido por: Biomove
Data: Maio 2025
"""

import serial
import numpy as np
import time
import os
import json
from joblib import load
import threading
import argparse
from collections import deque
import tensorflow as tf
from scipy import signal

class ProsthesisController:
    def __init__(self, port='/dev/ttyACM0', baud_rate=115200, model_path=None, 
                 model_type='svm', threshold=0.7, window_size=100, safety_timeout=2.0, simulate=False):
        """
        Inicializa o controlador da prótese.
        
        Args:
            port (str): Porta serial do Arduino
            baud_rate (int): Taxa de comunicação serial
            model_path (str): Caminho para o modelo treinado
            model_type (str): Tipo de modelo ('svm', 'mlp', 'cnn')
            threshold (float): Limiar de confiança para ativação
            window_size (int): Tamanho da janela para extração de características
            safety_timeout (float): Tempo máximo de ativação contínua (segundos)
        """
        self.port = port
        self.baud_rate = baud_rate
        self.model_path = model_path
        self.model_type = model_type
        self.threshold = threshold
        self.window_size = window_size
        self.safety_timeout = safety_timeout
        self.simulate = simulate
        
        # Buffers e estado
        self.data_buffer = deque(maxlen=window_size)
        self.current_movement = "repouso"
        self.last_activation_time = 0
        self.is_active = False
        self.calibration_values = {}
        self.user_profile = {}

        # Configuração dos filtros (sempre, inclusive em simulação)
        self.configure_filters()

        if self.simulate:
            print("⚙️  Modo SIMULAÇÃO ativado. Nenhuma configuração de hardware será feita.")
            self.ser = None
            self.load_model()
            return
        
        # Carrega o modelo
        self.load_model()
        
        # Inicializa comunicação serial
        try:
            self.ser = serial.Serial(port, baud_rate, timeout=1)
            print(f"Conexão serial estabelecida em {port}")
            time.sleep(2)  # Aguarda estabilização da conexão
        except serial.SerialException as e:
            print(f"Erro ao abrir porta serial {port}: {e}")
            raise
    
    def configure_filters(self):
        """Configura os filtros digitais para processamento do sinal EMG."""
        sample_rate = 500  # Taxa de amostragem em Hz

        # Filtro Notch (rejeita-faixa) para remover ruído da rede elétrica
        notch_b, notch_a = signal.iirnotch(
            w0=60/(sample_rate/2),  # 60Hz (frequência da rede elétrica)
            Q=30.0
        )
        self.notch_filter = (notch_b, notch_a)

        # Filtro passa-alta para remover offset DC e artefatos de movimento
        highpass_b, highpass_a = signal.butter(
            N=4,
            Wn=20/(sample_rate/2),  # 20Hz
            btype='highpass'
        )
        self.highpass_filter = (highpass_b, highpass_a)

        # Filtro passa-baixa para suavizar o sinal
        # Garante Wn < 1 para evitar erro "ValueError: Digital filter critical frequencies must be 0 < Wn < 1"
        lowpass_cutoff = min(200, sample_rate / 2 - 1)  # Garante Wn < 1
        lowpass_b, lowpass_a = signal.butter(
            N=4,
            Wn=lowpass_cutoff / (sample_rate / 2),
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
    
    def load_model(self):
        """Carrega o modelo de classificação."""
        if not self.model_path:
            print("Caminho do modelo não especificado. Operando em modo manual.")
            self.model = None
            self.scaler = None
            return
        
        try:
            if self.model_type == 'cnn':
                # Carrega modelo CNN
                self.model = tf.keras.models.load_model(self.model_path)
                
                # Carrega mapeamento de rótulos
                mapping_path = self.model_path.replace('.h5', '_mapping.json')
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'r') as f:
                        self.label_mapping = json.load(f)
                else:
                    print("Arquivo de mapeamento de rótulos não encontrado.")
                    self.label_mapping = {}
                
            else:
                # Carrega modelo scikit-learn
                self.model = load(self.model_path)
                
                # Carrega scaler
                scaler_path = self.model_path.replace('_model_', '_scaler_')
                if os.path.exists(scaler_path):
                    self.scaler = load(scaler_path)
                else:
                    print("Arquivo de scaler não encontrado. Usando normalização padrão.")
                    self.scaler = None
            
            print(f"Modelo {self.model_type} carregado com sucesso.")
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            self.model = None
            self.scaler = None
    
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
                    return None
                elif line.startswith("CALIBRATION_END"):
                    parts = line.split(',')
                    if len(parts) > 1:
                        baseline = int(parts[1])
                        print(f"Calibração concluída. Baseline: {baseline}")
                        self.calibration_values['baseline'] = baseline
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
    
    def extract_features(self, window):
        """
        Extrai características de uma janela de sinal EMG.
        
        Args:
            window (numpy.ndarray): Janela de sinal EMG
            
        Returns:
            numpy.ndarray: Vetor de características
        """
        if len(window) < self.window_size:
            # Preenche com zeros se a janela for menor que o tamanho esperado
            window = np.pad(window, (0, self.window_size - len(window)))
        
        # Características no domínio do tempo
        mean = np.mean(window)
        std = np.std(window)
        rms = np.sqrt(np.mean(np.square(window)))
        max_val = np.max(window)
        min_val = np.min(window)
        range_val = max_val - min_val
        
        # Características estatísticas
        skewness = np.mean(((window - mean) / std)**3) if std > 0 else 0
        kurtosis = np.mean(((window - mean) / std)**4) if std > 0 else 0
        
        # Cruzamentos por zero (simplificado)
        zero_crossings = np.sum(np.diff(np.signbit(window).astype(int)) != 0)
        
        # Energia do sinal
        energy = np.sum(np.square(window)) / len(window)
        
        # Retorna vetor de características
        return np.array([mean, std, rms, max_val, min_val, range_val, 
                         skewness, kurtosis, zero_crossings, energy])
    
    def predict_movement(self):
        """
        Prediz o movimento com base nos dados do buffer.
        
        Returns:
            tuple: (movimento_predito, confiança)
        """
        if not self.model or len(self.data_buffer) < self.window_size:
            return "repouso", 0.0
        
        # Converte buffer para array e aplica filtros
        window = np.array(self.data_buffer)
        filtered_window = self.apply_filters(window)
        
        # Extrai características
        features = self.extract_features(filtered_window)
        features = features.reshape(1, -1)  # Reshape para formato esperado pelo modelo
        
        # Normaliza características se houver scaler
        if self.scaler:
            features = self.scaler.transform(features)
        
        # Predição com base no tipo de modelo
        if self.model_type == 'cnn':
            # Reshape para formato CNN (amostras, timesteps, features)
            features_reshaped = features.reshape(features.shape[0], features.shape[1], 1)
            
            # Predição
            prediction = self.model.predict(features_reshaped, verbose=0)
            confidence = np.max(prediction)
            class_idx = np.argmax(prediction)
            
            # Converte índice para rótulo
            movement = "desconhecido"
            for label, idx in self.label_mapping.items():
                if idx == class_idx:
                    movement = label
                    break
        else:
            # Predição para modelos scikit-learn
            movement = self.model.predict(features)[0]
            
            # Tenta obter probabilidades se disponível
            try:
                proba = self.model.predict_proba(features)[0]
                confidence = np.max(proba)
            except:
                confidence = 1.0  # Se não houver probabilidades, assume confiança máxima
        
        return movement, confidence
    
    def send_motor_command(self, movement, confidence):
        """
        Envia comando para o motor com base no movimento predito.
        
        Args:
            movement (str): Movimento predito
            confidence (float): Confiança da predição
        """
        current_time = time.time()
        
        # Verifica se a confiança está acima do limiar
        if confidence >= self.threshold:
            # Verifica timeout de segurança
            if movement != "repouso" and current_time - self.last_activation_time > self.safety_timeout:
                # Desativa o motor após timeout de segurança
                self.send_command("MOTOR_STOP")
                self.is_active = False
                print(f"Timeout de segurança atingido. Motor desativado.")
                return
            
            # Atualiza tempo de ativação para movimentos não-repouso
            if movement != "repouso":
                self.last_activation_time = current_time
            
            # Envia comando apropriado com base no movimento
            if movement == "mao_fechada" and not self.is_active:
                self.send_command("MOTOR_CLOSE")
                self.is_active = True
                print(f"Ativando motor: FECHAR MÃO (confiança: {confidence:.2f})")
            
            elif movement == "mao_aberta" and not self.is_active:
                self.send_command("MOTOR_OPEN")
                self.is_active = True
                print(f"Ativando motor: ABRIR MÃO (confiança: {confidence:.2f})")
            
            elif movement == "repouso" and self.is_active:
                self.send_command("MOTOR_STOP")
                self.is_active = False
                print(f"Desativando motor (confiança: {confidence:.2f})")
        
        # Se a confiança estiver abaixo do limiar e o motor estiver ativo
        elif self.is_active:
            self.send_command("MOTOR_STOP")
            self.is_active = False
            print(f"Confiança baixa ({confidence:.2f}). Motor desativado.")
    
    def send_command(self, command):
        """
        Envia comando para o Arduino.
        
        Args:
            command (str): Comando a ser enviado
        """
        try:
            self.ser.write(f"{command}\n".encode())
        except Exception as e:
            print(f"Erro ao enviar comando: {e}")
    
    def calibrate(self):
        """Inicia o processo de calibração."""
        print("Iniciando calibração...")
        self.send_command("CALIBRATE")
    
    def load_user_profile(self, profile_path):
        """
        Carrega perfil de usuário com configurações personalizadas.
        
        Args:
            profile_path (str): Caminho para o arquivo de perfil
        """
        try:
            with open(profile_path, 'r') as f:
                self.user_profile = json.load(f)
            
            # Aplica configurações do perfil
            if 'threshold' in self.user_profile:
                self.threshold = self.user_profile['threshold']
            
            if 'safety_timeout' in self.user_profile:
                self.safety_timeout = self.user_profile['safety_timeout']
            
            print(f"Perfil de usuário carregado: {profile_path}")
            
        except Exception as e:
            print(f"Erro ao carregar perfil de usuário: {e}")
    
    def save_user_profile(self, profile_path):
        """
        Salva perfil de usuário com configurações atuais.
        
        Args:
            profile_path (str): Caminho para salvar o arquivo de perfil
        """
        # Atualiza perfil com configurações atuais
        self.user_profile.update({
            'threshold': self.threshold,
            'safety_timeout': self.safety_timeout,
            'calibration_values': self.calibration_values
        })
        
        try:
            with open(profile_path, 'w') as f:
                json.dump(self.user_profile, f, indent=4)
            
            print(f"Perfil de usuário salvo: {profile_path}")
            
        except Exception as e:
            print(f"Erro ao salvar perfil de usuário: {e}")
    
    def adaptive_calibration(self):
        """
        Realiza calibração adaptativa com base no uso contínuo.
        Esta função é executada em uma thread separada.
        """
        print("Iniciando calibração adaptativa...")
        
        # Contador de amostras para análise
        sample_count = 0
        baseline_sum = 0
        
        while True:
            time.sleep(5)  # Verifica a cada 5 segundos
            
            # Se houver dados de baseline recentes
            if 'baseline' in self.calibration_values:
                baseline_sum += self.calibration_values['baseline']
                sample_count += 1
                
                # A cada 12 amostras (aproximadamente 1 minuto)
                if sample_count >= 12:
                    # Calcula média dos baselines
                    avg_baseline = baseline_sum / sample_count
                    
                    # Verifica se há desvio significativo
                    if 'baseline' in self.user_profile:
                        stored_baseline = self.user_profile['baseline']
                        deviation = abs(avg_baseline - stored_baseline) / stored_baseline
                        
                        # Se o desvio for maior que 15%, sugere recalibração
                        if deviation > 0.15:
                            print("Desvio significativo detectado. Sugerindo recalibração...")
                            self.send_command("SUGGEST_CALIBRATION")
                    
                    # Atualiza o perfil do usuário
                    self.user_profile['baseline'] = avg_baseline
                    
                    # Reinicia contadores
                    sample_count = 0
                    baseline_sum = 0
    
    def run(self):
        """Executa o loop principal do controlador."""
        print("Iniciando controlador da prótese...")
        if self.simulate:
            print("🔁 Executando loop em MODO SIMULADO com gráfico ao vivo...")

            signal_data = []

            fig, ax = plt.subplots()
            line, = ax.plot([], [], lw=2)
            text_pred = ax.text(0.02, 0.95, '', transform=ax.transAxes)

            def init():
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 150)
                return line, text_pred

            def update(frame):
                fake_value = np.random.randint(10, 100)
                self.data_buffer.append(fake_value)
                signal_data.append(fake_value)
                if len(signal_data) > 100:
                    signal_data.pop(0)
                line.set_data(range(len(signal_data)), signal_data)

                if len(self.data_buffer) >= self.window_size:
                    movement, confidence = self.predict_movement()
                    text_pred.set_text(f'Movimento: {movement}\nConfiança: {confidence:.2f}')
                    text_pred.set_color("red")
                    text_pred.set_fontsize(12)
                    print(f"[SIMULADO] Previsão → {movement}  | Confiança: {confidence:.2f}")

                return line, text_pred

            ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=200, cache_frame_data=False)
            plt.title("Sinal EMG Simulado e Predição de Movimento")
            plt.xlabel("Tempo")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            plt.show()
            return

        # Inicia thread de calibração adaptativa
        adaptive_thread = threading.Thread(target=self.adaptive_calibration, daemon=True)
        adaptive_thread.start()

        try:
            while True:
                # Lê dados do sensor
                data = self.read_serial_data()
                if data:
                    raw_value, smoothed_value, baseline, threshold = data

                    # Adiciona ao buffer
                    self.data_buffer.append(smoothed_value)

                    # Prediz movimento se o buffer estiver cheio
                    if len(self.data_buffer) >= self.window_size:
                        movement, confidence = self.predict_movement()

                        # Envia comando para o motor
                        self.send_motor_command(movement, confidence)

                        # Atualiza movimento atual
                        self.current_movement = movement

                # Pequena pausa para não sobrecarregar a CPU
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\nPrograma interrompido pelo usuário")
        finally:
            # Garante que o motor seja desligado ao sair
            self.send_command("MOTOR_STOP")
            if self.ser is not None:
                self.ser.close()
                print("Conexão serial fechada")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description='Biomove Prosthesis Controller')
    parser.add_argument('--port', type=str, default='/dev/ttyACM0',
                        help='Porta serial do Arduino')
    parser.add_argument('--baud', type=int, default=115200,
                        help='Taxa de comunicação serial')
    parser.add_argument('--model', type=str, default=None,
                        help='Caminho para o modelo treinado')
    parser.add_argument('--model-type', type=str, default='svm', choices=['svm', 'mlp', 'cnn'],
                        help='Tipo de modelo')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Limiar de confiança para ativação')
    parser.add_argument('--profile', type=str, default=None,
                        help='Caminho para o perfil de usuário')
    
    args = parser.parse_args()
    
    try:
        controller = ProsthesisController(
            port=args.port,
            baud_rate=args.baud,
            model_path=args.model,
            model_type=args.model_type,
            threshold=args.threshold
        )
        
        # Carrega perfil de usuário se especificado
        if args.profile and os.path.exists(args.profile):
            controller.load_user_profile(args.profile)
        
        # Executa o controlador
        controller.run()
        
    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    main()
