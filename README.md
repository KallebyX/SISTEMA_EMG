# README - Sistema de Captação e Análise EMG Biomove

## Visão Geral

O Sistema de Captação e Análise EMG da Biomove é uma solução completa para aquisição, processamento, análise e classificação de sinais eletromiográficos (EMG), desenvolvido para controlar próteses mioelétricas de baixo custo. Este sistema utiliza o sensor MyoWare 2.0, Arduino, processamento em Python e algoritmos de machine learning para criar uma interface intuitiva entre o usuário e a prótese.

## Componentes do Sistema

O sistema é composto por quatro módulos principais:

1. **Captação de Sinal** - Código Arduino para leitura do sensor MyoWare 2.0
2. **Processamento e Visualização** - Scripts Python para filtragem e análise em tempo real
3. **Machine Learning** - Algoritmos para classificação de movimentos
4. **Controle da Prótese** - Sistema para acionamento do motor baseado nos movimentos detectados

## Requisitos de Hardware

- Arduino Mega ou compatível
- Sensor MyoWare 2.0
- Motor DC ou servo para acionamento da prótese
- Driver de motor (L298N ou similar)
- Chaves de fim de curso (opcional, para limites de movimento)
- Computador com Python 3.6+ para processamento e treinamento

## Requisitos de Software

- Arduino IDE
- Python 3.6+
- Bibliotecas Python:
  - numpy
  - scipy
  - matplotlib
  - pandas
  - scikit-learn
  - tensorflow (para modelos CNN)
  - pyserial

## Instalação

1. Clone este repositório:
```
git clone https://github.com/biomove/sistema-emg.git
cd sistema-emg
```

2. Instale as dependências Python:
```
pip install -r requirements.txt
```

3. Carregue os códigos Arduino:
   - `arduino_myoware.ino` para o Arduino conectado ao sensor MyoWare 2.0
   - `arduino_motor_control.ino` para o Arduino que controla o motor da prótese

## Estrutura de Arquivos

```
sistema_emg/
├── arduino_myoware.ino          # Código Arduino para leitura do sensor
├── arduino_motor_control.ino    # Código Arduino para controle do motor
├── emg_processor.py             # Processamento e visualização de sinais
├── emg_classifier.py            # Treinamento de modelos de machine learning
├── prosthesis_controller.py     # Controle da prótese baseado em classificação
├── dados_treinamento/           # Diretório para dados de treinamento
├── modelos_treinados/           # Diretório para modelos treinados
└── README.md                    # Este arquivo
```

## Uso do Sistema

### 1. Captação de Sinal

Carregue o código `arduino_myoware.ino` no Arduino conectado ao sensor MyoWare 2.0. Este código:
- Realiza a leitura analógica do sensor
- Aplica média móvel para suavização inicial
- Implementa calibração via botão
- Envia dados pela porta serial

Conexões do sensor MyoWare 2.0:
- VCC: 5V do Arduino
- GND: GND do Arduino
- SIG: Pino analógico A0 do Arduino

### 2. Processamento e Visualização

Execute o script `emg_processor.py` para:
- Receber dados do Arduino via serial
- Aplicar filtros digitais (passa-alta, passa-baixa, notch)
- Visualizar sinais em tempo real
- Gravar amostras para treinamento

```
python emg_processor.py --port /dev/ttyACM0
```

Comandos disponíveis:
- `r <label>` - Iniciar gravação com rótulo (ex: "r mao_fechada")
- `s` - Parar gravação
- `v` - Visualizar em tempo real
- `q` - Sair

### 3. Machine Learning

Execute o script `emg_classifier.py` para:
- Carregar dados de treinamento
- Extrair características dos sinais
- Treinar modelos (SVM, MLP, CNN)
- Avaliar desempenho
- Exportar modelos treinados

```
python emg_classifier.py
```

Os modelos serão salvos no diretório `modelos_treinados/` em formatos:
- `.pkl` para modelos scikit-learn (SVM, MLP)
- `.h5` para modelos TensorFlow/Keras (CNN)
- `.tflite` para uso em dispositivos embarcados

### 4. Controle da Prótese

Carregue o código `arduino_motor_control.ino` no Arduino que controla o motor da prótese.

Execute o script `prosthesis_controller.py` para:
- Carregar o modelo treinado
- Processar sinais EMG em tempo real
- Classificar movimentos
- Enviar comandos para o Arduino controlar o motor

```
python prosthesis_controller.py --port /dev/ttyACM0 --model modelos_treinados/svm_model_20250519_123456.pkl
```

Parâmetros:
- `--port`: Porta serial do Arduino
- `--model`: Caminho para o modelo treinado
- `--model-type`: Tipo de modelo (svm, mlp, cnn)
- `--threshold`: Limiar de confiança para ativação (0-1)
- `--profile`: Caminho para perfil de usuário (opcional)

## Filtros Implementados

O sistema implementa três tipos de filtros digitais:

1. **Filtro Notch (rejeita-faixa)**: Remove ruído da rede elétrica (60Hz)
2. **Filtro Passa-alta**: Remove offset DC e artefatos de movimento de baixa frequência (>20Hz)
3. **Filtro Passa-baixa**: Suaviza o sinal e remove ruídos de alta frequência (<450Hz)

## Características Extraídas para Machine Learning

Para cada janela de sinal EMG, são extraídas as seguintes características:

1. Média
2. Desvio padrão
3. RMS (Root Mean Square)
4. Valor máximo
5. Valor mínimo
6. Amplitude (max-min)
7. Assimetria (skewness)
8. Curtose (kurtosis)
9. Cruzamentos por zero
10. Energia do sinal

## Calibração Adaptativa

O sistema implementa calibração adaptativa que:
- Monitora o baseline do sinal EMG ao longo do tempo
- Detecta desvios significativos
- Sugere recalibração quando necessário
- Armazena perfis de usuário para personalização

## Segurança

Mecanismos de segurança implementados:
- Timeout de ativação contínua do motor
- Chaves de fim de curso para limites de movimento
- Verificação de confiança mínima para ativação
- Monitoramento de falhas de comunicação

## Sugestões para Expansão

### 1. Múltiplos Canais EMG
- Adicionar mais sensores MyoWare para captar diferentes grupos musculares
- Implementar fusão de dados para movimentos mais complexos
- Expandir para controle de dedos individuais

### 2. Feedback Sensorial
- Adicionar sensores de pressão na prótese
- Implementar feedback tátil para o usuário (vibradores, estimulação elétrica)
- Criar loop fechado de controle com feedback

### 3. Interface Wireless
- Substituir comunicação serial por Bluetooth ou WiFi
- Desenvolver app mobile para controle e monitoramento
- Implementar telemetria e diagnóstico remoto

### 4. Algoritmos Avançados
- Implementar redes neurais recorrentes (LSTM) para reconhecimento de sequências
- Adicionar aprendizado por reforço para adaptação contínua
- Explorar técnicas de transfer learning para reduzir tempo de treinamento

### 5. Integração com Outros Sensores
- Adicionar IMU (acelerômetro/giroscópio) para detecção de posição
- Implementar visão computacional para controle assistido
- Explorar interfaces cérebro-máquina para casos avançados

## Resolução de Problemas

### O Arduino não é detectado
- Verifique se o cabo USB está conectado corretamente
- Confirme se o driver FTDI está instalado
- Tente uma porta USB diferente

### Sinal EMG com muito ruído
- Verifique a preparação da pele (limpar com álcool)
- Confirme o posicionamento correto dos eletrodos
- Verifique a integridade dos cabos e conexões
- Ajuste os parâmetros dos filtros digitais

### Classificação imprecisa
- Colete mais amostras de treinamento
- Verifique a consistência dos movimentos durante o treinamento
- Experimente diferentes algoritmos de classificação
- Ajuste os hiperparâmetros do modelo

### Motor não responde
- Verifique as conexões do driver do motor
- Confirme se a fonte de alimentação é adequada
- Teste o motor diretamente com comandos simples
- Verifique o limiar de confiança para ativação

## Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## Contato

Para mais informações, entre em contato com a equipe Biomove:
- Email: kallebyevangelho03@gmail.com
