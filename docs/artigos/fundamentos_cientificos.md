# Documentação Científica - SISTEMA_EMG

## Introdução

O SISTEMA_EMG é uma plataforma avançada para aquisição, processamento e classificação de sinais eletromiográficos (EMG) com aplicação direta no controle de próteses mioelétricas. Este documento apresenta os fundamentos científicos que embasam o desenvolvimento do sistema, incluindo os princípios de eletromiografia, técnicas de processamento de sinais, algoritmos de aprendizado de máquina e métodos de controle de próteses.

## Fundamentos da Eletromiografia

### Origem dos Sinais EMG

Os sinais eletromiográficos (EMG) são potenciais elétricos gerados pela atividade muscular. Quando um músculo é ativado pelo sistema nervoso central, unidades motoras são recrutadas, resultando em potenciais de ação que se propagam ao longo das fibras musculares. Esses potenciais podem ser detectados por eletrodos posicionados na superfície da pele (EMG de superfície) ou inseridos diretamente no músculo (EMG intramuscular).

O sinal EMG de superfície, utilizado no SISTEMA_EMG, representa a soma espacial e temporal dos potenciais de ação das unidades motoras ativas sob a área dos eletrodos. As características deste sinal incluem:

- Amplitude: tipicamente entre 0-10 mV pico a pico
- Frequência: componentes principais entre 10-500 Hz
- Relação sinal-ruído: variável, dependendo de fatores como posicionamento dos eletrodos, impedância da pele e interferências externas

### Fatores que Influenciam os Sinais EMG

Diversos fatores podem influenciar a qualidade e as características dos sinais EMG:

1. **Anatômicos e Fisiológicos**:
   - Diâmetro das fibras musculares
   - Profundidade e localização das fibras ativas
   - Quantidade de tecido entre o músculo e o eletrodo
   - Velocidade de condução dos potenciais de ação
   - Recrutamento e taxa de disparo das unidades motoras

2. **Técnicos e Metodológicos**:
   - Tipo e configuração dos eletrodos
   - Distância intereletrodos
   - Localização dos eletrodos em relação às fibras musculares
   - Orientação dos eletrodos em relação às fibras
   - Impedância da interface eletrodo-pele

3. **Externos**:
   - Interferência da rede elétrica (60 Hz)
   - Artefatos de movimento
   - Crosstalk de músculos adjacentes
   - Equipamentos eletrônicos próximos

## Aquisição de Sinais EMG

### Hardware de Aquisição

O SISTEMA_EMG utiliza o sensor MyoWare 2.0 conectado a uma placa Arduino para aquisição de sinais EMG. O MyoWare 2.0 é um sensor EMG de superfície que incorpora:

- Amplificação diferencial com alto CMRR (Common Mode Rejection Ratio)
- Filtragem analógica inicial
- Retificação e suavização do sinal (opcional)
- Saída analógica proporcional à atividade muscular

A placa Arduino realiza a conversão analógico-digital (ADC) do sinal com as seguintes características:

- Resolução: 10-12 bits
- Taxa de amostragem: configurável, tipicamente 1000 Hz
- Interface serial para comunicação com o computador

### Posicionamento dos Eletrodos

O posicionamento correto dos eletrodos é crucial para a qualidade do sinal EMG. O SISTEMA_EMG segue as recomendações do SENIAM (Surface ElectroMyoGraphy for the Non-Invasive Assessment of Muscles):

- Eletrodos posicionados paralelamente às fibras musculares
- Distância intereletrodos de 2 cm
- Posicionamento no ventre muscular, evitando junções miotendinosas
- Eletrodo de referência posicionado em área eletricamente neutra

Para o controle de próteses de mão, os músculos comumente utilizados são:

- Flexor radial do carpo
- Extensor radial do carpo
- Flexor ulnar do carpo
- Extensor ulnar do carpo
- Flexor superficial dos dedos
- Extensor dos dedos

## Processamento de Sinais EMG

### Pré-processamento

O pré-processamento dos sinais EMG é essencial para remover ruídos e artefatos, melhorando a relação sinal-ruído. O SISTEMA_EMG implementa as seguintes técnicas:

1. **Filtragem Passa-Banda**: Remove componentes de frequência fora da faixa de interesse (10-500 Hz)
   ```python
   def bandpass_filter(signal, lowcut=10, highcut=500, fs=1000, order=4):
       nyq = 0.5 * fs
       low = lowcut / nyq
       high = highcut / nyq
       b, a = butter(order, [low, high], btype='band')
       return filtfilt(b, a, signal)
   ```

2. **Filtro Notch**: Remove interferência da rede elétrica (60 Hz)
   ```python
   def notch_filter(signal, freq=60, q=30, fs=1000):
       nyq = 0.5 * fs
       w0 = freq / nyq
       b, a = iirnotch(w0, q)
       return filtfilt(b, a, signal)
   ```

3. **Remoção de Tendência (Detrending)**: Elimina componentes de baixa frequência e drift da linha de base
   ```python
   def detrend_signal(signal):
       return signal - np.mean(signal)
   ```

4. **Normalização**: Padroniza a amplitude do sinal para comparação entre diferentes aquisições
   ```python
   def normalize_signal(signal):
       return (signal - np.mean(signal)) / np.std(signal)
   ```

### Extração de Características

A extração de características transforma o sinal EMG bruto em um conjunto de parâmetros que representam suas propriedades relevantes. O SISTEMA_EMG implementa características no domínio do tempo, frequência e tempo-frequência:

1. **Características no Domínio do Tempo**:
   - Valor RMS (Root Mean Square)
   - Valor Médio Absoluto (MAV)
   - Variância
   - Comprimento da Forma de Onda (WL)
   - Mudanças de Sinal (ZC)
   - Amplitude de Willison (WAMP)

2. **Características no Domínio da Frequência**:
   - Frequência Média
   - Frequência Mediana
   - Potência em Bandas Específicas
   - Razão de Potência entre Bandas

3. **Características no Domínio Tempo-Frequência**:
   - Coeficientes Wavelet
   - Transformada de Fourier de Curto Tempo (STFT)

Exemplo de implementação para características no domínio do tempo:

```python
def extract_time_domain_features(signal, window_size=256, window_step=128):
    features = {}
    windows = []
    
    # Segmentação em janelas
    for i in range(0, len(signal) - window_size + 1, window_step):
        window = signal[i:i+window_size]
        windows.append(window)
    
    # Extração de características para cada janela
    for i, window in enumerate(windows):
        # RMS
        features[f'rms_{i}'] = np.sqrt(np.mean(np.square(window)))
        
        # MAV
        features[f'mav_{i}'] = np.mean(np.abs(window))
        
        # Variância
        features[f'var_{i}'] = np.var(window)
        
        # WL
        features[f'wl_{i}'] = np.sum(np.abs(np.diff(window)))
        
        # ZC (com threshold para reduzir efeito do ruído)
        threshold = 0.01
        zc = np.sum(np.abs(np.diff(np.sign(window))) > threshold)
        features[f'zc_{i}'] = zc
        
        # WAMP
        threshold = 0.1
        wamp = np.sum(np.abs(np.diff(window)) > threshold)
        features[f'wamp_{i}'] = wamp
    
    return features
```

### Janelamento e Segmentação

A segmentação do sinal EMG em janelas é fundamental para o processamento em tempo real. O SISTEMA_EMG implementa duas abordagens:

1. **Janelamento Disjunto**: Janelas não se sobrepõem, maximizando a independência entre amostras
2. **Janelamento com Sobreposição**: Janelas se sobrepõem parcialmente, aumentando a resolução temporal

A escolha do tamanho da janela é um compromisso entre:
- Resolução temporal: janelas menores fornecem maior resolução
- Estabilidade das características: janelas maiores fornecem estimativas mais estáveis
- Atraso na classificação: janelas maiores introduzem maior atraso

O SISTEMA_EMG utiliza janelas de 200-300 ms com sobreposição de 50% como configuração padrão, balanceando esses fatores para controle de próteses em tempo real.

## Aprendizado de Máquina para Classificação de Gestos

### Algoritmos Implementados

O SISTEMA_EMG implementa diversos algoritmos de aprendizado de máquina para classificação de gestos a partir de sinais EMG:

1. **Support Vector Machine (SVM)**:
   - Kernel: RBF (Radial Basis Function)
   - Parâmetros otimizados via validação cruzada
   - Vantagens: boa generalização, robusto a outliers
   - Desvantagens: sensível à escolha de parâmetros, escalabilidade limitada

2. **Multilayer Perceptron (MLP)**:
   - Arquitetura: 3 camadas (entrada, oculta, saída)
   - Função de ativação: ReLU nas camadas ocultas, Softmax na saída
   - Otimizador: Adam
   - Vantagens: capacidade de modelar relações não-lineares complexas
   - Desvantagens: propenso a overfitting, sensível à inicialização

3. **Convolutional Neural Network (CNN)**:
   - Arquitetura: camadas convolucionais 1D seguidas de pooling e camadas densas
   - Vantagens: extração automática de características, invariância a translações
   - Desvantagens: requer mais dados de treinamento, computacionalmente intensivo

4. **Long Short-Term Memory (LSTM)**:
   - Arquitetura: células LSTM bidirecionais seguidas de camadas densas
   - Vantagens: modela dependências temporais, adequado para sinais sequenciais
   - Desvantagens: treinamento mais complexo, maior custo computacional

### Avaliação de Desempenho

O desempenho dos algoritmos é avaliado utilizando:

1. **Métricas**:
   - Acurácia: proporção de classificações corretas
   - Precisão: proporção de verdadeiros positivos entre os positivos preditos
   - Recall: proporção de verdadeiros positivos identificados corretamente
   - F1-Score: média harmônica entre precisão e recall
   - Matriz de confusão: visualização detalhada dos erros de classificação

2. **Protocolos de Validação**:
   - Validação cruzada k-fold (k=5)
   - Divisão treino-teste estratificada (80%-20%)
   - Validação entre sessões (treinamento e teste em dias diferentes)
   - Validação entre sujeitos (generalização para novos usuários)

### Estratégias para Melhorar o Desempenho

O SISTEMA_EMG implementa diversas estratégias para melhorar o desempenho da classificação:

1. **Seleção de Características**:
   - Análise de Componentes Principais (PCA)
   - Seleção baseada em importância (feature importance)
   - Eliminação recursiva de características

2. **Regularização**:
   - L1 e L2 para modelos lineares e redes neurais
   - Dropout para redes neurais
   - Early stopping para evitar overfitting

3. **Ensemble de Modelos**:
   - Votação majoritária entre diferentes classificadores
   - Stacking: meta-classificador treinado nas saídas de modelos base
   - Boosting: combinação sequencial de modelos

4. **Suavização Temporal**:
   - Filtro de média móvel nas predições
   - Rejeição de predições com baixa confiança
   - Histerese para evitar oscilações entre classes

## Controle de Próteses Mioelétricas

### Estratégias de Controle

O SISTEMA_EMG implementa diferentes estratégias para controle de próteses:

1. **Controle Direto**:
   - Mapeamento direto entre gestos classificados e comandos da prótese
   - Vantagens: intuitivo, baixa latência
   - Desvantagens: limitado a um conjunto discreto de movimentos

2. **Controle Proporcional**:
   - Amplitude do sinal EMG controla a velocidade ou força do movimento
   - Vantagens: controle mais natural e preciso
   - Desvantagens: requer calibração cuidadosa

3. **Controle Baseado em Padrões**:
   - Reconhecimento de padrões temporais nos sinais EMG
   - Vantagens: permite controle de múltiplos graus de liberdade
   - Desvantagens: maior complexidade, potencial atraso

### Interface com a Prótese INMOVE

A comunicação com a prótese INMOVE é realizada via protocolo serial com os seguintes comandos:

- "OPEN": Abre a mão protética
- "CLOSE": Fecha a mão protética
- "STOP": Para o movimento atual

O controlador implementa lógica de segurança:
- Timeout de segurança para comandos sem confirmação
- Filtragem de comandos espúrios
- Monitoramento de corrente para evitar sobrecarga

### Modo Simulado

O modo simulado permite o desenvolvimento e teste do sistema sem a necessidade da prótese física:

1. **Visualização 3D da Prótese Virtual**:
   - Renderização em tempo real dos movimentos da mão
   - Feedback visual imediato das classificações

2. **Bancos de Dados Públicos**:
   - Ninapro: Database for sEMG and kinematics
   - EMG-UKA: EMG Database from University of Koblenz-Landau
   - PhysioNet: EMG Database for Gesture Recognition

3. **Geração de Sinais Sintéticos**:
   - Modelos paramétricos baseados em características estatísticas
   - Adição controlada de ruído e artefatos
   - Simulação de fadiga muscular e variabilidade entre sessões

## Aplicações Clínicas e Educacionais

### Aplicações Clínicas

O SISTEMA_EMG pode ser utilizado em diversos contextos clínicos:

1. **Reabilitação**:
   - Biofeedback para reeducação muscular
   - Monitoramento do progresso da reabilitação
   - Adaptação de próteses às necessidades específicas do paciente

2. **Avaliação Neuromuscular**:
   - Quantificação da atividade muscular
   - Detecção de padrões anormais de ativação
   - Avaliação de fadiga muscular

3. **Personalização de Próteses**:
   - Ajuste fino dos parâmetros de controle
   - Adaptação a diferentes padrões de ativação muscular
   - Treinamento progressivo para uso de próteses

### Aplicações Educacionais

Como ferramenta educacional, o SISTEMA_EMG oferece:

1. **Visualização em Tempo Real**:
   - Demonstração dos princípios da eletromiografia
   - Visualização da relação entre contração muscular e sinal elétrico
   - Exploração interativa dos efeitos de diferentes gestos

2. **Experimentação**:
   - Testes de diferentes algoritmos de processamento
   - Comparação de estratégias de classificação
   - Desenvolvimento de novos métodos de controle

3. **Pesquisa**:
   - Plataforma para coleta e análise de dados
   - Prototipagem rápida de novos algoritmos
   - Validação de hipóteses sobre controle motor

## Desafios e Limitações

### Desafios Técnicos

1. **Variabilidade dos Sinais EMG**:
   - Variações entre sessões devido a posicionamento dos eletrodos
   - Fadiga muscular alterando características do sinal
   - Sudorese afetando a impedância eletrodo-pele

2. **Processamento em Tempo Real**:
   - Compromisso entre complexidade computacional e latência
   - Necessidade de algoritmos eficientes para dispositivos embarcados
   - Gerenciamento de recursos computacionais limitados

3. **Robustez a Condições Reais**:
   - Interferências eletromagnéticas em ambientes não controlados
   - Artefatos de movimento durante atividades diárias
   - Adaptação a diferentes níveis de atividade física

### Limitações Atuais

1. **Número de Gestos Reconhecíveis**:
   - Limitado pela quantidade de canais EMG
   - Crosstalk entre músculos próximos
   - Dificuldade em distinguir gestos sutilmente diferentes

2. **Feedback Sensorial**:
   - Ausência de propriocepção natural
   - Feedback visual como principal mecanismo de controle
   - Limitada integração de feedback tátil

3. **Adaptabilidade**:
   - Necessidade de recalibração periódica
   - Adaptação a mudanças fisiológicas de longo prazo
   - Personalização para diferentes usuários

## Direções Futuras

### Melhorias Técnicas

1. **Algoritmos Adaptativos**:
   - Aprendizado contínuo durante o uso
   - Adaptação automática a mudanças nas características do sinal
   - Transferência de aprendizado entre sessões

2. **Fusão de Sensores**:
   - Integração de EMG com sensores inerciais (IMU)
   - Incorporação de sensores de força e pressão
   - Visão computacional para controle contextual

3. **Miniaturização e Eficiência Energética**:
   - Desenvolvimento de hardware mais compacto
   - Otimização de algoritmos para baixo consumo
   - Sistemas embarcados autônomos

### Novas Funcionalidades

1. **Controle de Múltiplos Graus de Liberdade**:
   - Movimentos simultâneos de diferentes articulações
   - Controle independente de dedos individuais
   - Transições suaves entre diferentes posturas

2. **Interfaces Cérebro-Máquina Híbridas**:
   - Combinação de EMG com sinais EEG
   - Integração com interfaces neurais invasivas
   - Sistemas de controle multimodais

3. **Realidade Aumentada e Virtual**:
   - Treinamento imersivo para uso de próteses
   - Visualização avançada de atividade muscular
   - Gamificação da reabilitação

## Conclusão

O SISTEMA_EMG representa uma plataforma abrangente para aquisição, processamento e classificação de sinais EMG com aplicação direta no controle de próteses mioelétricas. Combinando hardware de aquisição de alta qualidade, algoritmos avançados de processamento de sinais e técnicas de aprendizado de máquina, o sistema oferece uma solução robusta tanto para aplicações clínicas quanto educacionais.

Os desafios técnicos e limitações atuais apontam para direções futuras promissoras, incluindo algoritmos adaptativos, fusão de sensores e novas funcionalidades que podem melhorar significativamente a experiência dos usuários de próteses mioelétricas.

## Referências

1. Oskoei, M. A., & Hu, H. (2007). Myoelectric control systems—A survey. Biomedical Signal Processing and Control, 2(4), 275-294.

2. Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012). Feature reduction and selection for EMG signal classification. Expert Systems with Applications, 39(8), 7420-7431.

3. Atzori, M., Gijsberts, A., Castellini, C., Caputo, B., Hager, A. G. M., Elsig, S., ... & Müller, H. (2014). Electromyography data for non-invasive naturally-controlled robotic hand prostheses. Scientific data, 1(1), 1-13.

4. Scheme, E., & Englehart, K. (2011). Electromyogram pattern recognition for control of powered upper-limb prostheses: State of the art and challenges for clinical use. Journal of Rehabilitation Research & Development, 48(6).

5. Farina, D., Jiang, N., Rehbaum, H., Holobar, A., Graimann, B., Dietl, H., & Aszmann, O. C. (2014). The extraction of neural information from the surface EMG for the control of upper-limb prostheses: emerging avenues and challenges. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 22(4), 797-809.

6. Côté-Allard, U., Fall, C. L., Drouin, A., Campeau-Lecours, A., Gosselin, C., Glette, K., ... & Gosselin, B. (2019). Deep learning for electromyographic hand gesture signal classification using transfer learning. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 27(4), 760-771.

7. Hargrove, L. J., Englehart, K., & Hudgins, B. (2007). A comparison of surface and intramuscular myoelectric signal classification. IEEE Transactions on Biomedical Engineering, 54(5), 847-853.

8. Zecca, M., Micera, S., Carrozza, M. C., & Dario, P. (2002). Control of multifunctional prosthetic hands by processing the electromyographic signal. Critical Reviews in Biomedical Engineering, 30(4-6).

9. Geng, Y., Samuel, O. W., Wei, Y., & Li, G. (2017). Improving the robustness of real-time myoelectric pattern recognition against varying hand positions. IEEE Transactions on Human-Machine Systems, 47(6), 1087-1096.

10. Jiang, N., Dosen, S., Müller, K. R., & Farina, D. (2012). Myoelectric control of artificial limbs—is there a need to change focus? IEEE Signal Processing Magazine, 29(5), 152-150.
