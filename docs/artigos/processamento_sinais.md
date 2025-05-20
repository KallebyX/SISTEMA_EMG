# Processamento de Sinais EMG

## Introdução

O processamento de sinais eletromiográficos (EMG) é uma etapa fundamental para a extração de informações relevantes que permitam o controle eficiente de próteses mioelétricas. Este documento detalha as técnicas de processamento implementadas no SISTEMA_EMG, abrangendo desde a aquisição do sinal bruto até a extração de características para classificação.

## Aquisição de Sinais EMG

### Princípios da Eletromiografia

A eletromiografia é a técnica de registro da atividade elétrica produzida pelos músculos durante sua contração. Os potenciais de ação das unidades motoras (PAUMs) são captados por eletrodos posicionados sobre a pele (EMG de superfície) ou inseridos diretamente no músculo (EMG intramuscular).

No SISTEMA_EMG, utilizamos a eletromiografia de superfície (sEMG) por ser não invasiva e adequada para aplicações de controle de próteses. Os sinais sEMG típicos apresentam:

- Amplitude: 0.1 a 5 mV
- Frequência: 10 a 500 Hz
- Componente DC: Presente devido à interface eletrodo-pele
- Ruídos: Interferência da rede elétrica (60 Hz), artefatos de movimento, crosstalk

### Hardware de Aquisição

O SISTEMA_EMG utiliza o sensor MyoWare 2.0 conectado a um Arduino para aquisição dos sinais EMG. O MyoWare 2.0 é um sensor EMG de superfície que inclui:

- Amplificação diferencial
- Filtragem analógica inicial
- Retificação do sinal (opcional)
- Envelope RMS (opcional)

A configuração do hardware segue o esquema:

1. Eletrodos de superfície → MyoWare 2.0 → Arduino (pino analógico)
2. Arduino → Computador (via USB)

O código de aquisição no Arduino:

```cpp
const int emgPin = A0;      // Pino analógico conectado ao MyoWare
const int sampleRate = 1000; // Taxa de amostragem em Hz
const long interval = 1000000 / sampleRate; // Intervalo em microssegundos

void setup() {
  Serial.begin(115200);
}

void loop() {
  static unsigned long lastMicros = 0;
  unsigned long currentMicros = micros();
  
  if (currentMicros - lastMicros >= interval) {
    lastMicros = currentMicros;
    
    // Lê o valor do sensor EMG (0-1023)
    int emgValue = analogRead(emgPin);
    
    // Envia para o computador
    Serial.println(emgValue);
  }
}
```

## Pré-processamento

### Remoção de Offset DC

O sinal EMG bruto frequentemente apresenta um offset DC devido à interface eletrodo-pele. Este offset é removido subtraindo a média do sinal:

```python
def detrend_signal(signal_data):
    """
    Remove o offset DC do sinal EMG.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
    
    Returns:
        numpy.ndarray: Sinal sem offset DC.
    """
    return signal_data - np.mean(signal_data)
```

### Normalização

A normalização é importante para garantir que os sinais estejam em uma escala consistente, especialmente quando se utiliza múltiplos canais ou sessões:

```python
def normalize_signal(signal_data, method='zscore'):
    """
    Normaliza o sinal EMG.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
        method (str): Método de normalização ('zscore', 'minmax', 'maxabs').
    
    Returns:
        numpy.ndarray: Sinal normalizado.
    """
    if method == 'zscore':
        # Normalização Z-score (média 0, desvio padrão 1)
        return (signal_data - np.mean(signal_data)) / np.std(signal_data)
    
    elif method == 'minmax':
        # Normalização Min-Max (escala 0-1)
        min_val = np.min(signal_data)
        max_val = np.max(signal_data)
        return (signal_data - min_val) / (max_val - min_val)
    
    elif method == 'maxabs':
        # Normalização pelo valor máximo absoluto (escala -1 a 1)
        max_abs = np.max(np.abs(signal_data))
        return signal_data / max_abs
    
    else:
        raise ValueError(f"Método de normalização não suportado: {method}")
```

## Filtragem Digital

### Filtro Passa-Banda

O filtro passa-banda remove componentes de frequência fora da faixa de interesse do sinal EMG (tipicamente 10-500 Hz):

```python
def bandpass_filter(signal_data, lowcut=10, highcut=500, fs=1000, order=4):
    """
    Aplica um filtro passa-banda ao sinal EMG.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
        lowcut (float): Frequência de corte inferior em Hz.
        highcut (float): Frequência de corte superior em Hz.
        fs (float): Frequência de amostragem em Hz.
        order (int): Ordem do filtro.
    
    Returns:
        numpy.ndarray: Sinal filtrado.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Garante que as frequências estão no intervalo válido
    low = max(0.001, min(low, 0.99))
    high = max(low + 0.001, min(high, 0.99))
    
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, signal_data)
```

### Filtro Notch

O filtro notch (rejeita-banda) é utilizado para remover a interferência da rede elétrica (60 Hz no Brasil):

```python
def notch_filter(signal_data, freq=60, q=30, fs=1000):
    """
    Aplica um filtro notch para remover interferência da rede elétrica.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
        freq (float): Frequência a ser removida em Hz.
        q (float): Fator de qualidade do filtro.
        fs (float): Frequência de amostragem em Hz.
    
    Returns:
        numpy.ndarray: Sinal filtrado.
    """
    nyq = 0.5 * fs
    freq_norm = freq / nyq
    
    # Garante que a frequência está no intervalo válido
    freq_norm = max(0.001, min(freq_norm, 0.99))
    
    b, a = signal.iirnotch(freq_norm, q)
    return signal.filtfilt(b, a, signal_data)
```

### Filtro Passa-Alta

O filtro passa-alta remove componentes de baixa frequência, incluindo artefatos de movimento e drift da linha de base:

```python
def highpass_filter(signal_data, cutoff=10, fs=1000, order=4):
    """
    Aplica um filtro passa-alta ao sinal EMG.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
        cutoff (float): Frequência de corte em Hz.
        fs (float): Frequência de amostragem em Hz.
        order (int): Ordem do filtro.
    
    Returns:
        numpy.ndarray: Sinal filtrado.
    """
    nyq = 0.5 * fs
    cutoff_norm = cutoff / nyq
    
    # Garante que a frequência está no intervalo válido
    cutoff_norm = max(0.001, min(cutoff_norm, 0.99))
    
    b, a = signal.butter(order, cutoff_norm, btype='high')
    return signal.filtfilt(b, a, signal_data)
```

### Filtro Passa-Baixa

O filtro passa-baixa remove componentes de alta frequência, suavizando o sinal e reduzindo ruídos:

```python
def lowpass_filter(signal_data, cutoff=500, fs=1000, order=4):
    """
    Aplica um filtro passa-baixa ao sinal EMG.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
        cutoff (float): Frequência de corte em Hz.
        fs (float): Frequência de amostragem em Hz.
        order (int): Ordem do filtro.
    
    Returns:
        numpy.ndarray: Sinal filtrado.
    """
    nyq = 0.5 * fs
    cutoff_norm = cutoff / nyq
    
    # Garante que a frequência está no intervalo válido
    cutoff_norm = max(0.001, min(cutoff_norm, 0.99))
    
    b, a = signal.butter(order, cutoff_norm, btype='low')
    return signal.filtfilt(b, a, signal_data)
```

## Segmentação e Janelamento

### Janelamento

O janelamento é aplicado para reduzir o efeito de borda na análise espectral e extração de características:

```python
def apply_window(signal_data, window_type='hamming'):
    """
    Aplica uma janela ao sinal.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
        window_type (str): Tipo de janela ('hamming', 'hanning', 'blackman', 'rectangular').
    
    Returns:
        numpy.ndarray: Sinal com janela aplicada.
    """
    if window_type == 'rectangular':
        window = np.ones_like(signal_data)
    else:
        if window_type == 'hamming':
            window = np.hamming(len(signal_data))
        elif window_type == 'hanning':
            window = np.hanning(len(signal_data))
        elif window_type == 'blackman':
            window = np.blackman(len(signal_data))
        else:
            raise ValueError(f"Tipo de janela não suportado: {window_type}")
    
    return signal_data * window
```

### Segmentação

A segmentação divide o sinal contínuo em janelas para análise:

```python
def segment_signal(signal_data, window_size=256, overlap=0.5):
    """
    Segmenta o sinal em janelas com sobreposição.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
        window_size (int): Tamanho da janela em amostras.
        overlap (float): Fração de sobreposição entre janelas (0 a 1).
    
    Returns:
        list: Lista de janelas (arrays numpy).
    """
    # Verifica parâmetros
    if overlap < 0 or overlap >= 1:
        raise ValueError("A sobreposição deve estar entre 0 e 1 (exclusivo)")
    
    # Calcula o passo entre janelas
    step = int(window_size * (1 - overlap))
    
    # Segmenta o sinal
    windows = []
    for i in range(0, len(signal_data) - window_size + 1, step):
        window = signal_data[i:i + window_size]
        windows.append(window)
    
    return windows
```

### Janela Deslizante para Processamento em Tempo Real

Para aplicações em tempo real, implementamos um processador de janela deslizante:

```python
def create_sliding_window_processor(window_size=256, overlap=0.5, fs=1000):
    """
    Cria um processador de janela deslizante para processamento em tempo real.
    
    Args:
        window_size (int): Tamanho da janela em amostras.
        overlap (float): Fração de sobreposição entre janelas (0 a 1).
        fs (float): Frequência de amostragem em Hz.
    
    Returns:
        callable: Função para processar novas amostras.
    """
    # Buffer para armazenar amostras
    buffer = np.zeros(window_size)
    step = int(window_size * (1 - overlap))
    position = 0
    
    def process_new_samples(new_samples):
        """
        Processa novas amostras e retorna resultados quando uma janela completa está disponível.
        
        Args:
            new_samples (numpy.ndarray): Novas amostras de sinal EMG.
        
        Returns:
            list: Lista de dicionários com os resultados do processamento, ou lista vazia se nenhuma janela completa estiver disponível.
        """
        nonlocal buffer, position
        
        # Adiciona novas amostras ao buffer
        n_new = len(new_samples)
        if n_new >= window_size:
            # Se há amostras suficientes, usa apenas as mais recentes
            buffer = new_samples[-window_size:]
            position = 0
        else:
            # Desloca o buffer e adiciona novas amostras
            buffer = np.roll(buffer, -n_new)
            buffer[-n_new:] = new_samples
            position += n_new
        
        # Verifica se há janelas completas disponíveis
        results = []
        while position >= step:
            # Processa a janela atual
            result = process_window(buffer, fs=fs)
            results.append(result)
            
            # Atualiza a posição
            position -= step
        
        return results
    
    return process_new_samples
```

## Extração de Características

### Características no Domínio do Tempo

#### Root Mean Square (RMS)

O RMS é uma medida da amplitude do sinal EMG e está relacionado ao nível de ativação muscular:

```python
def calculate_rms(signal_data):
    """
    Calcula o valor RMS (Root Mean Square) do sinal.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
    
    Returns:
        float: Valor RMS.
    """
    return float(np.sqrt(np.mean(np.square(signal_data))))
```

#### Mean Absolute Value (MAV)

O MAV é outra medida da amplitude do sinal EMG:

```python
def calculate_mav(signal_data):
    """
    Calcula o valor médio absoluto (Mean Absolute Value) do sinal.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
    
    Returns:
        float: Valor MAV.
    """
    return float(np.mean(np.abs(signal_data)))
```

#### Variância

A variância indica a dispersão do sinal EMG:

```python
def calculate_variance(signal_data):
    """
    Calcula a variância do sinal.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
    
    Returns:
        float: Variância.
    """
    return float(np.var(signal_data))
```

#### Waveform Length (WL)

O WL é uma medida da complexidade do sinal EMG:

```python
def calculate_waveform_length(signal_data):
    """
    Calcula o comprimento da forma de onda (Waveform Length) do sinal.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
    
    Returns:
        float: Comprimento da forma de onda.
    """
    return float(np.sum(np.abs(np.diff(signal_data))))
```

#### Zero Crossings (ZC)

O ZC conta quantas vezes o sinal cruza o zero, indicando o conteúdo de frequência:

```python
def calculate_zero_crossings(signal_data, threshold=0.0):
    """
    Calcula o número de cruzamentos por zero (Zero Crossings) do sinal.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
        threshold (float): Limiar para reduzir o efeito do ruído.
    
    Returns:
        int: Número de cruzamentos por zero.
    """
    # Conta os cruzamentos por zero
    zero_crossings = np.sum(np.diff(np.signbit(signal_data)) != 0)
    
    return int(zero_crossings)
```

#### Slope Sign Changes (SSC)

O SSC conta as mudanças de sinal da inclinação do sinal:

```python
def calculate_slope_sign_changes(signal_data, threshold=0.01):
    """
    Calcula o número de mudanças de sinal da inclinação (Slope Sign Changes) do sinal.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
        threshold (float): Limiar para reduzir o efeito do ruído.
    
    Returns:
        int: Número de mudanças de sinal da inclinação.
    """
    diff_signal = np.diff(signal_data)
    
    # Calcula as mudanças de sinal da inclinação
    signs = np.diff(np.sign(diff_signal))
    
    # Conta apenas mudanças significativas (acima do limiar)
    if threshold > 0:
        count = np.sum(np.abs(signs) >= threshold)
    else:
        count = np.sum(np.abs(signs) > 0)
    
    return int(count)
```

#### Willison Amplitude (WAMP)

O WAMP conta o número de vezes que a diferença entre amostras consecutivas excede um limiar:

```python
def calculate_willison_amplitude(signal_data, threshold=0.1):
    """
    Calcula a amplitude de Willison (Willison Amplitude) do sinal.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
        threshold (float): Limiar para contagem.
    
    Returns:
        int: Amplitude de Willison.
    """
    diff_signal = np.diff(signal_data)
    return int(np.sum(np.abs(diff_signal) > threshold))
```

### Características no Domínio da Frequência

#### Análise Espectral

A análise espectral é realizada utilizando o periodograma de Welch:

```python
def calculate_frequency_features(signal_data, fs=1000):
    """
    Calcula características no domínio da frequência.
    
    Args:
        signal_data (numpy.ndarray): Sinal EMG.
        fs (float): Frequência de amostragem em Hz.
    
    Returns:
        dict: Dicionário com características no domínio da frequência.
    """
    # Calcula o espectro de potência
    f, Pxx = signal.welch(signal_data, fs=fs, nperseg=min(256, len(signal_data)))
    
    # Frequência média
    mean_freq = float(np.sum(f * Pxx) / np.sum(Pxx))
    
    # Frequência mediana
    total_power = np.sum(Pxx)
    cumulative_power = np.cumsum(Pxx)
    median_freq_idx = np.where(cumulative_power >= total_power / 2)[0][0]
    median_freq = float(f[median_freq_idx])
    
    # Potência em bandas específicas
    low_band = float(np.sum(Pxx[(f >= 10) & (f <= 50)]))
    mid_band = float(np.sum(Pxx[(f > 50) & (f <= 100)]))
    high_band = float(np.sum(Pxx[(f > 100) & (f <= 200)]))
    
    return {
        'mean_freq': mean_freq,
        'median_freq': median_freq,
        'low_band_power': low_band,
        'mid_band_power': mid_band,
        'high_band_power': high_band,
        'band_power_ratio': float(high_band / (low_band + 1e-10))
    }
```

## Processamento Completo

### Pipeline de Processamento

O SISTEMA_EMG implementa um pipeline completo de processamento que integra todas as etapas:

```python
def process_window(signal_data, fs=1000, apply_filtering=True, window_type='hamming'):
    """
    Processa uma janela de sinal EMG, aplicando filtros e extraindo características.
    
    Args:
        signal_data (numpy.ndarray): Janela de sinal EMG.
        fs (float): Frequência de amostragem em Hz.
        apply_filtering (bool): Se True, aplica filtros ao sinal.
        window_type (str): Tipo de janela a ser aplicada.
    
    Returns:
        dict: Dicionário com o sinal original, sinal filtrado e características extraídas.
    """
    # Cria o dicionário de resultado
    result = {
        'signal': signal_data,
        'timestamp': np.datetime64('now')
    }
    
    # Aplica filtros se solicitado
    if apply_filtering and len(signal_data) > 20:
        # Remove offset DC
        filtered_signal = detrend_signal(signal_data)
        
        # Aplica filtro notch para remover interferência da rede elétrica
        filtered_signal = notch_filter(filtered_signal, freq=60, fs=fs)
        
        # Aplica filtro passa-banda
        nyq = 0.5 * fs
        lowcut = min(10, nyq * 0.8)
        highcut = min(500, nyq * 0.9)
        filtered_signal = bandpass_filter(filtered_signal, lowcut=lowcut, highcut=highcut, fs=fs)
    else:
        filtered_signal = signal_data
    
    # Aplica janela
    windowed_signal = apply_window(filtered_signal, window_type=window_type)
    
    # Adiciona sinais processados ao resultado
    result['filtered_signal'] = filtered_signal
    result['windowed_signal'] = windowed_signal
    
    # Extrai características
    features = extract_features(windowed_signal, fs=fs)
    result['features'] = features
    
    return result
```

## Visualização de Sinais

### Visualização em Tempo Real

O SISTEMA_EMG implementa visualização em tempo real dos sinais EMG utilizando DearPyGui:

```python
def create_signal_plot(width=600, height=300):
    """
    Cria um gráfico para visualização de sinais EMG em tempo real.
    
    Args:
        width (int): Largura do gráfico em pixels.
        height (int): Altura do gráfico em pixels.
    
    Returns:
        int: ID do gráfico.
    """
    with dpg.plot(label="Sinal EMG", width=width, height=height, anti_aliased=True):
        # Eixos
        dpg.add_plot_axis(dpg.mvXAxis, label="Tempo (s)", tag="x_axis")
        dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude", tag="y_axis")
        
        # Séries de dados
        dpg.add_line_series([], [], label="Sinal Bruto", parent="y_axis", tag="raw_signal")
        dpg.add_line_series([], [], label="Sinal Filtrado", parent="y_axis", tag="filtered_signal")
        
        # Legenda
        dpg.add_plot_legend()
    
    return "plot"

def update_signal_plot(plot_id, raw_signal, filtered_signal, fs=1000):
    """
    Atualiza o gráfico com novos dados.
    
    Args:
        plot_id (str): ID do gráfico.
        raw_signal (numpy.ndarray): Sinal EMG bruto.
        filtered_signal (numpy.ndarray): Sinal EMG filtrado.
        fs (float): Frequência de amostragem em Hz.
    """
    # Cria vetor de tempo
    t = np.arange(len(raw_signal)) / fs
    
    # Atualiza as séries de dados
    dpg.set_value("raw_signal", [t, raw_signal])
    dpg.set_value("filtered_signal", [t, filtered_signal])
    
    # Ajusta os limites dos eixos
    dpg.set_axis_limits("x_axis", 0, t[-1])
    
    y_min = min(np.min(raw_signal), np.min(filtered_signal))
    y_max = max(np.max(raw_signal), np.max(filtered_signal))
    margin = (y_max - y_min) * 0.1
    dpg.set_axis_limits("y_axis", y_min - margin, y_max + margin)
```

## Conclusão

O processamento de sinais EMG é uma etapa crítica para o controle eficiente de próteses mioelétricas. O SISTEMA_EMG implementa um pipeline completo de processamento, desde a aquisição do sinal bruto até a extração de características para classificação, utilizando técnicas modernas e robustas.

A combinação de filtragem digital, segmentação e extração de características permite a obtenção de informações relevantes dos sinais EMG, que são então utilizadas pelos algoritmos de aprendizado de máquina para classificação dos gestos e controle da prótese.

O sistema é flexível e permite a personalização dos parâmetros de processamento de acordo com as necessidades específicas de cada aplicação, garantindo o melhor desempenho possível em diferentes cenários de uso.
