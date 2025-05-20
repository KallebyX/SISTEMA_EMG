# Manual do Usuário - SISTEMA_EMG

## Introdução

Bem-vindo ao SISTEMA_EMG, uma plataforma avançada para aquisição, processamento e classificação de sinais eletromiográficos (EMG) com aplicação direta no controle de próteses mioelétricas. Este manual fornece instruções detalhadas sobre como instalar, configurar e utilizar o sistema em seus diferentes modos de operação.

## Instalação

### Requisitos do Sistema

**Hardware:**
- Arduino Uno/Mega/Nano
- Sensor MyoWare 2.0
- Prótese INMOVE (opcional para modo físico)
- Computador com porta USB

**Software:**
- Python 3.8 ou superior
- Bibliotecas Python (instaladas automaticamente via requirements.txt)
- Sistema operacional: Windows 10/11, macOS, Linux

### Procedimento de Instalação

1. **Clone o repositório ou extraia o arquivo ZIP:**
   ```
   git clone https://github.com/seu-usuario/SISTEMA_EMG.git
   ```

2. **Navegue até o diretório do projeto:**
   ```
   cd SISTEMA_EMG
   ```

3. **Instale as dependências:**
   ```
   pip install -r requirements.txt
   ```

4. **Conecte o hardware (se estiver usando o modo físico):**
   - Conecte o Arduino ao computador via USB
   - Conecte o sensor MyoWare 2.0 ao Arduino conforme o diagrama abaixo:
     - Pino VCC do MyoWare → 5V do Arduino
     - Pino GND do MyoWare → GND do Arduino
     - Pino SIG do MyoWare → Pino analógico A0 do Arduino
   - Posicione os eletrodos no músculo alvo seguindo as instruções na seção "Posicionamento de Eletrodos"

## Iniciando o Sistema

Para iniciar o SISTEMA_EMG, execute o seguinte comando no terminal:

```
python main.py
```

Por padrão, o sistema será iniciado no modo simulado com interface gráfica. Para opções adicionais, consulte a seção "Opções de Linha de Comando".

## Modos de Operação

O SISTEMA_EMG possui quatro modos principais de operação:

### 1. Modo de Simulação

Este modo permite experimentar o sistema usando sinais EMG simulados ou de bancos de dados públicos, sem necessidade de hardware.

**Como usar:**
1. Selecione "Simulação" na interface
2. Escolha a fonte de dados:
   - **Ninapro**: Database for sEMG and kinematics
   - **EMG-UKA**: EMG Database from University of Koblenz-Landau
   - **PhysioNet**: EMG Database for Gesture Recognition
   - **Sintético**: Sinais gerados algoritmicamente
3. Ajuste o nível de ruído conforme necessário
4. Clique em "Iniciar" para começar a simulação
5. Observe a visualização dos sinais e o controle da prótese virtual

### 2. Modo de Coleta

Este modo permite coletar seus próprios dados EMG para treinamento personalizado.

**Como usar:**
1. Selecione "Coleta" na interface
2. Escolha o gesto a ser coletado:
   - Repouso
   - Mão aberta
   - Mão fechada
   - Pinça
   - Apontar
3. Defina a duração da coleta (recomendado: 5 segundos por gesto)
4. Clique em "Iniciar Coleta" e realize o gesto solicitado
5. Repita para todos os gestos desejados
6. Exporte os dados coletados para CSV clicando em "Exportar Dados"

### 3. Modo de Treinamento

Este modo permite treinar modelos de aprendizado de máquina com seus dados coletados.

**Como usar:**
1. Selecione "Treinamento" na interface
2. Clique em "Carregar Dados" e selecione o arquivo CSV com os dados coletados
3. Selecione os modelos a serem treinados:
   - SVM (Support Vector Machine)
   - MLP (Multi-Layer Perceptron)
   - CNN (Convolutional Neural Network)
   - LSTM (Long Short-Term Memory)
4. Configure os parâmetros de treinamento:
   - Proporção treino/teste (recomendado: 80/20)
   - Validação cruzada (recomendado: 5 folds)
   - Número de épocas (para redes neurais)
5. Clique em "Iniciar Treinamento"
6. Visualize os resultados e métricas de desempenho
7. Exporte os modelos treinados clicando em "Exportar Modelos"

### 4. Modo de Execução

Este modo permite controlar a prótese em tempo real usando os modelos treinados.

**Como usar:**
1. Selecione "Execução" na interface
2. Clique em "Carregar Modelo" e selecione o modelo treinado
3. Para modo físico:
   - Selecione a porta serial do Arduino
   - Clique em "Conectar"
4. Ajuste o limiar de confiança (recomendado: 0.7)
5. Clique em "Iniciar Execução"
6. Realize gestos para controlar a prótese (física ou virtual)
7. Observe a visualização dos sinais e a classificação em tempo real

## Posicionamento de Eletrodos

O posicionamento correto dos eletrodos é crucial para a qualidade do sinal EMG:

1. **Preparação da pele:**
   - Limpe a área com álcool isopropílico
   - Remova pelos se necessário
   - Deixe a pele secar completamente

2. **Posicionamento para controle de prótese de mão:**
   - **Flexor radial do carpo**: Posicione os eletrodos no terço proximal do antebraço, na face anterior
   - **Extensor radial do carpo**: Posicione os eletrodos no terço proximal do antebraço, na face posterior

3. **Orientação dos eletrodos:**
   - Posicione os eletrodos paralelamente às fibras musculares
   - Mantenha uma distância de 2 cm entre os eletrodos
   - Posicione no ventre muscular, evitando junções miotendinosas
   - O eletrodo de referência deve estar em área eletricamente neutra (ex: proeminência óssea)

## Opções de Linha de Comando

O SISTEMA_EMG oferece várias opções de linha de comando para personalizar sua execução:

```
python main.py [opções]
```

Opções disponíveis:
- `--mode {physical,simulated}`: Modo de operação (físico ou simulado)
- `--port PORTA`: Porta serial do Arduino (apenas para modo físico)
- `--dataset CAMINHO`: Caminho para o dataset a ser usado no modo simulado
- `--model CAMINHO`: Caminho para o modelo pré-treinado a ser carregado
- `--no-gui`: Executa o sistema sem interface gráfica (modo console)

Exemplo:
```
python main.py --mode physical --port COM3
```

## Solução de Problemas

### Problemas de Conexão com Arduino

**Problema**: O sistema não detecta o Arduino.
**Solução**: 
- Verifique se o Arduino está conectado corretamente
- Confirme se o driver USB está instalado
- Tente uma porta USB diferente
- Verifique se a porta serial está correta

### Sinais EMG de Baixa Qualidade

**Problema**: Os sinais EMG estão ruidosos ou fracos.
**Solução**:
- Verifique o posicionamento dos eletrodos
- Limpe a pele novamente
- Verifique as conexões do sensor MyoWare
- Substitua os eletrodos se estiverem secos
- Ajuste os parâmetros de filtragem no software

### Classificação Imprecisa

**Problema**: O sistema não classifica corretamente os gestos.
**Solução**:
- Colete mais dados de treinamento
- Experimente diferentes modelos de aprendizado de máquina
- Ajuste o limiar de confiança
- Verifique a consistência dos gestos durante o treinamento e execução
- Tente normalizar os sinais EMG

## Manutenção e Cuidados

### Eletrodos

- Substitua os eletrodos regularmente (recomendado: a cada sessão)
- Armazene os eletrodos em local fresco e seco
- Não reutilize eletrodos descartáveis

### Sensor MyoWare 2.0

- Limpe os contatos com álcool isopropílico quando necessário
- Evite dobrar ou torcer o sensor
- Armazene em local seco e protegido

### Arduino

- Mantenha o firmware atualizado
- Proteja contra descargas eletrostáticas
- Evite desconectar durante a operação

## Recursos Adicionais

### Bancos de Dados EMG Públicos

- **Ninapro**: [http://ninapro.hevs.ch/](http://ninapro.hevs.ch/)
- **EMG-UKA**: [https://www.uni-koblenz-landau.de/en/campus-koblenz/fb4/ist/rgdv/research/datasetstools/emg-dataset](https://www.uni-koblenz-landau.de/en/campus-koblenz/fb4/ist/rgdv/research/datasetstools/emg-dataset)
- **PhysioNet**: [https://physionet.org/content/emgdb/1.0.0/](https://physionet.org/content/emgdb/1.0.0/)

### Documentação Científica

Para informações detalhadas sobre os fundamentos científicos do sistema, consulte os seguintes documentos na pasta `docs/artigos/`:

- `fundamentos_cientificos.md`: Princípios de eletromiografia e controle de próteses
- `algoritmos_aprendizado_maquina.md`: Detalhes sobre os algoritmos implementados
- `processamento_sinais.md`: Técnicas de processamento de sinais EMG

## Suporte e Contato

Para questões, sugestões ou colaborações, entre em contato através de:
- Email: seu-email@exemplo.com
- GitHub: [seu-usuario](https://github.com/seu-usuario)

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.
