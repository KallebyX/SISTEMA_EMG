# 🧠 SISTEMA EMG – BIOMOVE

<p align="center">
  <img src="docs/pt-br/assets/biomove_logo.jpeg" width="200" alt="Biomove Logo"/>
</p>

<p align="center">
  <img alt="Versão" src="https://img.shields.io/badge/vers%C3%A3o-1.0.0-blue?style=for-the-badge">
  <img alt="Licença MIT" src="https://img.shields.io/badge/licença-MIT-green?style=for-the-badge">
  <img alt="Documentação" src="https://img.shields.io/badge/wiki-disponível-lightgrey?style=for-the-badge">
  <img alt="Deploy" src="https://img.shields.io/badge/deploy-GitHub%20Pages-success?style=for-the-badge">
</p>

<p align="center">
  <img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/KallebyX/SISTEMA_EMG/gh-pages.yml?branch=main&label=build&style=for-the-badge">
  <img alt="Último commit" src="https://img.shields.io/github/last-commit/KallebyX/SISTEMA_EMG?style=for-the-badge">
  <img alt="Contribuições" src="https://img.shields.io/github/contributors/KallebyX/SISTEMA_EMG?style=for-the-badge">
  <img alt="Estrelas do GitHub" src="https://img.shields.io/github/stars/KallebyX/SISTEMA_EMG?style=for-the-badge">
  <img alt="Testes Automatizados" src="https://img.shields.io/badge/testes-automatizados-blueviolet?style=for-the-badge">
</p>

<p align="center"><strong>Controle inteligente de próteses mioelétricas com IA e sinais EMG</strong></p>

<p align="center">
  <a href="https://kallebyx.github.io/SISTEMA_EMG/pt-br/assets/biomove_documentacao_institucional.pdf">📘 PDF Institucional</a> •
  <a href="https://kallebyx.github.io/SISTEMA_EMG/pt-br/assets/sistema_emg_documentacao_final.pdf">📄 Documentação Técnica</a> •
  <a href="https://doi.org/10.37779/nt.v25i3.5214">📚 Artigo Científico</a>
</p>

<p align="center">
  <img src="docs/pt-br/assets/qr_biomove_pdf.png" width="130" alt="QR Code PDF Institucional">
</p>

---

## 🌐 Visão Geral

O **Sistema EMG da Biomove** é uma solução integrada para aquisição, processamento e classificação de sinais eletromiográficos (EMG), voltada para o controle de próteses mioelétricas acessíveis. Desenvolvido com Arduino, MyoWare 2.0, Python e machine learning, ele oferece uma alternativa de baixo custo e alto impacto social.

---

## 🚀 Tecnologias e Funcionalidades

- Captação de sinais EMG com sensor MyoWare 2.0
- Processamento digital com filtros (notch, passa-alta, passa-baixa)
- Extração de características com janelas e estatísticas
- Classificação com SVM, MLP, CNN e futuramente LSTM
- Controle de prótese real via Arduino e motores
- Modo de simulação com datasets públicos (Ninapro, Physionet, EMG-UKA)
- Interface gráfica gamificada com DearPyGui
- Segurança: timeout, chaves fim de curso, limiar de confiança
- Modularidade e expansibilidade com sensores IMU e conectividade futura

---

## 📚 Validação Científica

> **Desenvolvimento de um Sistema de Classificação de Movimentos da Mão Baseado em Sinais EMG Utilizando Aprendizado de Máquina**  
> *Disciplinarum Scientia – Série Naturais e Tecnológicas*, UFN – v. 25, n. 3, 2024  
> MOTA, K.E. et al.  
> [📖 Leia o artigo](https://doi.org/10.37779/nt.v25i3.5214)

---

## 🧠 Modos de Operação

- **Simulado**: utiliza sinais sintéticos ou de bancos públicos
- **Coleta**: registra sinais reais com Arduino + MyoWare
- **Treinamento**: treina modelos com dados rotulados
- **Execução**: usa modelos para controle em tempo real da prótese física

---

## 🏗️ Estrutura do Projeto

```
SISTEMA_EMG/
├── src/
│   ├── acquisition/
│   ├── processing/
│   ├── ml/
│   ├── control/
│   └── gui/
├── data/
├── models/
├── docs/
├── tests/
├── scripts/
├── main.py
└── requirements.txt
```

---

## 📦 Instalação

```bash
git clone https://github.com/KallebyX/SISTEMA_EMG.git
cd SISTEMA_EMG
pip install -r requirements.txt
```

---

## ▶️ Execução

```bash
python main.py
```

O sistema inicia em modo simulado por padrão. Use argumentos CLI para selecionar portas, datasets ou modelos.

---

## 📘 Exemplos

### Classificação de gesto com SVM

```python
from src.acquisition import synthetic_generator
from src.processing import filters, feature_extraction
from src.ml.models import svm_model
import numpy as np

signal = synthetic_generator.generate_synthetic_emg(duration=5, fs=1000)
filtered = filters.apply_all_filters(signal, fs=1000)
features = feature_extraction.extract_features(filtered)
model = svm_model.SVMModel()
model.load("models/default_svm.pkl")
print("Gesto:", model.predict([list(features.values())]))
```

---

### Visualização com DearPyGui

```python
import dearpygui.dearpygui as dpg
from src.gui import signal_visualizer
from src.acquisition import synthetic_generator

dpg.create_context()
dpg.create_viewport(title="Visualização EMG", width=800, height=600)
dpg.setup_dearpygui()

signal = synthetic_generator.generate_synthetic_emg(duration=5, fs=1000)
visualizer = signal_visualizer.SignalVisualizer()
visualizer.setup()
visualizer.update(signal)

dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
```

---

## 🔐 Segurança

- Timeout programado para evitar acionamentos contínuos
- Fim de curso físico nas próteses reais
- Sistema de confiança baseado em predição por limiar

---

## 📘 Documentação

- [Manual do Usuário (PT)](docs/manual_usuario_pt.md)
- [User Manual (EN)](docs/user_manual_en.md)
- [Artigos técnicos](docs/artigos/)
- [PDF institucional](https://kallebyx.github.io/SISTEMA_EMG/pt-br/assets/biomove_documentacao_institucional.pdf)

---

## 🌍 Futuro da Plataforma

- Detecção de múltiplos gestos com LSTM
- Feedback háptico (vibração, retorno tátil)
- Controle por app mobile e conexão Bluetooth/Wi-Fi
- Plataforma online de telemetria e aprendizado

---

## 📄 Termos Legais

- [📘 Termos de Uso](TERMS_OF_USE.md)
- [🔐 Política de Privacidade](PRIVACY_POLICY.md)

---

## 🧾 Licença

Distribuído sob a Licença MIT.  
© 2025 Biomove & ORYUM TECH. Todos os direitos reservados.

---

## 📬 Contato

- Desenvolvedor: [Kalleby Evangelho Mota](mailto:kallebyevangelho03@gmail.com)  
- Instagram: [@kallebyevangelho](https://instagram.com/kallebyevangelho)  
- Repositório: [github.com/KallebyX/SISTEMA_EMG](https://github.com/KallebyX/SISTEMA_EMG)

---

## 🚀 Conheça a Startup Biomove

Acesse a [Landing Page da Biomove](https://kallebyx.github.io/Biomove) e descubra como estamos ampliando o acesso à tecnologia assistiva no Brasil com inovação, ciência e inclusão.

---
