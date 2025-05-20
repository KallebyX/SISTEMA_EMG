"""
Algoritmos de Aprendizado de Máquina para Classificação de Sinais EMG

Este documento detalha os algoritmos de aprendizado de máquina implementados
no SISTEMA_EMG para classificação de gestos a partir de sinais eletromiográficos.
"""

# Algoritmos de Aprendizado de Máquina para Classificação de Sinais EMG

## Introdução

O SISTEMA_EMG implementa diversos algoritmos de aprendizado de máquina para classificação de gestos a partir de sinais eletromiográficos (EMG). Cada algoritmo possui características específicas que o tornam mais adequado para determinados cenários de uso. Este documento apresenta os fundamentos teóricos, implementação, vantagens e limitações de cada algoritmo.

## Support Vector Machine (SVM)

### Fundamentos Teóricos

O Support Vector Machine (SVM) é um algoritmo de aprendizado supervisionado que busca encontrar um hiperplano ótimo que separe as classes no espaço de características. Para problemas não linearmente separáveis, o SVM utiliza o "truque do kernel", que mapeia implicitamente os dados para um espaço de maior dimensão onde eles se tornam linearmente separáveis.

O SVM resolve o seguinte problema de otimização:

$$\min_{w, b, \xi} \frac{1}{2} w^T w + C \sum_{i=1}^{n} \xi_i$$

Sujeito a:
$$y_i (w^T \phi(x_i) + b) \geq 1 - \xi_i$$
$$\xi_i \geq 0, i = 1, ..., n$$

Onde:
- $w$ é o vetor normal ao hiperplano
- $b$ é o termo de viés
- $\xi_i$ são variáveis de folga para permitir erros de classificação
- $C$ é o parâmetro de regularização que controla o trade-off entre maximizar a margem e minimizar o erro de treinamento
- $\phi(x_i)$ é a função de mapeamento para o espaço de características de maior dimensão

### Implementação no SISTEMA_EMG

No SISTEMA_EMG, o SVM é implementado utilizando a biblioteca scikit-learn, com as seguintes configurações:

```python
from sklearn.svm import SVC

class SVMModel:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', probability=True):
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            decision_function_shape='ovr'
        )
        self.scaler = StandardScaler()
    
    def train(self, X, y):
        # Normaliza os dados
        X_scaled = self.scaler.fit_transform(X)
        
        # Treina o modelo
        self.model.fit(X_scaled, y)
        
        # Calcula a acurácia no conjunto de treinamento
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        
        return accuracy
    
    def predict(self, X):
        # Normaliza os dados
        X_scaled = self.scaler.transform(X)
        
        # Realiza a predição
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        # Normaliza os dados
        X_scaled = self.scaler.transform(X)
        
        # Retorna as probabilidades
        return self.model.predict_proba(X_scaled)
```

### Vantagens

- **Generalização**: O SVM é eficaz em espaços de alta dimensão, mesmo quando o número de dimensões é maior que o número de amostras.
- **Robustez**: É menos suscetível a overfitting em comparação com outros classificadores.
- **Versatilidade**: Diferentes funções de kernel permitem adaptar o algoritmo a diversos tipos de dados.
- **Eficiência de memória**: Usa apenas um subconjunto dos pontos de treinamento (vetores de suporte) na decisão.

### Limitações

- **Escalabilidade**: Não escala bem para grandes conjuntos de dados.
- **Sensibilidade a parâmetros**: O desempenho depende significativamente da escolha adequada de parâmetros como C e gamma.
- **Interpretabilidade**: Difícil de interpretar, especialmente com kernels não lineares.

## Multilayer Perceptron (MLP)

### Fundamentos Teóricos

O Multilayer Perceptron (MLP) é uma rede neural artificial feedforward que consiste em múltiplas camadas de neurônios: uma camada de entrada, uma ou mais camadas ocultas e uma camada de saída. Cada neurônio em uma camada está conectado a todos os neurônios da camada seguinte, formando uma rede totalmente conectada.

A saída de cada neurônio é calculada como:

$$y = \phi\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

Onde:
- $\phi$ é a função de ativação (geralmente ReLU, sigmoid ou tanh)
- $w_i$ são os pesos das conexões
- $x_i$ são as entradas
- $b$ é o termo de viés

O treinamento do MLP é realizado através do algoritmo de retropropagação (backpropagation), que ajusta os pesos para minimizar uma função de perda, geralmente usando o método do gradiente descendente.

### Implementação no SISTEMA_EMG

No SISTEMA_EMG, o MLP é implementado utilizando a biblioteca scikit-learn, com as seguintes configurações:

```python
from sklearn.neural_network import MLPClassifier

class MLPModel:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', 
                 alpha=0.0001, batch_size='auto', learning_rate='adaptive', 
                 max_iter=200, random_state=None):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=random_state
        )
        self.scaler = StandardScaler()
    
    def train(self, X, y):
        # Normaliza os dados
        X_scaled = self.scaler.fit_transform(X)
        
        # Treina o modelo
        self.model.fit(X_scaled, y)
        
        # Calcula a acurácia no conjunto de treinamento
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        
        return accuracy
    
    def predict(self, X):
        # Normaliza os dados
        X_scaled = self.scaler.transform(X)
        
        # Realiza a predição
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        # Normaliza os dados
        X_scaled = self.scaler.transform(X)
        
        # Retorna as probabilidades
        return self.model.predict_proba(X_scaled)
```

### Vantagens

- **Capacidade de modelagem**: Pode modelar relações não lineares complexas.
- **Adaptabilidade**: Pode ser aplicado a uma ampla variedade de problemas.
- **Aprendizado de características**: Aprende automaticamente representações hierárquicas dos dados.
- **Paralelização**: O treinamento pode ser facilmente paralelizado.

### Limitações

- **Overfitting**: Propenso a overfitting, especialmente com conjuntos de dados pequenos.
- **Sensibilidade à inicialização**: O desempenho pode variar significativamente dependendo da inicialização dos pesos.
- **Custo computacional**: Treinamento pode ser computacionalmente intensivo.
- **Hiperparâmetros**: Requer ajuste cuidadoso de múltiplos hiperparâmetros.

## Convolutional Neural Network (CNN)

### Fundamentos Teóricos

As Redes Neurais Convolucionais (CNNs) são uma classe especializada de redes neurais projetadas para processar dados com estrutura de grade, como séries temporais (1D) ou imagens (2D). As CNNs usam operações de convolução em vez de multiplicação de matriz geral em pelo menos uma de suas camadas.

A operação de convolução 1D é definida como:

$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t-\tau)d\tau$$

Na forma discreta, para sinais EMG:

$$(f * g)[n] = \sum_{m=-\infty}^{\infty} f[m]g[n-m]$$

As CNNs consistem tipicamente em:
- **Camadas convolucionais**: Aplicam filtros para extrair características locais
- **Camadas de pooling**: Reduzem a dimensionalidade e tornam a representação invariante a pequenas translações
- **Camadas totalmente conectadas**: Realizam a classificação final com base nas características extraídas

### Implementação no SISTEMA_EMG

No SISTEMA_EMG, a CNN é implementada utilizando TensorFlow/Keras:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

class CNNModel:
    def __init__(self, input_shape, num_classes, filters=[64, 128], 
                 kernel_size=3, pool_size=2, dense_units=100, dropout_rate=0.5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = StandardScaler()
        
        self._build_model()
    
    def _build_model(self):
        model = Sequential()
        
        # Primeira camada convolucional
        model.add(Conv1D(filters=self.filters[0], kernel_size=self.kernel_size, 
                         activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        
        # Segunda camada convolucional
        model.add(Conv1D(filters=self.filters[1], kernel_size=self.kernel_size, 
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        
        # Flatten e camadas densas
        model.add(Flatten())
        model.add(Dense(self.dense_units, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compilação
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        self.model = model
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        # Normaliza os dados
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape para formato 3D (amostras, timesteps, features)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Treina o modelo
        history = self.model.fit(
            X_reshaped, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        # Retorna a acurácia final no conjunto de treinamento
        return history.history['accuracy'][-1]
    
    def predict(self, X):
        # Normaliza os dados
        X_scaled = self.scaler.transform(X)
        
        # Reshape para formato 3D
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Realiza a predição
        y_pred = self.model.predict(X_reshaped)
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X):
        # Normaliza os dados
        X_scaled = self.scaler.transform(X)
        
        # Reshape para formato 3D
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Retorna as probabilidades
        return self.model.predict(X_reshaped)
```

### Vantagens

- **Extração automática de características**: Aprende automaticamente características relevantes dos sinais EMG.
- **Invariância a translações**: As operações de pooling tornam o modelo robusto a pequenas variações na posição temporal.
- **Redução de parâmetros**: O compartilhamento de parâmetros nas camadas convolucionais reduz o número total de parâmetros.
- **Hierarquia de características**: Aprende características de baixo nível nas primeiras camadas e de alto nível nas camadas posteriores.

### Limitações

- **Dados de treinamento**: Requer grandes quantidades de dados para treinamento eficaz.
- **Custo computacional**: Treinamento e inferência podem ser computacionalmente intensivos.
- **Overfitting**: Pode sofrer de overfitting em conjuntos de dados pequenos.
- **Interpretabilidade**: Difícil de interpretar as características aprendidas.

## Long Short-Term Memory (LSTM)

### Fundamentos Teóricos

As redes Long Short-Term Memory (LSTM) são um tipo especial de Rede Neural Recorrente (RNN) capaz de aprender dependências de longo prazo. As LSTMs foram projetadas para superar o problema do desvanecimento do gradiente que afeta as RNNs tradicionais.

A arquitetura LSTM inclui uma célula de memória com três "portões" que controlam o fluxo de informação:
- **Portão de esquecimento**: Decide quais informações da célula de memória devem ser descartadas.
- **Portão de entrada**: Decide quais novos valores serão armazenados na célula de memória.
- **Portão de saída**: Decide quais partes da célula de memória serão produzidas como saída.

As equações que governam uma célula LSTM são:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
$$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t * \tanh(C_t)$$

Onde:
- $f_t$ é o portão de esquecimento
- $i_t$ é o portão de entrada
- $\tilde{C}_t$ é o candidato a novo valor da célula
- $C_t$ é o estado da célula
- $o_t$ é o portão de saída
- $h_t$ é a saída da célula
- $\sigma$ é a função sigmoid
- $W$ e $b$ são os pesos e vieses aprendidos

### Implementação no SISTEMA_EMG

No SISTEMA_EMG, a LSTM é implementada utilizando TensorFlow/Keras:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout

class LSTMModel:
    def __init__(self, input_shape, num_classes, lstm_units=64, 
                 bidirectional=True, dense_units=100, dropout_rate=0.5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.bidirectional = bidirectional
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = StandardScaler()
        
        self._build_model()
    
    def _build_model(self):
        model = Sequential()
        
        # Camada LSTM (bidirecional ou não)
        if self.bidirectional:
            model.add(Bidirectional(LSTM(self.lstm_units, return_sequences=True), 
                                   input_shape=self.input_shape))
        else:
            model.add(LSTM(self.lstm_units, return_sequences=True, 
                          input_shape=self.input_shape))
        
        model.add(Dropout(self.dropout_rate))
        
        # Segunda camada LSTM
        if self.bidirectional:
            model.add(Bidirectional(LSTM(self.lstm_units)))
        else:
            model.add(LSTM(self.lstm_units))
        
        model.add(Dropout(self.dropout_rate))
        
        # Camadas densas
        model.add(Dense(self.dense_units, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compilação
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        self.model = model
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        # Normaliza os dados
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape para formato 3D (amostras, timesteps, features)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Treina o modelo
        history = self.model.fit(
            X_reshaped, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        # Retorna a acurácia final no conjunto de treinamento
        return history.history['accuracy'][-1]
    
    def predict(self, X):
        # Normaliza os dados
        X_scaled = self.scaler.transform(X)
        
        # Reshape para formato 3D
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Realiza a predição
        y_pred = self.model.predict(X_reshaped)
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X):
        # Normaliza os dados
        X_scaled = self.scaler.transform(X)
        
        # Reshape para formato 3D
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Retorna as probabilidades
        return self.model.predict(X_reshaped)
```

### Vantagens

- **Memória de longo prazo**: Capaz de capturar dependências temporais de longo prazo nos sinais EMG.
- **Robustez a variações temporais**: Lida bem com variações na duração dos gestos.
- **Bidirecionalidade**: A versão bidirecional considera tanto o contexto passado quanto o futuro.
- **Desempenho**: Geralmente supera outros modelos em tarefas de classificação de séries temporais.

### Limitações

- **Complexidade computacional**: Treinamento mais lento e computacionalmente intensivo.
- **Overfitting**: Requer técnicas de regularização como dropout para evitar overfitting.
- **Sensibilidade a hiperparâmetros**: O desempenho depende significativamente da escolha de hiperparâmetros.
- **Latência**: A natureza sequencial pode introduzir latência na inferência em tempo real.

## Ensemble de Modelos

### Fundamentos Teóricos

Os métodos de ensemble combinam as predições de múltiplos modelos base para melhorar o desempenho e a robustez. Existem várias estratégias de ensemble:

1. **Votação**: Combina as predições de diferentes modelos através de votação majoritária (para classificação) ou média (para regressão).
2. **Bagging**: Treina múltiplos modelos em subconjuntos aleatórios do conjunto de treinamento (ex: Random Forest).
3. **Boosting**: Treina modelos sequencialmente, com cada modelo focando nos erros dos anteriores (ex: AdaBoost, Gradient Boosting).
4. **Stacking**: Usa as predições de múltiplos modelos como entrada para um meta-modelo.

### Implementação no SISTEMA_EMG

No SISTEMA_EMG, implementamos um ensemble baseado em votação ponderada pela confiança:

```python
class EnsembleModel:
    def __init__(self, models=None, weights=None):
        self.models = models if models is not None else []
        self.weights = weights if weights is not None else []
        
        # Normaliza os pesos se fornecidos
        if self.weights:
            sum_weights = sum(self.weights)
            self.weights = [w / sum_weights for w in self.weights]
    
    def add_model(self, model, weight=1.0):
        self.models.append(model)
        self.weights.append(weight)
        
        # Renormaliza os pesos
        sum_weights = sum(self.weights)
        self.weights = [w / sum_weights for w in self.weights]
    
    def predict(self, X):
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Obtém predições de cada modelo
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        
        # Converte para array numpy
        predictions = np.array(predictions)
        
        # Para cada amostra, conta os votos para cada classe
        n_samples = X.shape[0]
        n_models = len(self.models)
        
        # Determina o número de classes a partir das predições
        all_classes = np.unique(np.concatenate([pred.flatten() for pred in predictions]))
        n_classes = len(all_classes)
        
        # Inicializa matriz de votos
        votes = np.zeros((n_samples, n_classes))
        
        # Conta votos ponderados
        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            for j in range(n_samples):
                class_idx = int(pred[j])
                votes[j, class_idx] += weight
        
        # Retorna a classe com mais votos para cada amostra
        return np.argmax(votes, axis=1)
    
    def predict_proba(self, X):
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Obtém probabilidades de cada modelo
        probas = []
        for i, model in enumerate(self.models):
            model_proba = model.predict_proba(X)
            probas.append(model_proba * self.weights[i])
        
        # Soma as probabilidades ponderadas
        ensemble_proba = np.zeros_like(probas[0])
        for proba in probas:
            ensemble_proba += proba
        
        # Normaliza para garantir que a soma seja 1
        row_sums = ensemble_proba.sum(axis=1)
        ensemble_proba = ensemble_proba / row_sums[:, np.newaxis]
        
        return ensemble_proba
```

### Vantagens

- **Robustez**: Menos suscetível a overfitting que modelos individuais.
- **Precisão**: Geralmente oferece melhor desempenho que qualquer modelo individual.
- **Estabilidade**: Reduz a variância e aumenta a estabilidade das predições.
- **Versatilidade**: Pode combinar modelos de diferentes tipos.

### Limitações

- **Complexidade computacional**: Requer treinamento e inferência de múltiplos modelos.
- **Interpretabilidade**: Mais difícil de interpretar que modelos individuais.
- **Overhead de memória**: Requer armazenamento de múltiplos modelos.
- **Diminuição de retornos**: A melhoria de desempenho pode ser marginal após certo ponto.

## Avaliação e Seleção de Modelos

### Métricas de Avaliação

No SISTEMA_EMG, utilizamos as seguintes métricas para avaliar o desempenho dos modelos:

1. **Acurácia**: Proporção de predições corretas.
   $$\text{Acurácia} = \frac{\text{Número de predições corretas}}{\text{Número total de predições}}$$

2. **Precisão**: Proporção de verdadeiros positivos entre os positivos preditos.
   $$\text{Precisão} = \frac{\text{Verdadeiros Positivos}}{\text{Verdadeiros Positivos + Falsos Positivos}}$$

3. **Recall (Sensibilidade)**: Proporção de verdadeiros positivos identificados corretamente.
   $$\text{Recall} = \frac{\text{Verdadeiros Positivos}}{\text{Verdadeiros Positivos + Falsos Negativos}}$$

4. **F1-Score**: Média harmônica entre precisão e recall.
   $$\text{F1} = 2 \times \frac{\text{Precisão} \times \text{Recall}}{\text{Precisão} + \text{Recall}}$$

5. **Matriz de Confusão**: Tabela que mostra as predições corretas e incorretas para cada classe.

### Validação Cruzada

Para avaliar de forma robusta o desempenho dos modelos, utilizamos validação cruzada k-fold:

```python
def cross_validate_model(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Treina o modelo
        model.train(X_train, y_train)
        
        # Realiza predições
        y_pred = model.predict(X_test)
        
        # Calcula métricas
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='weighted'))
        recalls.append(recall_score(y_test, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    
    # Retorna médias e desvios padrão
    return {
        'accuracy': (np.mean(accuracies), np.std(accuracies)),
        'precision': (np.mean(precisions), np.std(precisions)),
        'recall': (np.mean(recalls), np.std(recalls)),
        'f1': (np.mean(f1_scores), np.std(f1_scores))
    }
```

### Seleção de Modelo

O SISTEMA_EMG implementa um processo automatizado de seleção de modelo baseado em validação cruzada:

```python
def select_best_model(X, y, models_to_evaluate):
    results = {}
    
    for name, model in models_to_evaluate.items():
        print(f"Avaliando modelo: {name}")
        cv_results = cross_validate_model(model, X, y)
        results[name] = cv_results
        
        print(f"  Acurácia: {cv_results['accuracy'][0]:.4f} ± {cv_results['accuracy'][1]:.4f}")
        print(f"  F1-Score: {cv_results['f1'][0]:.4f} ± {cv_results['f1'][1]:.4f}")
    
    # Seleciona o melhor modelo baseado na acurácia média
    best_model_name = max(results, key=lambda k: results[k]['accuracy'][0])
    best_model = models_to_evaluate[best_model_name]
    
    print(f"\nMelhor modelo: {best_model_name}")
    print(f"Acurácia: {results[best_model_name]['accuracy'][0]:.4f}")
    
    return best_model, results
```

## Conclusão

O SISTEMA_EMG implementa uma variedade de algoritmos de aprendizado de máquina para classificação de gestos a partir de sinais EMG, cada um com suas próprias vantagens e limitações. A escolha do algoritmo mais adequado depende de fatores como o tamanho do conjunto de dados, a complexidade do problema, os requisitos de tempo real e os recursos computacionais disponíveis.

Para aplicações em tempo real com recursos computacionais limitados, o SVM pode ser a melhor escolha devido à sua eficiência. Para cenários onde a precisão é crítica e há recursos computacionais suficientes, as redes neurais (MLP, CNN, LSTM) ou ensembles podem oferecer melhor desempenho.

O sistema permite que o usuário experimente diferentes algoritmos e selecione o mais adequado para seu caso específico, ou mesmo combine múltiplos modelos em um ensemble para maximizar o desempenho.
