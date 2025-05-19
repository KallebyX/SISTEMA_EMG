/*
 * Biomove - Sistema de Captação EMG com MyoWare 2.0
 * 
 * Este código realiza a leitura do sensor MyoWare 2.0 e envia os dados
 * via comunicação serial para processamento em Python.
 * 
 * Desenvolvido por: Biomove
 * Data: Maio 2025
 */

// Definição de pinos
const int myowarePin = A0;      // Pino analógico para o sensor MyoWare 2.0
const int ledPin = 13;          // LED para indicação visual
const int thresholdLed = 9;     // LED para indicação de threshold
const int calibrationButton = 2; // Botão para calibração

// Parâmetros de configuração
const int sampleRate = 500;     // Taxa de amostragem em Hz
const long baudRate = 115200;   // Taxa de comunicação serial
const int bufferSize = 10;      // Tamanho do buffer para média móvel

// Variáveis de controle
int emgValue = 0;               // Valor atual do EMG
int emgBuffer[10];              // Buffer para média móvel
int bufferIndex = 0;            // Índice atual do buffer
int emgBaseline = 0;            // Valor de linha de base (calibração)
int emgThreshold = 200;         // Limiar para detecção de ativação muscular
bool isCalibrating = false;     // Flag para modo de calibração
unsigned long lastSampleTime = 0; // Controle de tempo para amostragem
unsigned long calibrationStartTime = 0; // Tempo de início da calibração
const int calibrationDuration = 3000; // Duração da calibração em ms

void setup() {
  // Inicializa comunicação serial
  Serial.begin(baudRate);
  
  // Configura pinos
  pinMode(ledPin, OUTPUT);
  pinMode(thresholdLed, OUTPUT);
  pinMode(calibrationButton, INPUT_PULLUP);
  
  // Inicializa buffer
  for (int i = 0; i < bufferSize; i++) {
    emgBuffer[i] = 0;
  }
  
  // Aguarda estabilização do sensor
  delay(1000);
  
  // Mensagem de inicialização
  Serial.println("BIOMOVE_EMG_INIT");
  
  // Pisca LED para indicar inicialização
  for (int i = 0; i < 3; i++) {
    digitalWrite(ledPin, HIGH);
    delay(100);
    digitalWrite(ledPin, LOW);
    delay(100);
  }
}

void loop() {
  // Verifica se é hora de coletar uma amostra
  unsigned long currentTime = millis();
  if (currentTime - lastSampleTime >= (1000 / sampleRate)) {
    lastSampleTime = currentTime;
    
    // Lê o valor do sensor
    emgValue = analogRead(myowarePin);
    
    // Adiciona ao buffer de média móvel
    emgBuffer[bufferIndex] = emgValue;
    bufferIndex = (bufferIndex + 1) % bufferSize;
    
    // Calcula média móvel
    int emgSmoothed = calculateMovingAverage();
    
    // Verifica se está em modo de calibração
    if (isCalibrating) {
      handleCalibration();
    }
    
    // Verifica se o botão de calibração foi pressionado
    if (digitalRead(calibrationButton) == LOW && !isCalibrating) {
      startCalibration();
    }
    
    // Verifica se o sinal ultrapassou o limiar
    if (emgSmoothed > (emgBaseline + emgThreshold)) {
      digitalWrite(thresholdLed, HIGH);
    } else {
      digitalWrite(thresholdLed, LOW);
    }
    
    // Envia dados pela serial
    Serial.print(emgValue);
    Serial.print(",");
    Serial.print(emgSmoothed);
    Serial.print(",");
    Serial.print(emgBaseline);
    Serial.print(",");
    Serial.println(emgBaseline + emgThreshold);
    
    // Pisca LED para indicar leitura
    digitalWrite(ledPin, !digitalRead(ledPin));
  }
}

// Calcula a média móvel do buffer
int calculateMovingAverage() {
  long sum = 0;
  for (int i = 0; i < bufferSize; i++) {
    sum += emgBuffer[i];
  }
  return sum / bufferSize;
}

// Inicia o processo de calibração
void startCalibration() {
  isCalibrating = true;
  calibrationStartTime = millis();
  
  // Limpa o buffer para nova calibração
  for (int i = 0; i < bufferSize; i++) {
    emgBuffer[i] = 0;
  }
  
  // Notifica início da calibração
  Serial.println("CALIBRATION_START");
  
  // Pisca LED rapidamente para indicar calibração
  for (int i = 0; i < 5; i++) {
    digitalWrite(ledPin, HIGH);
    delay(50);
    digitalWrite(ledPin, LOW);
    delay(50);
  }
}

// Gerencia o processo de calibração
void handleCalibration() {
  // Verifica se o tempo de calibração terminou
  if (millis() - calibrationStartTime >= calibrationDuration) {
    // Calcula a linha de base como a média atual
    emgBaseline = calculateMovingAverage();
    
    // Termina a calibração
    isCalibrating = false;
    
    // Notifica fim da calibração
    Serial.print("CALIBRATION_END,");
    Serial.println(emgBaseline);
    
    // Pisca LED para indicar fim da calibração
    for (int i = 0; i < 3; i++) {
      digitalWrite(ledPin, HIGH);
      delay(100);
      digitalWrite(ledPin, LOW);
      delay(100);
    }
  } else {
    // Durante a calibração, mantém o LED aceso
    digitalWrite(ledPin, HIGH);
  }
}
