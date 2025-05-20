/*
 * Biomove - Código de Controle do Motor para Prótese
 * 
 * Este código implementa o controle do motor da prótese
 * baseado nos comandos recebidos via comunicação serial.
 * 
 * Desenvolvido por: Biomove
 * Data: Maio 2025
 */

// Definição de pinos
const int motorPinA = 5;      // Pino de controle do motor (direção A)
const int motorPinB = 6;      // Pino de controle do motor (direção B)
const int enablePin = 7;      // Pino de habilitação do driver do motor
const int statusLedPin = 13;  // LED para indicação de status
const int limitSwitchOpen = 2;  // Chave de fim de curso para mão aberta
const int limitSwitchClose = 3; // Chave de fim de curso para mão fechada

// Parâmetros de configuração
const long baudRate = 115200;   // Taxa de comunicação serial
const int motorSpeed = 200;     // Velocidade do motor (0-255)
const int motorTimeout = 5000;  // Timeout de segurança para o motor (ms)

// Variáveis de controle
String command = "";           // Comando recebido
bool isComplete = false;       // Flag para comando completo
unsigned long motorStartTime = 0; // Tempo de início da ativação do motor
bool motorActive = false;      // Estado do motor
int currentDirection = 0;      // Direção atual (1=abrir, -1=fechar, 0=parado)

void setup() {
  // Inicializa comunicação serial
  Serial.begin(baudRate);
  
  // Configura pinos
  pinMode(motorPinA, OUTPUT);
  pinMode(motorPinB, OUTPUT);
  pinMode(enablePin, OUTPUT);
  pinMode(statusLedPin, OUTPUT);
  pinMode(limitSwitchOpen, INPUT_PULLUP);
  pinMode(limitSwitchClose, INPUT_PULLUP);
  
  // Inicializa motor desligado
  stopMotor();
  
  // Mensagem de inicialização
  Serial.println("BIOMOVE_MOTOR_INIT");
  
  // Pisca LED para indicar inicialização
  for (int i = 0; i < 3; i++) {
    digitalWrite(statusLedPin, HIGH);
    delay(100);
    digitalWrite(statusLedPin, LOW);
    delay(100);
  }
}

void loop() {
  // Verifica se há dados na serial
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    // Verifica se é fim de comando
    if (c == '\n') {
      isComplete = true;
    } else {
      // Adiciona caractere ao comando
      command += c;
    }
  }
  
  // Processa comando completo
  if (isComplete) {
    processCommand();
    
    // Limpa comando e flag
    command = "";
    isComplete = false;
  }
  
  // Verifica timeout de segurança
  if (motorActive && (millis() - motorStartTime > motorTimeout)) {
    stopMotor();
    Serial.println("MOTOR_TIMEOUT");
  }
  
  // Verifica chaves de fim de curso
  checkLimitSwitches();
}

// Processa comando recebido
void processCommand() {
  command.trim();  // Remove espaços em branco
  
  if (command == "MOTOR_OPEN") {
    openHand();
    Serial.println("OPENING_HAND");
  }
  else if (command == "MOTOR_CLOSE") {
    closeHand();
    Serial.println("CLOSING_HAND");
  }
  else if (command == "MOTOR_STOP") {
    stopMotor();
    Serial.println("MOTOR_STOPPED");
  }
  else if (command == "STATUS") {
    sendStatus();
  }
  else if (command == "CALIBRATE") {
    calibrateMotor();
  }
  else {
    Serial.println("UNKNOWN_COMMAND");
  }
}

// Abre a mão (ativa o motor na direção de abertura)
void openHand() {
  // Verifica se já está no limite de abertura
  if (digitalRead(limitSwitchOpen) == LOW) {
    Serial.println("ALREADY_OPEN");
    return;
  }
  
  // Configura direção do motor para abrir
  digitalWrite(motorPinA, HIGH);
  digitalWrite(motorPinB, LOW);
  
  // Ativa o motor
  digitalWrite(enablePin, HIGH);
  digitalWrite(statusLedPin, HIGH);
  
  // Atualiza estado
  motorActive = true;
  motorStartTime = millis();
  currentDirection = 1;
}

// Fecha a mão (ativa o motor na direção de fechamento)
void closeHand() {
  // Verifica se já está no limite de fechamento
  if (digitalRead(limitSwitchClose) == LOW) {
    Serial.println("ALREADY_CLOSED");
    return;
  }
  
  // Configura direção do motor para fechar
  digitalWrite(motorPinA, LOW);
  digitalWrite(motorPinB, HIGH);
  
  // Ativa o motor
  digitalWrite(enablePin, HIGH);
  digitalWrite(statusLedPin, HIGH);
  
  // Atualiza estado
  motorActive = true;
  motorStartTime = millis();
  currentDirection = -1;
}

// Para o motor
void stopMotor() {
  // Desliga o motor
  digitalWrite(motorPinA, LOW);
  digitalWrite(motorPinB, LOW);
  digitalWrite(enablePin, LOW);
  digitalWrite(statusLedPin, LOW);
  
  // Atualiza estado
  motorActive = false;
  currentDirection = 0;
}

// Verifica chaves de fim de curso
void checkLimitSwitches() {
  // Se o motor estiver ativo, verifica os limites
  if (motorActive) {
    // Verifica limite de abertura
    if (currentDirection == 1 && digitalRead(limitSwitchOpen) == LOW) {
      stopMotor();
      Serial.println("LIMIT_OPEN_REACHED");
    }
    
    // Verifica limite de fechamento
    if (currentDirection == -1 && digitalRead(limitSwitchClose) == LOW) {
      stopMotor();
      Serial.println("LIMIT_CLOSE_REACHED");
    }
  }
}

// Envia status atual
void sendStatus() {
  Serial.print("STATUS,");
  Serial.print(motorActive ? "ACTIVE," : "INACTIVE,");
  
  if (currentDirection == 1) {
    Serial.print("OPENING,");
  } else if (currentDirection == -1) {
    Serial.print("CLOSING,");
  } else {
    Serial.print("STOPPED,");
  }
  
  Serial.print(digitalRead(limitSwitchOpen) == LOW ? "OPEN_LIMIT," : "NOT_OPEN,");
  Serial.println(digitalRead(limitSwitchClose) == LOW ? "CLOSE_LIMIT" : "NOT_CLOSED");
}

// Calibra o motor (ciclo completo de abertura e fechamento)
void calibrateMotor() {
  Serial.println("CALIBRATION_START");
  
  // Primeiro abre completamente
  openHand();
  
  // Aguarda até atingir o limite ou timeout
  unsigned long startTime = millis();
  while (digitalRead(limitSwitchOpen) != LOW && (millis() - startTime < motorTimeout)) {
    delay(10);
  }
  
  // Para o motor
  stopMotor();
  delay(1000);  // Pausa de 1 segundo
  
  // Agora fecha completamente
  closeHand();
  
  // Aguarda até atingir o limite ou timeout
  startTime = millis();
  while (digitalRead(limitSwitchClose) != LOW && (millis() - startTime < motorTimeout)) {
    delay(10);
  }
  
  // Para o motor
  stopMotor();
  
  Serial.println("CALIBRATION_COMPLETE");
}
