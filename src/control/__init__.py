def __init__(self, port='/dev/ttyACM0', baud_rate=115200, model_path=None, 
             model_type='svm', threshold=0.7, window_size=100, safety_timeout=2.0, simulate=False):
    """
    Inicializa o controlador da prótese.
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

    # ⚠️ Sempre configurar filtros, até em simulação
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