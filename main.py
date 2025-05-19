import typer
from src.model import train
from src.control.prosthesis_controller import ProsthesisController

app = typer.Typer()

@app.command()
def train_model():
    """Treina o modelo de classificação EMG"""
    train.run()

@app.command()
def control(
    port: str = "/dev/ttyACM0",
    baud: int = 115200,
    model: str = "models/svm_model.pkl",
    model_type: str = "svm",
    threshold: float = 0.7,
    simulate: bool = typer.Option(False, help="Executa o controle em modo simulado")
):
    """Executa o controle da prótese (modo real ou simulado)"""
    controller = ProsthesisController(
        port=port,
        baud_rate=baud,
        model_path=model,
        model_type=model_type,
        threshold=threshold,
        simulate=simulate
    )
    controller.run()

if __name__ == "__main__":
    app()
