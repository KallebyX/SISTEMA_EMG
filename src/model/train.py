from sklearn.svm import SVC
from joblib import dump
from src.model import dataset_builder, metrics
from src.common.logger import get_logger
from joblib import load

logger = get_logger()
def load_model(self):
    self.model = load(self.model_path)
def run():
    logger.info("Iniciando treino do modelo SVM")
    X, y = dataset_builder.build_dataset("data/processed/emg_dataset.csv")
    model = SVC(kernel='rbf')
    model.fit(X, y)
    dump(model, "models/svm_model.pkl")
    logger.info("Modelo salvo com sucesso em models/svm_model.pkl")
