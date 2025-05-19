import numpy as np
import pandas as pd

def build_dataset(path):
    # Simulação: substitua com sua lógica
    data = pd.read_csv(path)
    X = data.drop("label", axis=1)
    y = data["label"]
    return X, y
