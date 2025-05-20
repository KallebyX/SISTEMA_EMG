import pytest
import numpy as np
from sklearn.svm import SVC

def test_svm_training():
    X = np.random.rand(10, 5)
    y = [0, 1] * 5
    model = SVC()
    model.fit(X, y)
    assert model.predict([X[0]])[0] in [0, 1]
