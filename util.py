import numpy as np


def majority(labels: np.ndarray):
    return np.argmax(np.bincount(labels))
