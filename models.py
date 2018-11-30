import numpy as np


class ClassificationModel:
    def __init__(self, feature_num: int, debug: bool = False):
        self.feature_num = feature_num
        self.debug = debug

    @classmethod
    def build(cls, mtx:np.ndarray, debug: bool = False):
        raise NotImplementedError

    @classmethod
    def build_from_dict(cls, d: dict, debug: bool = False):
        raise NotImplementedError

    def to_json(self, indent=2):
        raise NotImplementedError

    def classify(self, data: np.ndarray):
        raise NotImplementedError