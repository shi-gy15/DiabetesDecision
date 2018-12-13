import models
import numpy as np


class KNN(models.ClassificationModel):
    def __init__(self, feature_num: int, **kwargs):
        super(KNN, self).__init__(feature_num, **kwargs)
        self.numerical_indexes = kwargs.get('numerical_indexes', [])
        self.numerical_data = None
        self.nominal_data = None
        self.ranges = None
        self.num_samples = kwargs.get('num_samples', 0)
        self.k = kwargs.get('k', 10)
        self.labels = None

    @classmethod
    def build(cls, mtx: np.ndarray, **kwargs):
        knn = KNN(mtx.shape[1] - 1, **models.filter_args(kwargs, ['debug', 'numerical_indexes', 'k']))

        data = mtx[:, :-1]
        labels = mtx[:, -1]
        knn.num_samples = data.shape[0]
        knn.numerical_data, knn.nominal_data = knn.split_data(data)
        knn.labels = labels
        knn.ranges = np.max(knn.numerical_data, axis=0) + 1
        return knn

    @classmethod
    def build_from_dict(cls, d: dict, **kwargs):
        raise NotImplementedError

    @staticmethod
    def nominal_square_distance(a: np.ndarray, b: np.ndarray):
        return np.sum(a != b)

    def numerical_square_distance(self, a: np.ndarray, b: np.ndarray):
        return np.sum((np.abs(a - b) / self.ranges) ** 2)

    def distance(self, a: np.ndarray, b: np.ndarray or tuple):
        numerical_a, nominal_a = self.split_data(a)
        if type(b) == tuple:
            numerical_b, nominal_b = b
        else:
            numerical_b, nominal_b = self.split_data(b)

        return np.sqrt(self.numerical_square_distance(numerical_a, numerical_b)
                       + self.nominal_square_distance(nominal_a, nominal_b))

    def split_data(self, data: np.ndarray):
        if len(data.shape) == 1:
            numerical = data[self.numerical_indexes]
            nominal = data.copy()
            nominal[self.numerical_indexes] = 0
        elif len(data.shape) == 2:
            numerical = data[:, self.numerical_indexes]
            nominal = data.copy()
            nominal[:, self.numerical_indexes] = 0
        else:
            raise ValueError('more than 2 dims to split')
        return numerical, nominal

    def to_json(self, indent=2):
        raise NotImplementedError

    def classify(self, data: np.ndarray):
        distances = np.array([self.distance(data, (self.numerical_data[k, :], self.nominal_data[k, :]))
                              for k in range(self.num_samples)])
        k_nearest = np.argpartition(distances, self.k - 1)[:self.k]
        if self.debug:
            print('%d nearest neighbors vote: %s' % (self.k, str(k_nearest)))
        return np.uint8(np.argmax(np.bincount(self.labels[k_nearest])))
