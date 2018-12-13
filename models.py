import numpy as np
from sklearn import metrics


def filter_args(args: dict, keys: list):
    return {k: v for k, v in args.items() if k in keys}


class ClassificationModel:
    '''
    `N`: sample num
    `n`: feature num
    `m`: label(class) num
    @:param mtx[N*(n+1)]: input data and labels (last column)
    @:param debug: if `True`, debug information is printed

    `build` and `classify` methods must be implemented
    '''
    def __init__(self, feature_num: int, **kwargs):
        self.feature_num = feature_num
        self.debug = kwargs['debug'] if 'debug' in kwargs else False
        print('Initializing %s, args: %s' % (self.__class__.__name__, str(kwargs)))

    @classmethod
    def build(cls, mtx: np.ndarray, **kwargs):
        raise NotImplementedError

    @classmethod
    def build_from_dict(cls, d: dict, **kwargs):
        return None

    def to_json(self, indent=2):
        return '{}'

    def classify(self, data: np.ndarray) -> np.uint8:
        raise NotImplementedError

    def classify_all(self, data: np.ndarray):
        return np.asarray([self.classify(row) for row in data], dtype=np.uint8)

    def test(self, mtx: np.ndarray):
        num_samples = mtx.shape[0]
        data = mtx[:, :-1]
        truth = mtx[:, -1]
        result = self.classify_all(data)
        class_bin = set(truth) | set(result)

        accuracy = metrics.accuracy_score(truth, result)
        precision = metrics.precision_score(truth, result, average='macro')
        recall = metrics.recall_score(truth, result, average='macro')
        F1 = metrics.f1_score(truth, result, average='macro')

        return [accuracy, precision, recall, F1]
        # accuracy = {}
        # precision = {}
        # recall = {}
        # F1 = {}
        #
        # for class_id in class_bin:
        #     result_ovr = result == class_id
        #     truth_ovr = truth == class_id
        #     sum_ab = np.count_nonzero(truth_ovr)
        #     sum_ac = np.count_nonzero(result_ovr)
        #     sum_ad = np.count_nonzero(truth_ovr == result_ovr)
        #     a = (sum_ab + sum_ac + sum_ad - num_samples) / 2
        #     accuracy[class_id] = a / num_samples
        #     precision[class_id] = a / sum_ac
        #     recall[class_id] = a / sum_ab
        #     F1[class_id] = 2 * precision[class_id] * recall[class_id] / (precision[class_id] + recall[class_id])

