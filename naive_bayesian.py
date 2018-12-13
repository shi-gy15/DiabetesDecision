import models
import numpy as np
import json


class NaiveBayesianNetwork(models.ClassificationModel):
    def __init__(self, feature_num: int, **kwargs):
        super(NaiveBayesianNetwork, self).__init__(feature_num, **kwargs)
        self.p_class = None
        self.p_attrs = None
        self.class_num = None

    @classmethod
    def build(cls, mtx: np.ndarray, **kwargs):
        bayes = NaiveBayesianNetwork(mtx.shape[1] - 1, **models.filter_args(kwargs, ['debug']))
        row_num = mtx.shape[0]
        labels = mtx[:, -1]
        # class probability
        bayes.p_class = np.bincount(labels) / row_num
        bayes.class_num = bayes.p_class.shape[0]
        # attribute probability
        bayes.p_attrs = []
        for c in range(bayes.class_num):
            sub_area = mtx[mtx[:, -1] == c]
            sub_row_num = sub_area.shape[0]
            bayes.p_attrs.append([np.bincount(sub_area[:, j]) / sub_row_num for j in range(bayes.feature_num)])
        return bayes

    @classmethod
    def build_from_dict(cls, d: dict, **kwargs):
        p_class = np.asarray(d['c'])
        p_attrs = [[np.asarray(attr) for attr in p_c] for p_c in d['a']]
        bayes = NaiveBayesianNetwork(len(p_attrs[0]), **models.filter_args(kwargs, ['debug']))
        bayes.p_class = p_class
        bayes.p_attrs = p_attrs
        bayes.class_num = p_class.shape[0]
        return bayes

    def classify(self, data: np.ndarray):
        max_p = 0
        max_class = 0
        for c in range(self.class_num):
            p = self.p_class[c]
            if self.debug:
                print('P(C%d) = %.2f' % (c, p))
            p_c = self.p_attrs[c]
            for j, attr in enumerate(p_c):
                pa = attr[data[j]] if data[j] < attr.shape[0] else 1e-5
                p *= pa
                if self.debug:
                    print('* P(A%d=%d)(== %.2f)' % (j, data[j], pa))
            if self.debug:
                print('P(A|C%d) = %.2f' % (c, p))
            if p > max_p:
                max_p = p
                max_class = c

        return max_class

    def to_json(self, indent=2):
        listp_class = self.p_class.tolist()
        listp = [[attr.tolist() for attr in p_c] for p_c in self.p_attrs]
        d = {
            'c': listp_class,
            'a': listp
        }
        return json.dumps(d, ensure_ascii=False, indent=indent)
