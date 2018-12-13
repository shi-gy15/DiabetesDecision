import models
import numpy as np
import mcsvm_min
import cvxopt


class SVM(models.ClassificationModel):
    def __init__(self, feature_num: int, **kwargs):
        super(SVM, self).__init__(feature_num, **kwargs)
        self.sample_num = kwargs.get('sample_num', 0)
        self.classifier = None
        self.kernel = kwargs.get('kernel', 'linear')
        self.sigma = kwargs.get('sigma', 1.0)
        self.degree = kwargs.get('degree', 1)

    @classmethod
    def build(cls, mtx: np.ndarray, **kwargs):
        data = mtx[:, :-1]
        labels = mtx[:, -1]
        construct_args = models.filter_args(kwargs, ['debug', 'kernel', 'sigma', 'degree'])
        construct_args['sample_num'] = data.shape[0]
        svm = SVM(data.shape[0], **construct_args)

        X = cvxopt.matrix(data.astype(np.double))
        y = cvxopt.matrix(labels.tolist())
        svm.classifier = mcsvm_min.mcsvm(X, y, 1, kernel=svm.kernel, sigma=svm.sigma, degree=svm.degree)
        return svm

    @classmethod
    def build_from_dict(cls, d: dict, debug: bool = False):
        raise NotImplementedError

    def classify(self, data: np.ndarray):
        input_data = cvxopt.matrix(data.reshape((1, -1)).astype(np.double))
        classes = self.classifier(input_data)
        return np.uint8(classes[0])

    def classify_all(self, data: np.ndarray):
        return np.uint8(self.classifier(cvxopt.matrix(data.astype(np.double))))
