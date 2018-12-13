from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


def regression_multinomial(datas, labels):
    reg = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial')
    classifier = reg.fit(datas, labels)


    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)

def regression_binary(datas, labels):
    reg = LogisticRegression(penalty='l2', solver='liblinear', multi_class='ovr')
    classifier = reg.fit(datas, labels)
