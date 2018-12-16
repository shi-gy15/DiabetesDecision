from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


def regression_multinomial(datas, labels):
    reg = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial')
    reg.fit(datas, labels)
    print(reg.coef_)
    print(reg.intercept_)
