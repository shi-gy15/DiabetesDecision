import numpy as np
import decision_tree
import csvio
import csv
import json
import naive_bayesian
import support_vector_machine
import k_nearest_neighbors
import ball_tree
import regression

from decision_tree import DecisionTree
from naive_bayesian import NaiveBayesianNetwork
from k_nearest_neighbors import KNN
from ball_tree import BallTree
from support_vector_machine import SVM
import sys

dias = csvio.Storage()
tests = csvio.Storage()

x1 = np.asarray([0, 1, 4, 1,
                        1, 1, 0, 0,
                        1, 1, 0, 0, 0,
                        0, 0, 3, 4,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        1, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 2, 0, 0,
                        0, 0, 0, 1,
                        1], dtype=np.uint8)

x2 = np.asarray([0, 0, 0, 0])
x3 = np.asarray([1, 2, 1, 1])


def dopre():
    csvio.preprocessing('diabetic_data.csv')


def train_models(train_set: np.ndarray, debug: bool):
    numerical_indexes = [2, 6, 9, 10, 11, 12, 13, 14, 16]

    model_instances = {
        'ball_tree_10': ball_tree.BallTree.build(train_set, numerical_indexes=numerical_indexes, k=10, leaf_size=64, debug=debug),
        # 'knn_10': k_nearest_neighbors.KNN.build(train_set, numerical_indexes=numerical_indexes, k=10, debug=debug),
        # 'knn_20': k_nearest_neighbors.KNN.build(train_set, numerical_indexes=numerical_indexes, k=20, debug=debug),
        # 'knn_40': k_nearest_neighbors.KNN.build(train_set, numerical_indexes=numerical_indexes, k=40, debug=debug),
        # 'd_tree': decision_tree.DecisionTree.build(train_set, debug=debug),
        # 'bayes': naive_bayesian.NaiveBayesianNetwork.build(train_set, debug=debug),
        # 'svm_linear': support_vector_machine.SVM.build(train_set, kernel='linear', sigma=1.0, degree=1, debug=debug),
        # 'svm_poly': support_vector_machine.SVM.build(train_set, kernel='poly', sigma=1.0, degree=2, debug=debug)
    }

    return model_instances


def mainproc(debug):
    stor = csvio.Storage()
    stor.load_csv('pre_num.csv', titled=True)

    # test
    test_round = 1
    train_result = None
    test_result = None

    for i in range(test_round):
        print('Round [%d]' % i)
        train_set, test_set = stor.give(0.75)
        instances = train_models(train_set, debug)
        # init
        if train_result is None:
            train_result = {k: np.zeros((test_round, 4), dtype=np.double) for k in instances.keys()}
        if test_result is None:
            test_result = {k: np.zeros((test_round, 4), dtype=np.double) for k in instances.keys()}

        for model_name, model_instance in instances.items():
            print('Test on train set: %s' % model_name)
            train_result[model_name][i, :] = model_instance.test(train_set)
            # print('Test on test set: %s' % model_name)
            # test_result[model_name][i, :] = model_instance.test(test_set)
            del model_instance

        del instances

    train_mean = {name: np.mean(perf, axis=0) for name, perf in train_result.items()}
    test_mean = {name: np.mean(perf, axis=0) for name, perf in test_result.items()}
    with open('balltreetrain.txt', 'w') as f:
        f.write('------------------------------ performance on train set ------------------------------\n')
        for model_name, result in train_result.items():
            f.write('Model: [%s]\n' % model_name)
            for i in range(test_round):
                f.write('[%d]. Accuracy: [%f], Precision: [%f], Recall: [%f], F1-score: [%f]\n' % (i, *(tuple(result[i, :]))))
            f.write('Mean Accuracy: [%f], Precision: [%f], Recall: [%f], F1-score: [%f]\n' % tuple(train_mean[model_name]))

        f.write('------------------------------ performance on test set ------------------------------\n')
        for model_name, result in test_result.items():
            f.write('Model: [%s]\n' % model_name)
            for i in range(test_round):
                f.write('[%d]. Accuracy: [%f], Precision: [%f], Recall: [%f], F1-score: [%f]\n' % (i, *(tuple(result[i, :]))))
            f.write(
                'Mean Accuracy: [%f], Precision: [%f], Recall: [%f], F1-score: [%f]\n' % tuple(test_mean[model_name]))






def lookloadpre():
    with open('preresult.json', 'r') as f:
        d = json.load(f)
        tree = decision_tree.DecisionTree.build_from_dict(d)
        print(tree.classes)
        # json.dump(tree.dict_tree(), open('prereresult.json', 'w'), indent=2)




def lookdata():
    with open('pre.csv') as f:
        reader = csv.reader(f)
        stor = csvio.Storage()
        stor.load_csv(reader, True)

        datas = stor.data[:, :-1]
        labels = stor.data[:, -1]

        for i in range(20, 42):
            print(i, stor.titles[i], np.bincount(datas[:, i]))

        es = []
        for i in range(stor.column_num - 1):
            es.append((i, stor.titles[i], decision_tree.single_entropy(datas[:, i], labels)))
            # print(i, stor.titles[i], dtree.single_entropy(datas[:, i], labels))

        es = sorted(es, key=lambda x: x[2])
        # with open('entro.txt', 'w') as fo:
        #     [print(ei, file=fo) for ei in es]


def testone():
    global dias
    dias.load_csv('pre.csv', titled=True)
    trainset, testset = dias.give(0.75)
    tree = BallTree.build(trainset, k=10, debug=False)
    print(tree.test(testset[:2000, :]))

def lookreg():
    global dias
    dias.load_csv('pre_num.csv', titled=True)
    data, labels = dias.mtx[:, :-1], dias.mtx[:, -1]
    regression.regression_multinomial(data, labels)


if __name__ == '__main__':
    # dias.load_csv('pre.csv', titled=True)
    # tests.load_csv('test.txt', titled=False)
    # mainproc(debug=False)
    # testone()
    lookreg()