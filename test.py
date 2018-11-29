import numpy as np
import dtree
import csvio
import csv
import json
import sys


def dopre():
    csvio.preprocessing('diabetic_data.csv')


def looktest():
    with open('test.txt', 'r') as f:
        reader = csv.reader(f)
        stor = csvio.Storage()
        stor.load_csv(reader, titled=False)
        tree = dtree.DecisionTree.bfs_build(stor.data)
        json.dump(tree.dict_tree(), open('testresult.json', 'w'), indent=2)

        # print(tree.display())


def lookloadtest():
    with open('testresult.json', 'r') as f:
        d = json.load(f)
        tree = dtree.DecisionTree.build_from_dict(d)
        print(json.dumps(tree.dict_tree(), indent=2))


def lookloadpre():
    with open('preresult.json', 'r') as f:
        d = json.load(f)
        tree = dtree.DecisionTree.build_from_dict(d)
        json.dump(tree.dict_tree(), open('prereresult.json', 'w'), indent=2)



def lookpre():
    with open('pre.csv', 'r') as f:
        reader = csv.reader(f)
        stor = csvio.Storage()
        stor.load_csv(reader, titled=True)
        tree = dtree.DecisionTree.bfs_build(stor.data)
        # print(tree.display(), file=open('origin_preresult.txt', 'w'))

        json.dump(tree.dict_tree(), open('preresult.json', 'w'), indent=2)

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
            es.append((i, stor.titles[i], dtree.single_entropy(datas[:, i], labels)))
            # print(i, stor.titles[i], dtree.single_entropy(datas[:, i], labels))

        es = sorted(es, key=lambda x: x[2])
        with open('entro.txt', 'w') as fo:
            [print(ei, file=fo) for ei in es]


if __name__ == '__main__':
    # sys.setrecursionlimit(1500)
    lookloadpre()