import numpy as np
import dtree
import csvio
import csv
import json
import sys

if __name__ == '__main__':
    sys.setrecursionlimit(1500)
    # with open('pre.csv') as f:
    #     reader = csv.reader(f)
    #     stor = csvio.Storage()
    #     stor.load_csv(reader, True)
    #     json.dump(stor.mappings, open('mappings.json', 'w'), indent=4, ensure_ascii=False)

    # csvio.preprocessing('diabetic_data.csv')
    with open('pre.csv') as f:
        reader = csv.reader(f)
        stor = csvio.Storage()
        stor.load_csv(reader, True)
        print(stor.last_index)
        print(stor.row_num)

        datas = stor.data[:, :-1]
        labels = stor.data[:, -1]

        for i in range(36, 42):
            print(np.bincount(datas[:, i]))
        # for i in range(stor.column_num - 1):
        #     print(i, dtree.single_entropy(datas[:, i], labels))
        # tree = dtree.DecisionTree.build(stor.data)
        #
        # with open('tree.txt', 'w', encoding='utf-8') as f:
        #     f.write(tree.display())
