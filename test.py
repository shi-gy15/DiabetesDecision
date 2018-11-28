import numpy as np
import dtree
import csvio
import csv

if __name__ == '__main__':
    # a = np.asarray([1, 1, 1, 2, 2, 3, 3, 3])
    # labels = np.asarray([1, 1, 2, 2, 2, 1, 2, 2])
    # print(dtree.single_entropy(a, labels))
    # all_data = csvio.test_get_data('test.txt')
    # tree = dtree.DecisionTree.build(all_data)
    # tree.display()
    # print(tree.root.d)
    # print(dtree.build(all_data))
    # print(csvio.indexes)
    with open('diabetic_data.csv') as f:
        reader = csv.reader(f)
        stor = csvio.Storage()
        stor.load_csv(reader, True)
        print(stor.last_index)
        # print(np.bincount(stor.data[:, 12]))
        print(stor.mappings[11])
        print(1)