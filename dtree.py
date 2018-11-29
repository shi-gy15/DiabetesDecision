import csvio
import numpy as np
import queue

_information_table = {

}

# [feature 0]
# 1: [feature 1]
#    1: 1



def majority(labels: np.ndarray):
    return np.argmax(np.bincount(labels))


def information(bins: np.ndarray):
    # assert there are more than one kind of features
    num_samples = np.sum(bins)
    t = tuple(sorted(bins.tolist()))
    bins = bins[bins != 0]
    if t not in _information_table:
        _information_table[t] = -np.sum(bins * np.log2(bins / num_samples)) / num_samples
    return _information_table[t]


def single_entropy(column: np.ndarray, labels: np.ndarray):
    union = np.stack([column, labels], axis=1)
    bins = np.bincount(column)
    num_samples = labels.shape[0]
    entropy = 0
    for (i, sep_num) in enumerate(bins):
        if sep_num != 0:
            sep_area = union[union[:, 0] == i]
            # entropy += sep_area.shape[0] * binary_information(sep_area[:, 1])
            entropy += sep_area.shape[0] * information(np.bincount(sep_area[:, 1]))

    entropy /= num_samples
    return entropy


def max_entropy(num: int, classes: int):
    cnt = num // classes
    bins = np.full((classes, ), cnt)
    bins[:num - cnt * classes] += 1
    return information(bins)


def best_feature(datas: np.ndarray, labels: np.ndarray, used_features: np.ndarray):
    # gain = I(p, n) - E
    # so choose the minimum new E
    min_entropy = 2
    current_bests = set()
    for i in range(datas.shape[1]):
        if not used_features[i]:
            temp_entropy = single_entropy(datas[:, i], labels)
            if temp_entropy < min_entropy:
                min_entropy = temp_entropy
                current_bests = {i}
            elif temp_entropy == min_entropy:
                current_bests.add(i)
    return current_bests


class DecisionTreeNode:
    def __init__(self, depth: int):
        self.depth = depth

    def display(self, indent=0):
        pass


class DecisionTreeBranchNode(DecisionTreeNode):
    def __init__(self, feature_index: int, depth: int):
        super(DecisionTreeBranchNode, self).__init__(depth)
        self.children = {}
        self.feature_index = feature_index
        self.tag = None

    def display(self, indent=0):
        space = ' ' * indent
        description = '[feature %d]\n' % self.feature_index
        for index, child in self.children.items():
            description += '%s%d: %s' % (space, index, child.display(indent=indent + 3))
        return description


class DecisionTreeLeafNode(DecisionTreeNode):
    def __init__(self, label: int, depth: int):
        super(DecisionTreeLeafNode, self).__init__(depth)
        self.label = label

    def display(self, indent=0):
        return '%d\n' % self.label


class DecisionTree:
    def __init__(self, feature_num: int):
        self.feature_num = feature_num
        # self.used_features = np.zeros(feature_num, dtype=bool)
        self.root = None
        self.nodeq = queue.Queue()


    def create_node(self, mtx: np.ndarray, used_features: np.ndarray) -> DecisionTreeNode:
        datas = mtx[:, :-1]
        labels = mtx[:, -1]
        used_features = used_features.copy()
        # all in one class
        if (labels == labels[0]).all():
            return DecisionTreeLeafNode(labels[0])
        # all used, no features left
        if used_features.all():
            return DecisionTreeLeafNode(majority(labels))

        feature_index = best_feature(datas, labels, used_features)
        used_features[feature_index] = True
        branch = DecisionTreeBranchNode(feature_index)

        values = set(datas[:, feature_index])
        for val in values:
            branch.children[val] = self.create_node(mtx[mtx[:, feature_index] == val], used_features)
        return branch

    def display(self):
        return(self.root.display())

    @classmethod
    def build(cls, mtx: np.ndarray):
        tree = DecisionTree(mtx.shape[1] - 1)
        used_features = np.zeros(tree.feature_num, dtype=bool)
        tree.root = tree.create_node(mtx, used_features)
        return tree

    @classmethod
    def bfs_build(cls, mtx: np.ndarray):
        tree = DecisionTree(mtx.shape[1] - 1)
        used_features = np.zeros(tree.feature_num, dtype=bool)
        tree.nodeq.put((mtx, used_features, None, None), block=False)

        cur_depth = 0

        while not tree.nodeq.empty():
            param = tree.nodeq.get(block=False)
            if param[2] is not None and param[2].depth > cur_depth:
                cur_depth = param[2].depth
                print('into layer %d' % cur_depth)
            tree.bfs_create(*param)

        return tree

    def bfs_create(self, mtx: np.ndarray, used_features: np.ndarray, parent: DecisionTreeBranchNode, parent_val: int):
        datas = mtx[:, :-1]
        labels = mtx[:, -1]

        depth = 0 if parent is None else parent.depth + 1

        # all in one class
        if (labels == labels[0]).all():
            node = DecisionTreeLeafNode(labels[0], depth)
        # all used, no features left
        elif used_features.all():
            node = DecisionTreeLeafNode(majority(labels), depth)
        else:
            # branch
            bests = best_feature(datas, labels, used_features)
            feature_index = min(bests)
            used_features[list(bests)] = True
            node = DecisionTreeBranchNode(feature_index, depth)

            values = set(datas[:, feature_index])
            for val in values:
                self.nodeq.put((mtx[mtx[:, feature_index] == val], used_features.copy(), node, val), block=False)

        if parent is None:
            self.root = node
        else:
            parent.children[parent_val] = node


# def _calc_binary_information(p: int, n: int):
#     s = p + n
#     return -(p * np.log2(p / s) + n * np.log2(n / s)) / s
#
#
# def binary_information(labels: np.ndarray):
#     # assume labels number in [0, 1, 2]
#
#     p, n = np.bincount(labels, minlength=2)
#     p, n = max(p, n), min(p, n)
#     if n == 0:
#         return 0
#     if (p, n) not in _information_table:
#         _information_table[(p, n)] = _calc_binary_information(p, n)
#     return _information_table[(p, n)]


# def gain(column: np.ndarray, labels: np.ndarray):
#     return binary_information(labels) - single_entropy(column, labels)


def create_dtree(mtx: np.ndarray, used_features: np.ndarray):
    datas = mtx[:, :-1]
    labels = mtx[:, -1]
    # all in one class
    if (labels == labels[0]).all():
        return labels[0]
    # no features left
    if datas.shape[1] == 1:
        return majority(labels)

    feature_index = best_feature(datas, labels, used_features)
    used_features[feature_index] = True
    tree = {
        feature_index: {}
    }

    values = set(datas[:, feature_index])
    for val in values:
        tree[feature_index][val] = create_dtree(mtx[mtx[:, feature_index] == val], used_features)
    return tree


def build(mtx: np.ndarray):
    used_f = np.zeros(mtx.shape[1] - 1, dtype=bool)
    return create_dtree(mtx, used_f)

