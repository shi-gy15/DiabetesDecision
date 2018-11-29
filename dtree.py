import csvio
import numpy as np

_information_table = {

}

# [feature 0]
# 1: [feature 1]
#    1: 1



def majority(labels: np.ndarray):
    return np.argmax(np.bincount(labels))


def single_entropy(column: np.ndarray, labels: np.ndarray):
    union = np.stack([column, labels], axis=1)
    bins = np.bincount(column)
    num_samples = labels.shape[0]
    entropy = 0
    for (i, sep_num) in enumerate(bins):
        if sep_num != 0:
            sep_area = union[union[:, 0] == i]
            entropy += sep_area.shape[0] * binary_information(sep_area[:, 1])

    entropy /= num_samples
    return entropy


def best_feature(datas: np.ndarray, labels: np.ndarray, used_features: np.ndarray):
    # gain = I(p, n) - E
    # so choose the minimum new E
    min_entropy = 1
    current_best = 0
    for i in range(datas.shape[1]):
        if not used_features[i]:
            temp_entropy = single_entropy(datas[:, i], labels)
            if temp_entropy < min_entropy:
                min_entropy = temp_entropy
                current_best = i
    return current_best


class DecisionTreeNode:
    def __init__(self):
        pass

    def display(self, indent=0):
        pass


class DecisionTreeBranchNode(DecisionTreeNode):
    def __init__(self, feature_index: int):
        super(DecisionTreeBranchNode, self).__init__()
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
    def __init__(self, label: int):
        super(DecisionTreeLeafNode, self).__init__()
        self.label = label

    def display(self, indent=0):
        return '%d\n' % self.label


class DecisionTree:
    def __init__(self, feature_num: int):
        self.feature_num = feature_num
        self.used_features = np.zeros(feature_num, dtype=bool)
        self.root = None

    def create_node(self, mtx: np.ndarray) -> DecisionTreeNode:
        datas = mtx[:, :-1]
        labels = mtx[:, -1]
        # all in one class
        if (labels == labels[0]).all():
            return DecisionTreeLeafNode(labels[0])
        # no features left
        if datas.shape[1] == 0:
            return DecisionTreeLeafNode(majority(labels))

        feature_index = best_feature(datas, labels, self.used_features)
        self.used_features[feature_index] = True
        branch = DecisionTreeBranchNode(feature_index)

        values = set(datas[:, feature_index])
        for val in values:
            branch.children[val] = self.create_node(mtx[mtx[:, feature_index] == val])
        return branch

    def display(self):
        print(self.root.display())

    @classmethod
    def build(cls, mtx: np.ndarray):
        tree = DecisionTree(mtx.shape[1] - 1)
        tree.root = tree.create_node(mtx)
        return tree


def information(labels: np.ndarray):
    # assert there are more than one kind of features
    num_samples = labels.shape[0]
    return sum([0 if sample == 0 else -sample * np.log2(sample / num_samples) for sample in np.bincount(labels)]) / num_samples


def _calc_binary_information(p: int, n: int):
    s = p + n
    return -(p * np.log2(p / s) + n * np.log2(n / s)) / s


def binary_information(labels: np.ndarray):
    # assume labels number in [0, 1, 2]

    p, n = np.bincount(labels, minlength=2)
    p, n = max(p, n), min(p, n)
    if n == 0:
        return 0
    if (p, n) not in _information_table:
        _information_table[(p, n)] = _calc_binary_information(p, n)
    return _information_table[(p, n)]


def gain(column: np.ndarray, labels: np.ndarray):
    return binary_information(labels) - single_entropy(column, labels)


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

