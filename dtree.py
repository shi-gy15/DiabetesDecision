import csvio
import numpy as np
import queue

_information_table = {

}


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
        return ''

    def dict_node(self):
        return None


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

    def dict_node(self):
        d = {
            'f': str(self.feature_index)
        }
        for v, node in self.children.items():
            d[str(v)] = node.dict_node()
        return d


class DecisionTreeLeafNode(DecisionTreeNode):
    def __init__(self, label: int, depth: int):
        super(DecisionTreeLeafNode, self).__init__(depth)
        self.label = label

    def display(self, indent=0):
        return '%d\n' % self.label

    def dict_node(self):
        return str(self.label)


class DecisionTree:
    def __init__(self, feature_num: int):
        self.feature_num = feature_num
        self.root = None
        self.nodeq = queue.Queue()

    def dfs_create_node(self, mtx: np.ndarray, used_features: np.ndarray, depth: int) -> DecisionTreeNode:
        datas = mtx[:, :-1]
        labels = mtx[:, -1]
        used_features = used_features.copy()
        # all in one class
        if (labels == labels[0]).all():
            return DecisionTreeLeafNode(labels[0], depth)
        # all used, no features left
        if used_features.all():
            return DecisionTreeLeafNode(majority(labels), depth)

        bests = best_feature(datas, labels, used_features)
        feature_index = min(bests)
        used_features[list(bests)] = True
        branch = DecisionTreeBranchNode(feature_index, depth)

        values = set(datas[:, feature_index])
        for val in values:
            branch.children[val] = self.dfs_create_node(mtx[mtx[:, feature_index] == val], used_features, depth + 1)
        return branch

    def display(self):
        return self.root.display() if self.root is not None else ''

    def dict_tree(self):
        return self.root.dict_node() if self.root is not None else {}

    @classmethod
    def build_from_dict(cls, d: dict):
        tree = DecisionTree(0)
        tree.nodeq.put((d, None, None), block=False)
        while not tree.nodeq.empty():
            param = tree.nodeq.get(block=False)
            tree.init_node(*param)
        return tree

    def init_node(self, d: dict or str, parent: DecisionTreeBranchNode, parent_val: int):
        depth = 0 if parent is None else parent.depth + 1
        if type(d) == str:
            node = DecisionTreeLeafNode(np.uint8(d), depth)
        else:
            feature_index = np.uint8(d['f'])
            if feature_index + 1 > self.feature_num:
                self.feature_num = feature_index + 1
            node = DecisionTreeBranchNode(feature_index, depth)
            del(d['f'])
            for k, subd in d.items():
                self.nodeq.put((subd, node, np.uint8(k)))
        if parent is None:
            self.root = node
        else:
            parent.children[parent_val] = node

    @classmethod
    def dfs_build(cls, mtx: np.ndarray):
        tree = DecisionTree(mtx.shape[1] - 1)
        used_features = np.zeros(tree.feature_num, dtype=bool)
        tree.root = tree.dfs_create_node(mtx, used_features, 0)
        return tree

    @classmethod
    def bfs_build(cls, mtx: np.ndarray):
        tree = DecisionTree(mtx.shape[1] - 1)
        used_features = np.zeros(tree.feature_num, dtype=bool)
        tree.nodeq.put((mtx, used_features, None, None), block=False)
        while not tree.nodeq.empty():
            param = tree.nodeq.get(block=False)
            tree.bfs_create_node(*param)
        return tree

    def bfs_create_node(self, mtx: np.ndarray, used_features: np.ndarray, parent: DecisionTreeBranchNode, parent_val: int):
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
