import numpy as np
import queue
import models
import json

_information_table = {

}


def majority(labels: np.ndarray):
    return np.argmax(np.bincount(labels))


def information(bins: np.ndarray):
    # assert there are more than one kind of features
    global _information_table
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


def best_feature(data: np.ndarray, labels: np.ndarray, used_features: np.ndarray):
    # gain = I(p, n) - E
    # so choose the minimum new E
    min_entropy = 2
    current_bests = set()
    for i in range(data.shape[1]):
        if not used_features[i]:
            temp_entropy = single_entropy(data[:, i], labels)
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
    def __init__(self, feature_index: int, depth: int, default_label):
        super(DecisionTreeBranchNode, self).__init__(depth)
        self.children = {}
        self.feature_index = feature_index
        self.default_label = default_label
        self.tag = None

    def display(self, indent=0):
        space = ' ' * indent
        description = '[feature %d]\n' % self.feature_index
        for index, child in self.children.items():
            description += '%s%d: %s' % (space, index, child.display(indent=indent + 3))
        description += '%sdefault: %d' % (space, self.default_label)
        return description

    def dict_node(self):
        d = {
            'f': str(self.feature_index),
            'd': str(self.default_label)
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


class DecisionTree(models.ClassificationModel):
    def __init__(self, feature_num: int, **kwargs):
        super(DecisionTree, self).__init__(feature_num, **kwargs)
        self.root = None
        self.nodeq = queue.Queue()

    def display(self):
        return self.root.display() if self.root is not None else ''

    def dict_tree(self):
        return self.root.dict_node() if self.root is not None else {}

    def to_json(self, indent=2):
        return json.dumps(self.dict_tree(), ensure_ascii=False, indent=indent)

    @classmethod
    def build_from_dict(cls, d: dict, **kwargs):
        tree = DecisionTree(0, **models.filter_args(kwargs, ['debug']))

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
            default_label = np.uint8(d['d'])
            if feature_index + 1 > self.feature_num:
                self.feature_num = feature_index + 1
            node = DecisionTreeBranchNode(feature_index, depth, default_label)
            del(d['f'])
            del(d['d'])
            for k, sub_d in d.items():
                self.nodeq.put((sub_d, node, np.uint8(k)))
        if parent is None:
            self.root = node
        else:
            parent.children[parent_val] = node

    @classmethod
    def build(cls, mtx: np.ndarray, **kwargs):
        tree = DecisionTree(mtx.shape[1] - 1, **models.filter_args(kwargs, ['debug']))

        used_features = np.zeros(tree.feature_num, dtype=bool)
        tree.nodeq.put((mtx, used_features, None, None), block=False)
        cur_layer = 0
        while not tree.nodeq.empty():
            param = tree.nodeq.get(block=False)
            if tree.debug:
                if param[2] is not None and param[2].depth > cur_layer:
                    cur_layer = param[2].depth
                    print('layer %d' % cur_layer)
            tree.create_node(*param)
        return tree

    def create_node(self, mtx: np.ndarray, used_features: np.ndarray, parent: DecisionTreeBranchNode, parent_val: int):
        data = mtx[:, :-1]
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
            bests = best_feature(data, labels, used_features)
            feature_index = min(bests)
            used_features[list(bests)] = True
            default_label = np.argmax(np.bincount(labels))
            node = DecisionTreeBranchNode(feature_index, depth, default_label)

            values = set(data[:, feature_index])
            for val in values:
                self.nodeq.put((mtx[mtx[:, feature_index] == val], used_features.copy(), node, val), block=False)

        if parent is None:
            self.root = node
        else:
            parent.children[parent_val] = node

    def classify(self, data: np.ndarray):
        node = self.root
        while type(node) == DecisionTreeBranchNode:
            feature = data[node.feature_index]
            if self.debug:
                print('choose feature %d, data is %d' % (node.feature_index, feature))
            if feature in node.children:
                node = node.children[feature]
            else:
                if self.debug:
                    print('no matching, choose most common: %d' % node.default_label)
                return node.default_label

        if self.debug:
            print('label %d' % node.label)
        return node.label
