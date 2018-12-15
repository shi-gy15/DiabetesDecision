import models
import numpy as np
import heapq
import util


class NeighborsHeapNode:
    '''
    Considering distance, heap used in this problem should be big-heap,
    but module `heapq` only supports small-heap.
    Thus, define inverse `__lt__` method to fit the small-heap,
    where the smallest (in original code, largest) element is on top of the heap.
    '''
    def __init__(self, distance, index):
        self.distance = distance
        self.index = index

    def __lt__(self, other):
        return self.distance > other.distance

    def __eq__(self, other):
        return self.distance == other.distance


class NeighborsHeap:
    def __init__(self, k):
        self.heap = [NeighborsHeapNode(np.inf, 0) for _ in range(k)]
        self.max_size = k

    def top(self):
        return heapq.nsmallest(1, self.heap)[0]
        # return self.heap[0]

    def pushpop(self, distance, index):
        heapq.heappushpop(self.heap, NeighborsHeapNode(distance, index))

    def indices(self):
        return np.asarray([node.index for node in self.heap])


class BallTree(models.ClassificationModel):
    def __init__(self, feature_num: int, **kwargs):
        super(BallTree, self).__init__(feature_num, **kwargs)
        self.data = None
        self.num_samples = kwargs.get('num_samples', 0)
        self.k = kwargs.get('k', 10)
        self.leaf_size = kwargs.get('leaf_size', 64)
        self.numerical_indices = kwargs.get('numerical_indices', [])
        self.ranges = None

        self.num_levels = 1 + np.log2((self.num_samples - 1) // self.leaf_size)
        self.num_nodes = int(2 ** self.num_levels) - 1
        self.index_array = np.arange(self.num_samples, dtype=np.int32)
        self.node_radius = np.zeros(self.num_nodes, dtype=np.double)
        self.node_index_start = np.zeros(self.num_nodes, dtype=np.int32)
        self.node_index_end = np.zeros(self.num_nodes, dtype=np.int32)
        self.node_is_leaf = np.zeros(self.num_nodes, dtype=bool)
        self.node_centroids = np.zeros((self.num_nodes, self.feature_num), dtype=np.double)
        self.labels = None
        self.indexed_labels = None

        self.query_heap = None

    @classmethod
    def build(cls, mtx: np.ndarray, **kwargs):
        construct_args = models.filter_args(kwargs, ['debug', 'leaf_size', 'k', 'numerical_indices'])
        construct_args['num_samples'] = mtx.shape[0]
        ball = BallTree(mtx.shape[1] - 1, **construct_args)

        data = mtx[:, :-1]
        labels = mtx[:, -1]
        ball.data = data
        ball.ranges = np.max(ball.data[:, ball.numerical_indices], axis=0) + 1 if ball.numerical_indices != [] else 1

        ball.dfs_build(0, 0, ball.num_samples)
        ball.labels = labels
        return ball

    def dfs_build(self, i_node, index_start, index_end):

        self.init_node(i_node, index_start, index_end)

        if 2 * i_node + 1 >= self.num_nodes:
            self.node_is_leaf[i_node] = True

        elif index_end - index_start < 2:
            self.node_is_leaf[i_node] = True

        else:
            self.node_is_leaf[i_node] = False
            n_mid = int((index_end + index_start) // 2)
            self._partition_indices(index_start, index_end, n_mid)
            if 2 * i_node + 1 < self.num_nodes:
                self.dfs_build(2 * i_node + 1, index_start, n_mid)
            if 2 * i_node + 2 < self.num_nodes:
                self.dfs_build(2 * i_node + 2, n_mid, index_end)

    def init_node(self, i_node, index_start, index_end):
        self.node_centroids[i_node, :] = \
            np.sum(self.data[self.index_array[index_start:index_end], :], axis=0) / (index_end - index_start)

        sq_radius = np.max(self.reduced_distance(self.node_centroids[i_node, :], self.data[self.index_array[index_start:index_end]]))

        self.node_radius[i_node] = np.sqrt(sq_radius)
        self.node_index_start[i_node] = index_start
        self.node_index_end[i_node] = index_end

    def _partition_indices(self, index_start, index_end, split_index):
        val = self.data[self.index_array[index_start:index_end], :]
        split_dim = np.argmax(np.max(val, axis=0) - np.min(val, axis=0))
        data_to_split = self.data[self.index_array[index_start:index_end], split_dim]
        partition_index = np.argpartition(data_to_split, split_index - index_start)
        self.index_array[index_start:index_end] = self.index_array[partition_index + index_start]

    def reduced_distance(self, a: np.ndarray, b: np.ndarray):
        '''
        `a` is set to be 1-d while `b` can be 1-d or n-d, since distance bewteen two n-d arrays is ambiguous.
        :param a: 1-d array
        :param b: 1-d or n-d array
        :return: float value if both `a` and `b` are 1-d, else N*1 array.
        '''
        num_a, nom_a = self.split_data(a)
        num_b, nom_b = self.split_data(b)
        return self.numerical_square_distance(num_a, num_b) + self.nominal_square_distance(nom_a, nom_b)

    def min_reduced_distance(self, i_node: int, X: np.ndarray):
        d = self.reduced_distance(self.node_centroids[i_node, :], X)
        return max(0, np.sqrt(d) - self.node_radius[i_node]) ** 2

    def split_data(self, data: np.ndarray):
        if len(data.shape) == 1:
            numerical = data[self.numerical_indices]
            nominal = data.copy()
            nominal[self.numerical_indices] = 0
        elif len(data.shape) == 2:
            numerical = data[:, self.numerical_indices]
            nominal = data.copy()
            nominal[:, self.numerical_indices] = 0
        else:
            raise ValueError('more than 2 dims to split')
        return numerical, nominal

    def query(self, X):
        X = np.asarray(X, dtype=float)
        self.query_heap = NeighborsHeap(self.k)
        self.dfs_query(0, X, self.min_reduced_distance(0, X))
        neighbors = self.query_heap.indices()

        return neighbors, self.labels[neighbors]

    def dfs_query(self, i_node: int, X: np.ndarray, rdist: float):
        if rdist > self.query_heap.top().distance:
            return
        elif self.node_is_leaf[i_node]:
            start_index = self.node_index_start[i_node]
            end_index = self.node_index_end[i_node]
            distances = self.reduced_distance(X, self.data[self.index_array[start_index:end_index]])
            for i in range(start_index, end_index):
                self.query_heap.pushpop(distances[i - start_index], self.index_array[i])
        else:
            i1 = 2 * i_node + 1
            i2 = i1 + 1
            if i2 >= self.num_nodes:
                self.dfs_query(i1, X, self.min_reduced_distance(i1, X))
            else:
                rdist_left = self.min_reduced_distance(i1, X)
                rdist_right = self.min_reduced_distance(i2, X)

                if rdist_left <= rdist_right:
                    self.dfs_query(i1, X, rdist_left)
                    self.dfs_query(i2, X, rdist_right)
                else:
                    self.dfs_query(i2, X, rdist_right)
                    self.dfs_query(i1, X, rdist_left)

    @staticmethod
    def nominal_square_distance(a: np.ndarray, b: np.ndarray):
        return np.sum(a != b) \
            if len(b.shape) == 1 \
            else np.sum(a != b, axis=1)

    def numerical_square_distance(self, a: np.ndarray, b: np.ndarray):
        return np.sum((np.abs(a - b) / self.ranges) ** 2) \
            if len(b.shape) == 1 \
            else np.sum((np.abs(a - b) / self.ranges) ** 2, axis=1)

    @classmethod
    def build_from_dict(cls, d: dict, **kwargs):
        return None

    def to_json(self, indent=2):
        return '{}'

    def classify(self, data: np.ndarray) -> np.uint8:
        return util.majority(self.query(data)[1])

