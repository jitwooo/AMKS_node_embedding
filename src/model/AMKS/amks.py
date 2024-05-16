from model.myGraph import MyGraph
from model.utils import compute_laplacian_eigenxx, compute_amks, get_padded_neighbor_vec
import numpy as np
import networkx as nx


class AMKS:
    def __init__(self) -> None:
        self._step = None
        self._sigma = None
        self._time = None
        self._g = None
        self._reindex_dict = None
        self._embedding_vec = None
        self.model_name = 'AMKS'

    def set_step(self, step):
        self._step = step

    def set_sigma(self, sigma):
        self._sigma = sigma

    def set_time(self, time):
        self._time = time

    def set_g(self, g: MyGraph):
        self._g = g

    def compute_emb_vec(self):
        eigenvalues, eigenvectors, reinedex_dict = compute_laplacian_eigenxx(self._g)
        self._reindex_dict = reinedex_dict
        P = None
        self._embedding_vec = compute_amks(eigenvalues, eigenvectors, P, self._sigma, self._step, self._time)

    def get_embedding_vec(self, index_list):
        return np.array([self._embedding_vec[self._reindex_dict[index]] for index in index_list])


class MultiHopAMKS(AMKS):
    def __init__(self) -> None:
        super().__init__()
        self._maxhop = None
        self.model_name = 'MultiHopAMKS'

    def set_maxhop(self, max_hop):
        self._maxhop = max_hop  # whz理解： 参数L ，多少同心圆的意思

    def compute_spec_hop_amks(self, amks: np.ndarray, neighbors_index: np.ndarray):
        gathered_amks = amks[neighbors_index]  # whz理解：取出该范围邻居的嵌入向量表示，并求和
        combined_amks = np.sum(gathered_amks, axis=1)  # whz理解：按列x相加，三维矩阵变成二维矩阵y
        return combined_amks

    def compute_multihop_amks(self):
        amks = self._embedding_vec
        padded_amks = np.vstack(
            [amks, np.zeros((1, amks.shape[1]))])  # np.vstack():在竖直方向上堆叠,相当于多了一行 全是0，即第2812行，用来代表没有邻居
        temp_g = nx.relabel_nodes(self._g, self._reindex_dict)
        g = MyGraph()
        g.add_nodes_from(temp_g)
        g.add_edges_from(temp_g.edges)
        g.set_graph_tag(self._g.graph_tag)
        padded_neighbor_vec = get_padded_neighbor_vec(g)
        results = [None] * (self._maxhop + 1)
        for hop in range(self._maxhop + 1):
            results[hop] = self.compute_spec_hop_amks(padded_amks, padded_neighbor_vec[hop])
            # whz理解：此时的值只代表每一层同心圆的值，不是圈内的，如层数为x时，只代表距离节点i为x的所有邻居，而不是所有<=x 的邻居
            results[hop] = results[hop].reshape(
                (-1, 1))  # 可以在这打个断点看看各个值，特别是维度,比如padded_neighbor_vec，以最大邻居数作为列维度，不足的以2812代替（0-2811）
        # results(6,2812*30)
        result = np.stack(results, axis=1)  # whz理解：矩阵的行向量转成列向量，results(2812*30,6)
        result = np.cumsum(result, axis=1)  # whz理解： 前缀和，即当前累计和,按列相加求和,可以想象成一个同心圆，每一层的值为圈内所有节点的嵌入值之和，即所有<=x 的邻居。
        self._embedding_vec = result.reshape((len(g), -1))  # (2812,180)

    def compute_emb_vec(self):
        super().compute_emb_vec()
        self.compute_multihop_amks()
