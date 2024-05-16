from collections import defaultdict

import networkx as nx

from model.graphWave import metrics
from model.utils import get_padded_neighbor_vec, compute_laplacian_eigenxx
import model.graphWave.utils as utils
import numpy as np
import pygsp
from tqdm import tqdm
from scipy.stats import wasserstein_distance
import multiprocessing
from model.myGraph import MyGraph


class HSD(object):
    # def __init__(self, graph: nx.Graph, graphName: str, scale: float, hop: int):
    def __init__(self, graph: MyGraph, graphName: str, scale: float, hop: int):
        """
        :param graph:
        :param graphName:
        :param scale: time
        :param hop:  k-hop neighbors
        """
        self.graph = graph
        self.graphName = graphName
        self.scale = scale
        self.hop = hop
        self.A = nx.adjacency_matrix(graph).todense()
        self.L = nx.laplacian_matrix(graph).todense()

        self.nodes = list(nx.nodes(graph))
        self.n_node = len(self.nodes)
        self.idx2node, self.reinedex_dict = utils.build_node_idx_map(graph)
        self.embedding_vec = None  # 按照idx顺序索引的节点嵌入向量

        # self.hierarchy = []
        # padding_vec = get_padded_neighbor_vec(self.graph)
        # for i in range(min(self.hop, len(padding_vec))):
        #     self.hierarchy.append(padding_vec[i])
        self.hierarchy = []
        temp_g = nx.relabel_nodes(self.graph, self.reinedex_dict)
        g = MyGraph()
        g.add_nodes_from(temp_g)
        g.add_edges_from(temp_g.edges)
        g.set_graph_tag(self.graph.graph_tag)
        padding_vec = get_padded_neighbor_vec(g)
        for i in range(min(self.hop, len(padding_vec))):
            self.hierarchy.append(padding_vec[i])

        self.eigenvalues = None
        self.eigenvectors = None
        self.reinedex_dict = None
        self.eigenvalues, self.eigenvectors, self.reinedex_dict = compute_laplacian_eigenxx(self.graph)
        self.idx2node.clear()
        for k, v in self.reinedex_dict.items():
            self.idx2node[v] = k

    def calculate_wavelets(self, scale, approx=False) -> np.ndarray:
        """
        获得每个节点的热扩散，返回矩阵大小为(n,n)
        """
        if approx:
            G = pygsp.graphs.Graph(self.A)
            G.estimate_lmax()
            heat_filter = pygsp.filters.Heat(G, tau=[scale * G._lmax])
            chebyshev = pygsp.filters.approximations.compute_cheby_coeff(heat_filter, m=50)
            wavelets = np.empty(shape=(self.n_node, self.n_node))
            for idx in range(self.n_node):
                impulse = np.zeros(self.n_node, dtype=np.float)
                impulse[idx] = 1.0
                coeff = pygsp.filters.approximations.cheby_op(G, chebyshev, impulse)
                wavelets[idx, :] = coeff

        else:

            wavelets = np.dot(
                np.dot(self.eigenvectors, np.diag(np.exp(-1 * scale * self.eigenvalues))),
                np.transpose(self.eigenvectors),
            )

        # threshold = np.vectorize(lambda x: x if x > 1e-4 * 1.0 / self.n_node else 0)
        # wavelets = threshold(wavelets)
        wavelets = wavelets * (wavelets > (1e-4 * 1.0 / self.n_node))
        return wavelets

    def get_hierarchical_coeffcients(self, wavelets) -> dict:
        coeffs_dict = dict()
        for idx, node in enumerate(self.nodes):
            assert idx == node, "the idx not equal the node"
            neighbor_layers = self.hierarchy[node]
            coeffs = []
            for neighbor_set in neighbor_layers:
                tmp = []
                for neighbor in neighbor_set:
                    idx2 = self.reinedex_dict[neighbor]
                    tmp.append(wavelets[idx, idx2])
                coeffs.append(tmp)
            coeffs_dict[node] = coeffs
        return coeffs_dict

    def calculate_structural_distance(self, scale, approx=False, metric="euclidean"):
        """
        工作流程：
            1. 读入图的不同阶的邻居节点(最短距离)
            2. 读取不同邻居节点在同一尺度下接收到的原点扩散出来的热量，保存在不同的向量中
            3. 通过不同阶邻居接收到的热量对比通过wasserstein_distance(看作一个概率)计算出来一个相似度，并加和，
            就得到了两个节点间的结构相似度
        """
        wavelets = self.calculate_wavelets(scale, approx)
        coeffs_dict = self.get_hierarchical_coeffcients(wavelets)

        dist_mat = np.zeros((self.n_node, self.n_node), dtype=float)
        for idx1, node1 in tqdm(enumerate(self.nodes)):
            for idx2 in range(idx1 + 1, self.n_node):
                node2 = self.nodes[idx2]
                coeffs_layers1, coeffs_layers2 = coeffs_dict[node1], coeffs_dict[node2]
                distance = 0.0
                for hop in range(self.hop + 1):
                    # coeffs doesn't have to share same length
                    coeffs1, coeffs2 = coeffs_layers1[hop], coeffs_layers2[hop]
                    distance += wasserstein_distance(coeffs1, coeffs2)
                dist_mat[idx1, idx2] = dist_mat[idx2, idx1] = distance

        return dist_mat

    def parallel_calculate_HSD(self, n_workers=3, metric="euclidean"):
        """
        和calculate_structural_distance相同，只是改为了并行处理的形式
        """
        distMat = np.zeros((self.n_node, self.n_node), dtype=float)
        pool = multiprocessing.Pool(n_workers)
        states = {}
        for idx in range(self.n_node):
            res = pool.apply_async(HSD._calculate_worker, args=(self, idx, metric))
            states[idx] = res
        pool.close()
        pool.join()

        results = []
        for idx in range(self.n_node):
            results.append(states[idx].get())

        for idx1, dists in enumerate(results):
            for idx2 in range(idx1 + 1, self.n_node):
                distMat[idx1, idx2] = distMat[idx2, idx1] = dists[idx2]

        self.distMat = distMat
        return self.distMat

    def _calculate_worker(self, startIndex: int, metric: str) -> np.ndarray:
        dists = np.zeros(self.n_node)
        node = self.nodes[startIndex]
        layers = self.hierarchy[node]
        for idx in range(startIndex + 1, self.n_node):
            other = self.nodes[idx]
            _layers = self.hierarchy[other]
            d = 0.0
            for hop in range(self.hop):
                r1, r2 = layers[hop], _layers[hop]
                p, q = [], []
                for neighbor in r1:
                    p.append(self.wavelets[startIndex, self.reinedex_dict[neighbor]])

                for neighbor in r2:
                    q.append(self.wavelets[startIndex, self.reinedex_dict[neighbor]])

                d += metrics.calculate_distance(p, q, metric)

            dists[idx] = d

        return dists


class MultiHSD(HSD):
    def __init__(self, graph: MyGraph, graphName: str, hop: int, n_scales: int):
        super(MultiHSD, self).__init__(graph, graphName, 0, hop)
        self.n_scales = n_scales
        self.scales = None
        self.embeddings = None

        self.init()

    def init(self, is_stable=True, max_range=50.0):
        # 初始化，估算特征值，确定尺度范围，构建层级表示hierarchy
        # 像这样估算尺度范围，有什么道理嘛？
        if is_stable:
            self.scales = np.exp(
                np.linspace(np.log(0.01), np.log(max_range), self.n_scales)
            )
        ## self.hierarchy = hierarchy.read_hierarchical_representation(
        ##     self.graphName, self.hop
        ## )
        else:
            G = pygsp.graphs.Graph(self.A)
            G.estimate_lmax()
            self.scales = np.exp(
                np.linspace(np.log(0.01), np.log(G.lmax), self.n_scales)
            )
            ##self.hierarchy = hierarchy.read_hierarchical_representation(
            ##    self.graphName, self.hop
            ##)

    def embed(self) -> dict:
        """
        将节点的结构表示，通过热核嵌入为向量，不同尺度下，不同层
        """
        print(len(self.nodes))
        embeddings = []
        for scale in tqdm(self.scales):
            wavelets = self.calculate_wavelets(scale, approx=False)
            wavelets = np.vstack([wavelets, np.zeros((1, wavelets.shape[1]), dtype=wavelets.dtype)])
            temp_embeddins = []
            for cur_level in self.hierarchy:
                temp_vec = wavelets[cur_level, np.arange(wavelets.shape[1]).reshape(-1, 1)]
                temp_embeddins.append(
                    np.hstack(
                        [
                            np.sum(temp_vec, axis=1, keepdims=True),
                            np.mean(temp_vec, axis=1, keepdims=True),
                            np.sum(cur_level < wavelets.shape[0], axis=1, keepdims=True)
                        ]
                    )
                )

            embeddings.append(
                np.hstack(temp_embeddins)
            )
        self.embedding_vec = np.hstack(embeddings)
        return self.embedding_vec

    def get_embedding_vec(self, nodes):
        return np.array([self.embedding_vec[self.reinedex_dict[node]] for node in nodes])

    def get_statistics(self, wavelets: np.ndarray, node: int) -> list:
        """
        计算节点i不同阶邻居接收到热量的和与均值
        """
        descriptor = []
        neighborhoods = self.hierarchy[node]
        node_idx = self.reinedex_dict[node]
        for hop, level in enumerate(neighborhoods):
            coeffs = []
            for neighbor in level:
                if neighbor == "":
                    continue
                coeffs.append(wavelets[node_idx, self.reinedex_dict[neighbor]])

            if len(coeffs) > 0:
                statistics = [np.sum(coeffs), np.mean(coeffs), len(coeffs)]
            else:
                statistics = [0.0, 0.0, 0.0]
            descriptor.extend(statistics)
        return descriptor

    def get_layer_sum(self, wavelets: np.ndarray, node: str) -> list:
        """
        计算节点i不同阶邻居节点接收到热量的总和
        """
        layers_sum = [0] * (self.hop + 1)
        neighborhoods = self.hierarchy[node]
        node_idx = self.reinedex_dict[node]
        for hop, level in enumerate(neighborhoods):
            for neighbor in level:
                if neighbor == "":
                    continue
                layers_sum[hop] += wavelets[node_idx, self.reinedex_dict[neighbor]]
        return layers_sum

    def parallel_embed(self, n_workers) -> dict:
        """
        embed的并行改进
        """
        pool = multiprocessing.Pool(n_workers)
        states = {}
        for idx, scale in enumerate(self.scales):
            res = pool.apply_async(self.calculate_wavelets, args=(scale, False))
            states[idx] = res
        pool.close()
        pool.join()

        results = []
        for idx in range(self.n_scales):
            results.append(states[idx].get())

        embeddings = defaultdict(list)
        for idx, _ in enumerate(self.scales):
            wavelets = results[idx]
            for node in self.nodes:
                # 每一层简单求和
                # embeddings[node].extend(self.get_layer_sum(wavelets, node))
                # 每一层用三元组作为描述符
                embeddings[node].extend(self.get_statistics(wavelets, node))
        self.embeddings = embeddings
        return embeddings

    def parallel_calculate_structural_distance(self, n_workers: int):
        dist_sum_mat = np.zeros((self.n_node, self.n_node), dtype=float)

        pool = multiprocessing.Pool(n_workers)
        result_list = []
        for scale in self.scales:
            res = pool.apply_async(
                HSD.calculate_structural_distance, args=(self, scale, True)
            )
            result_list.append(res)
        pool.close()
        pool.join()

        for res in result_list:
            dist_mat = res.get()
            for idx1 in range(self.n_node):
                for idx2 in range(idx1 + 1, self.n_node):
                    dist_sum_mat[idx1, idx2] += dist_mat[idx1, idx2]
                    dist_sum_mat[idx2, idx1] = dist_sum_mat[idx1, idx2]

        return dist_sum_mat
