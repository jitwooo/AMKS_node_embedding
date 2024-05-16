from model.myGraph import MyGraph
import numpy as np
import networkx as nx
from running import caches


def gauss_kernel(e: np.ndarray, mu: float, sigma: float):
    """对数正态分布"""
    return np.exp(-np.power((e - mu), 2) / (2 * sigma * sigma))


def is_vector(m: np.ndarray):
    return len(m.shape) == 1


def is_matrix(m: np.ndarray):
    return len(m.shape) == 2


def is_square_matrix(m: np.ndarray):
    if is_matrix(m):
        return m.shape[0] == m.shape[1]
    return False


def sinc(x):
    return np.sinc(x / np.pi)


def cal_amks(eigenvalues, eigenvectors, P, sigma, desc_size, log_eigenvalue, mu_s, time):
    amks_embedding_vectors = np.zeros((eigenvectors.shape[0], desc_size))
    P = np.power(eigenvectors, 2)
    eigenvalues = eigenvalues.reshape(1, -1)
    log_eigenvalue = log_eigenvalue.reshape(1, -1)  # reshape方便后面的转制操作 ([]) -->([[]])
    F1 = sinc(time * (eigenvalues - eigenvalues.T))
    for i in range(desc_size):
        F2 = gauss_kernel(log_eigenvalue, mu_s[i], sigma)
        F2 = F2 * F2.T
        temp = np.sum(np.triu(F2))  # triu 取上三角矩阵，求和得到一个标量
        F = F1 * F2 / temp
        temp_vec = P.T * (P @ F).T  # 点积，既对应位置元素相乘
        temp_vec = np.sum(temp_vec, axis=0)  # 每一列的行相加，得到一个行向量
        amks_embedding_vectors[:, i] = temp_vec
    return amks_embedding_vectors


# add
@caches.LRUMemCache(cache_on=True, hash_fn=lambda x, y=False: (x.graph_tag, y))
@caches.ClassComputeEigenxxDiskCache(cache_on=True)
def compute_laplacian_eigenxx(g: MyGraph, is_normalized=False):
    """
    计算图拉普拉斯矩阵
    带有顶点序号字典
    """
    node_seq = sorted(g.nodes)
    L = nx.laplacian_matrix(g, nodelist=node_seq).todense()
    L = np.array(L)

    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # I have no idea what do you want to do
    reindex_dict = {v: i for i, v in enumerate(node_seq)}

    return eigenvalues, eigenvectors, reindex_dict


def compute_amks(eigenvalue: np.ndarray, eigenvector: np.ndarray, P, amks_variance, desc_size, time=0.5,
                 conv_kernel_fn=gauss_kernel):
    """
    Args:
    eval: 特征值
    evec: 特征向量
    P: 公式中的投影矩阵，实现上目前使用特征向量的平方
    amks_variance: sigma
    desc_size: 采样点数量
    conv_kernel_fn: 卷积核
    """
    if not (isinstance(eigenvalue, np.ndarray) and isinstance(eigenvector, np.ndarray)):
        raise ValueError("eval and evec must be np.ndarray")
    if not (is_vector(eigenvalue) and is_square_matrix(eigenvector)):
        raise ValueError("eval must be vector and evec must be matrix")
    if eigenvalue.shape[0] != eigenvector.shape[0]:
        raise ValueError(
            f"the eval is with shape ({eigenvalue.shape[0]}), and evec's shape must be ({eigenvalue.shape[0]},{eigenvalue.shape[0]})")

    log_eigenvalue = np.log(np.maximum(eigenvalue, 10e-6))
    # mu_s = np.linspace(np.min(log_eigenvalue[1]), np.max(log_eigenvalue) / 1.02, desc_size)
    mu_s = np.linspace(np.min(log_eigenvalue), np.max(log_eigenvalue) / 1.02, desc_size)
    sigma = max((mu_s[1] - mu_s[0]) * amks_variance, 1e-3)

    amks_embedding_vector = cal_amks(eigenvalue, eigenvector, P, sigma, desc_size, log_eigenvalue, mu_s, time)
    return amks_embedding_vector


def is_consecutive(seq):
    return (max(seq) - min(seq) + 1) == len(seq)


def is_interger(seq):
    for e in seq:
        if not isinstance(e, int):
            return False
    return True


def is_zero_start(seq):
    return min(seq) == 0


def extract_subgraphs(g: MyGraph):
    """

    :param g:
    :return: 连通分量
    """
    subgraph_nodes = nx.connected.connected_components(g)
    return [nx.subgraph(g, nodes) for nodes in subgraph_nodes]


def get_padded_neighbor_dict(g: MyGraph):
    """
    the dict stored with format\n
    {
        ...
        hop_i : [[i hop neighbors of v_i]...]\n
        ...
    }\n
    and all the list in the mapping list of hop_i must be with the\n
    same length
    """
    nodes_seq = list(g.nodes)
    if not is_interger(nodes_seq):
        raise ValueError("the lable of nodes must be integer")
    if not is_consecutive(sorted(nodes_seq)):
        raise ValueError("the node label of g must be consecutive")
    if not is_zero_start(nodes_seq):
        raise ValueError("the node lable must start from zero")
    subgraphs = extract_subgraphs(g)
    max_diameter = max([nx.diameter(subgraph) for subgraph in subgraphs])
    padding_node_label = len(g)
    neighbor_dict = {hop: [[] for _ in range(len(g))] for hop in range(max_diameter + 1)}
    for v in g.nodes:
        hop_dict = dict(nx.single_source_shortest_path_length(g, v, cutoff=max_diameter))
        for node, hop in hop_dict.items():
            neighbor_dict[hop][v].append(node)
    for hop in neighbor_dict:
        nbrs = neighbor_dict[hop]
        cur_max_size = len(max(nbrs, key=len))
        for v in range(len(g)):
            neighbor_dict[hop][v].extend(
                [padding_node_label] * (cur_max_size - len(neighbor_dict[hop][v])))
    return neighbor_dict


def get_padded_neighbor_vec(g: MyGraph):
    padded_neighbor_dict = get_padded_neighbor_dict(g)
    padded_neighbor_vec = {}
    for hop in padded_neighbor_dict:
        padded_neighbor_vec[hop] = np.array(padded_neighbor_dict[hop])
    return padded_neighbor_vec


def get_mus_sigma(g: MyGraph, amks_variance, desc_size):
    eval, evec, reinedex_dict = compute_laplacian_eigenxx(g)
    log_E = np.log(np.maximum(eval, 1e-6))
    mus = np.linspace(np.min(log_E[1]), np.max(log_E) / 1.02, desc_size)  # weihz:创建等差数列，均匀取不同的 miu(高斯核里面的均值)，总共step个
    sigma = max((mus[1] - mus[0]) * amks_variance, 1e-3)
    return mus, sigma


def compute_wks_col(u_power, conv_kernel):
    return np.sum(u_power * (conv_kernel.reshape(1, -1)), axis=1, keepdims=False) / np.sum(conv_kernel, keepdims=False)


def compute_wks(e: np.ndarray, u: np.ndarray, wks_variance, step,
                conv_kernel_fn=gauss_kernel):  # whz:卷积核函数默认高斯核，WKS里面就是,参考雄哥公式 3-15
    """
    Args:
    e: 特征值
    u: 特征向量
    wks_variance: sigma
    step: 采样点数量
    conv_kernel_fn: 卷积核
    """
    if not (isinstance(e, np.ndarray) and isinstance(u, np.ndarray)):
        raise ValueError("e and u must be np.ndarray")
    if not (is_vector(e) and is_square_matrix(u)):
        raise ValueError("e must be vector and u must be matrix")
    if e.shape[0] != u.shape[0]:
        raise ValueError(
            f"the e is with shape ({e.shape[0]}), and u's shape must be ({e.shape[0]},{e.shape[0]})")

    log_e = np.log(np.maximum(e, 1e-6))
    mus = np.linspace(np.min(log_e[1]), np.max(log_e) / 1.02, step)  # weihz:创建等差数列，均匀取不同的 miu(高斯核里面的均值)，总共step个
    sigma = max((mus[1] - mus[0]) * wks_variance, 1e-3)
    u_power = np.power(u, 2)
    wks_embedding_vec = np.zeros((u.shape[0], step))

    for col_index in range(step):
        wks_embedding_vec[:, col_index] = compute_wks_col(u_power, conv_kernel_fn(log_e, mus[col_index], sigma))

    return wks_embedding_vec


def get_padded_neighbor_vec(g: MyGraph):
    padded_neighbor_dict = get_padded_neighbor_dict(g)
    padded_neighbor_vec = {}
    for hop in padded_neighbor_dict:
        padded_neighbor_vec[hop] = np.array(padded_neighbor_dict[hop])
    return padded_neighbor_vec
