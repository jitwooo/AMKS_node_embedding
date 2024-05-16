from typing import Dict, Tuple
import networkx as nx
from model.myGraph import MyGraph
from model.utils import is_interger, is_consecutive, is_zero_start, extract_subgraphs
import numpy as np


def build_node_idx_map(graph) -> Tuple[Dict, Dict]:
    """
    建立图节点与标号之间的映射关系，方便采样。
    :param graph:
    :return:
    """
    node2idx = {}
    idx2node = {}
    node_size = 0
    for node in nx.nodes(graph):
        node2idx[node] = node_size
        idx2node[node_size] = node
        node_size += 1
    return idx2node, node2idx


