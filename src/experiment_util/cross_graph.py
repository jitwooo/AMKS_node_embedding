from running import rw_directory
from model.myGraph import MyGraph
import numpy as np

import networkx as nx


def load_node_name(g: MyGraph):
    path = rw_directory.node_name_path(g.graph_tag)
    name2id = {}
    with open(path) as f:
        for line in f:
            name, node = line.strip().split("\t")
            name2id[name] = int(node)
    return name2id


def emb_evaluate(emb_1, emb_2, k_list=None):
    if k_list is None:
        k_list = [20, 40]
    emb_1 /= np.linalg.norm(emb_1, axis=1).reshape(-1, 1)
    emb_2 /= np.linalg.norm(emb_2, axis=1).reshape(-1, 1)

    all_results = {k: [] for k in k_list}
    for i in range(emb_1.shape[0]):
        v = emb_1[i]
        scores = emb_2.dot(v)

        idxs = scores.argsort()[::-1]
        for k in k_list:
            all_results[k].append(i in idxs[:k])
    res = dict((k, sum(all_results[k]) / len(all_results[k])) for k in k_list)

    return res


def extract_sub_graph_with_level(g: MyGraph, node_id, level):
    cur_level = [node_id]
    visited = set()
    for _ in range(level + 1):
        next_level = []
        for v in cur_level:
            if v not in visited:
                next_level.extend(list(g.neighbors(v)))
                visited.add(v)
        cur_level = next_level
    temp_g = nx.subgraph(g, list(visited))
    sub_g = MyGraph()
    sub_g.add_nodes_from(temp_g.nodes)
    sub_g.add_edges_from(temp_g.edges)
    sub_g.set_graph_tag(g.graph_tag + f"-node-{node_id}-level-{level}")
    return sub_g
