from enum import Enum
from running import rw_directory
import networkx as nx
from model.myGraph import MyGraph


class kLocalGraph(Enum):
    sigir = "SIGIR"
    minnesota = "minnesota"
    kdd = "KDD"
    usa = "usa"
    karate = "karate"
    cora_test = "cora_test"
    cikm = "CIKM"
    sigmod = "SIGMOD"
    mkarate2 = "mkarate2"
    barbell = "barbell"
    tree = "tree"
    icde = "ICDE"
    dmela = "dmela"
    citeseer = "citeseer"
    icdm = "ICDM"
    cora = "cora"
    europe = "europe"
    bio_grid_human = "bio_grid_human"
    bio_dmela = "bio_dmela"
    mkarate = "mkarate"
    amazon_electronics_computers = "amazon_electronics_computers"
    amazon_electronics_photo = "amazon_electronics_photo"
    ms_academic_cs = "ms_academic_cs"
    karate_mirrored = "karate-mirrored"
    karate_mirrored0 = "karate-mirrored0"
    test_data = "test_data"


def read_edge_list(path):
    edge_list = []
    with open(path, 'r') as f:
        for line in f:
            x, y = line.strip().split()
            edge_list.append((int(x), int(y)))
    return edge_list


def load_local_graph_by_name(graph_name):
    path = rw_directory.graph_edges_path(graph_name)
    try:
        temp_g = nx.read_edgelist(path, nodetype=int, data=False)
    except:
        temp_g = nx.read_gpickle(path)
    g = MyGraph()
    g.add_nodes_from(temp_g)
    g.add_edges_from(temp_g.edges)
    g.set_graph_tag(graph_name)
    return g


def load_local_graph(graph: kLocalGraph):
    graph_name = graph.value
    return load_local_graph_by_name(graph_name)
