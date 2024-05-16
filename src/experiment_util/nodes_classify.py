from enum import Enum
from model.myGraph import MyGraph
from running import rw_directory
import numpy as np
from sklearn import utils as sk_utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


class kLabelType(Enum):
    eigencentrality = "EigenCentrality"
    pagerank = "PageRank"
    plain = ""


def load_node_label(g: MyGraph, label_type: kLabelType):
    graph_name = g.graph_tag
    label_type = label_type.value
    path = rw_directory.graph_label_path(graph_name, label_type)
    node_2_labels = {}
    with open(path, 'r') as f:
        for line in f:
            node, label = line.strip().split()
            node_2_labels[int(node)] = int(label)
    labels = []
    for _, label in sorted(node_2_labels.items(), key=lambda x: x[0]):
        labels.append(label)
    return np.array(labels)


def knn_evaluate(data, labels, metric="minkowski", cv=5, n_neighbor=10):
    """
    基于节点的相似度进行KNN分类，在嵌入之前进行，为了验证通过层次化相似度的优良特性。
    """
    data, labels = sk_utils.shuffle(data, labels)
    knn = KNeighborsClassifier(weights='uniform', algorithm="auto", n_neighbors=n_neighbor, metric=metric, p=2)
    test_scores = cross_val_score(knn, data, y=labels, cv=cv, scoring="accuracy")
    # logging.info(f"KNN: tests scores:{test_scores}, mean_score={np.mean(test_scores)}\n")
    return np.mean(test_scores)
