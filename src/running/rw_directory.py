import os
from model.myGraph import MyGraph

root_path = '/home/zjh/project2/Graph_Embedding_Ex1'
data_path = '/home/zjh/project2/Graph_Embedding_Ex1/data'


# root_path = '/home/zjh/graph_embedding'
# data_path = '/home/zjh/whz/data'


def is_path(path_name):
    result_ = os.path.join(root_path, path_name)
    if not os.path.exists(result_):
        os.mkdir(result_)


def result_path(experiment_index, file_name):
    is_path('results')
    results = os.path.join(root_path, 'results', 'ex' + str(experiment_index))
    if not os.path.exists(results):
        os.mkdir(results)

    return os.path.join(results, file_name)


def temp_file_path(experiment_index, file_name):
    is_path('temp_files')
    temp_path = os.path.join(root_path, 'temp_files', 'ex' + str(experiment_index))
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    return os.path.join(temp_path, file_name)


def log_file_path(file_name):
    log_path = os.path.join(root_path, 'log_files')
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    return os.path.join(log_path, file_name)


def graph_edges_path(graph_name):
    # data_path = os.path.join(root_path, 'data')
    return os.path.join(data_path, 'graph', f"{graph_name}.edgelist")


def graph_label_path(graph_name, label_type):
    # data_path = os.path.join(root_path, 'data')
    return os.path.join(data_path, 'label', '_'.join([graph_name, label_type]).rstrip("_") + ".label")


def node_name_path(graph_name):
    # data_path = os.path.join(root_path, 'data')
    return os.path.join(data_path, 'graph', f"{graph_name}.dict")


def get_cached_eigenxx_path(graph_name: str):
    cached_path = os.path.join(data_path, 'cache_data', "eigenxx")
    if not os.path.exists(cached_path):
        os.mkdir(cached_path)
    return os.path.join(cached_path, graph_name)


def get_reindex_dict_path(graph_name: str):
    dir_path = os.path.join(data_path, 'cache_data', "reindex_dict")
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return os.path.join(dir_path, graph_name)
