from model.myGraph import MyGraph
import networkx as nx
import matplotlib.pyplot as plt
import sys
import numpy as np


def load_fork_tree(forks: list):
    """
    生成多叉树
    forks:多叉树每一层各顶点拥有子顶点数量
    """
    if len(forks) == 0:
        raise ValueError("forks can't be empty")

    all_level_records = []
    last_level_nodes = [[0]]
    cur_level_nodes = []
    all_level_records.append(last_level_nodes)
    g = MyGraph()
    g.add_node(0)
    for fork in forks:
        for i in range(len(last_level_nodes)):
            for node in last_level_nodes[i]:
                cur_nodes = list(range(len(g), len(g) + fork))
                cur_level_nodes.append(cur_nodes)
                g.add_edges_from([(node, cur_node) for cur_node in cur_nodes])
        all_level_records.append(cur_level_nodes)
        last_level_nodes = cur_level_nodes
        cur_level_nodes = []

    g.set_graph_tag(str(forks))
    return g, all_level_records


def plot_networkx(graph, role_labels, node_color=None):
    cmap = plt.get_cmap('tab20_r')
    x_range = np.linspace(0.1, 0.9, len(np.unique(role_labels)))
    coloring = {u: cmap(x_range[i]) for i, u in enumerate(np.unique(role_labels))}
    if node_color is None:
        node_color = [coloring[role_labels[i]] for i in graph.nodes]
    nx.draw_networkx(graph, pos=nx.layout.kamada_kawai_layout(graph),
                     node_color=node_color, cmap='tab20_r')
    return node_color


def cycle(start, len_cycle, role_start=0, plot=False):
    """
    Builds a cycle graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    plot        :    boolean -- should the shape be printed?
    OUTPUT:
    -------------
    graph           :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + len_cycle))
    for i in range(len_cycle - 1):
        graph.add_edges_from([(start + i, start + i + 1)])
    graph.add_edges_from([(start + len_cycle - 1, start)])
    roles = [role_start] * len_cycle
    if plot is True:
        plot_networkx(graph, roles)
    return graph, roles


def house(start, role_start=0, plot=False, attached=False):
    """
    Builds a house-like  graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    plot        :    boolean -- should the shape be printed?
    OUTPUT:
    -------------
    graph           :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 5))
    graph.add_edges_from([(start, start + 1), (start + 1, start + 2),
                          (start + 2, start + 3), (start + 3, start)])
    graph.add_edges_from([(start, start + 2), (start + 1, start + 3)])
    graph.add_edges_from([(start + 4, start), (start + 4, start + 1)])
    if not attached:
        roles = [role_start, role_start, role_start + 1,
                 role_start + 1, role_start + 2]
    else:
        roles = [role_start, role_start + 1, role_start + 2,
                 role_start + 2, role_start + 3]
    if plot is True:
        plot_networkx(graph, roles)
    return graph, roles


def star(start, nb_branches, role_start=0, plot=False):
    """
    Builds a star graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    nb_branches :    int corresponding graph to the nb of star branches
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    plot        :    boolean -- should the shape be printed?
    OUTPUT:
    -------------
    graph           :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + nb_branches + 1))
    for k in range(1, nb_branches + 1):
        graph.add_edges_from([(start, start + k)])
    roles = [role_start + 1] * (nb_branches + 1)
    roles[0] = role_start
    if plot is True:
        plot_networkx(graph, roles)
    return graph, roles


def fan(start, nb_branches, role_start=0, plot=False):
    """
    Builds a fan-like graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    nb_branches :    int corresponding graph to the nb of fan branches
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    plot        :    boolean -- should the shape be printed?
    OUTPUT:
    -------------
    graph           :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph, roles = star(start, nb_branches, role_start=role_start)
    begin = 1
    end = nb_branches - 1
    cnt = 1
    while begin <= end:
        roles[begin] += cnt
        roles[end] = roles[begin]
        begin += 1
        end -= 1
        cnt += 1
    for k in range(1, nb_branches - 1):
        # roles[k] += 1
        # roles[k + 1] += 1
        graph.add_edges_from([(start + k, start + k + 1)])
    if plot is True:
        plot_networkx(graph, roles)
    return graph, roles


def build_circle_structure_split(shapes: list, intervals: int, repeats):
    basis_width = len(shapes) * (intervals + 1) * repeats
    basis_circle, basis_roles = cycle(0, basis_width)
    basis_roles = []
    role_start = 1
    repeat_length = len(shapes) + intervals * len(shapes)
    for _ in range(repeats):
        basis_roles.extend(list(range(repeat_length)))

    id_start = max(basis_circle) + 1
    role_start = max(basis_roles) + 1
    roles = [] + basis_roles

    attach_nodes = list(range(0, len(basis_circle), intervals + 1))

    seen_shapes = {}
    for i, attach_node in enumerate(attach_nodes):
        attach_shape_cmd = shapes[i % len(shapes)]
        if attach_shape_cmd[0] is house:
            attach_shape, attach_shape_roles = attach_shape_cmd[0](
                *([id_start] + attach_shape_cmd[1:] + [0, False, True]))
        else:
            attach_shape, attach_shape_roles = attach_shape_cmd[0](*([id_start] + attach_shape_cmd[1:]))
        if attach_shape_cmd[0] not in seen_shapes:
            attach_shape_roles = [v + role_start for v in attach_shape_roles]
            seen_shapes[attach_shape_cmd[0]] = attach_shape_roles
        else:
            attach_shape_roles = seen_shapes[attach_shape_cmd[0]]
        basis_circle.add_edges_from(attach_shape.edges)
        basis_circle.add_edge(attach_node, id_start)
        roles.extend(attach_shape_roles)
        id_start = max(basis_circle) + 1
        role_start = max(roles) + 1

    def encode_shapes(shapes):
        res = []
        for val in shapes:
            if isinstance(val, (tuple, list)):
                res.append(encode_shapes(val))
            elif callable(val):
                res.append(val.__name__)
            else:
                res.append(str(val))
        return "_".join(res)

    ret_g = MyGraph()
    ret_g.add_edges_from(basis_circle.edges)
    ret_g.set_graph_tag(f"{sys._getframe().f_code.co_name}_{encode_shapes(shapes)}_{intervals}_{repeats}")
    return ret_g, roles
