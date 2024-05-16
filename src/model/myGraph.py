import networkx as nx


class MyGraph(nx.Graph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)
        if hasattr(self, "_graph_tag"):
            raise TypeError(f"{type(super())} has attr _graph_tag")
        self._graph_tag = None

    def set_graph_tag(self, graph_tag):
        self._graph_tag = graph_tag

    @property
    def graph_tag(self):
        return self._graph_tag

# class MyGraph(nx.Graph):
#     def __init__(self, incoming_graph_data=None, **attr):
#         super().__init__(incoming_graph_data=incoming_graph_data, **attr)


# 并不明白自己封装一层的意义在哪
