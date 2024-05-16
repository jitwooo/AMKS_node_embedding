import sys

package_path = '/home/zjh/graph_embedding/src'
if package_path not in sys.path:
    sys.path.append(package_path)

from experiment_util.parms_tune import Params, BestScoreParamRecorder, search_grid_generator
from model.local_graph import kLocalGraph, load_local_graph, load_local_graph_by_name
from experiment_util.cross_graph import load_node_name, emb_evaluate
from model.AMKS.amks import MultiHopAMKS
import numpy as np
from running.log import get_logger

import datetime
from tqdm import tqdm


class CurParams(Params):
    def __init__(self) -> None:
        super().__init__()
        self._hop = None
        self._sigma = None
        self._step = None

    def update(self, other):
        self._hop = other._hop
        self._sigma = other._sigma
        self._step = other._step

    def to_str(self):
        return f"step:{self._step}_sigma:{self._sigma}_hop:{self._hop}"


if __name__ == '__main__':
    # KDD-ICDM-1 ; SIGIR-CIKM ; SIGMOD-ICDE-1
    graph_name_one = kLocalGraph.sigir
    graph_name_two = kLocalGraph.cikm
    # graph_name_one = sys.argv[1]
    # graph_name_two = sys.argv[2]

    graph_one = load_local_graph(graph_name_one)
    graph_one_node_name = load_node_name(graph_one)

    graph_two = load_local_graph(graph_name_two)
    graph_two_node_name = load_node_name(graph_two)

    common_node_name = set([k for k in graph_one_node_name if graph_one_node_name[k] in graph_one]).intersection(
        set([k for k in graph_two_node_name if graph_two_node_name[k] in graph_two]))
    graph_one_common_id = [graph_one_node_name[name] for name in sorted(common_node_name)]
    graph_two_common_id = [graph_two_node_name[name] for name in sorted(common_node_name)]

    embed_model_one = MultiHopAMKS()
    embed_model_two = MultiHopAMKS()

    embed_model_one.set_g(graph_one)
    embed_model_two.set_g(graph_two)

    time = 0.3
    embed_model_one.set_time(time)
    embed_model_two.set_time(time)

    max_hop = 5
    sigma = 0.4
    step = 30

    sigma_range = np.linspace(0, 5, 20)[1:]
    hop_range = list(range(0, 6))
    step_range = list(range(3, 30, 3))
    k_list = [20, 40]  # Top-k

    recorder_20 = BestScoreParamRecorder(CurParams(), -1, "Top" + str(k_list[0]))
    recorder_40 = BestScoreParamRecorder(CurParams(), -1, "Top" + str(k_list[1]))
    params = CurParams()
    logger = get_logger("ex9_cross_graph_AMKS_" + graph_name_one.value + '-' + graph_name_two.value)
    logger.info("time = " + str(time))

    logger.info("sigma_range:{}".format(sigma_range))
    logger.info("hop_range:{}".format(hop_range))
    logger.info("step_range:{}".format(step_range))

    start_time = datetime.datetime.now()
    for sigma, hop, step in tqdm(list(search_grid_generator(sigma_range, hop_range, step_range))):
        params._hop = hop
        params._sigma = sigma
        params._step = step

        embed_model_one.set_maxhop(hop)
        embed_model_one.set_sigma(sigma)
        embed_model_one.set_step(step)
        embed_model_one.compute_emb_vec()
        graph_one_embed_vec = embed_model_one.get_embedding_vec(
            graph_one_common_id)  # 原版本：sorted(graph_one_common_id) ，不应该对id排序。

        embed_model_two.set_maxhop(hop)
        embed_model_two.set_sigma(sigma)
        embed_model_two.set_step(step)
        embed_model_two.compute_emb_vec()
        graph_two_embed_vec = embed_model_two.get_embedding_vec(graph_two_common_id)  # sorted(graph_two_common_id)

        ans = emb_evaluate(graph_one_embed_vec, graph_two_embed_vec, k_list)
        top_20_score = ans[k_list[0]]
        top_40_score = ans[k_list[1]]

        recorder_20.update_params(params, top_20_score, logger)
        recorder_40.update_params(params, top_40_score, logger)

    time_spent = datetime.datetime.now() - start_time
    min = time_spent.seconds // 60
    sec = time_spent.seconds % 60
    logger.info("time spent: " + str(min) + " min" + str(sec) + " s")
