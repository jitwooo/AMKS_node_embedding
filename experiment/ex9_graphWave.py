import sys

package_path = '/home/zjh/graph_embedding/src'
if package_path not in sys.path:
    sys.path.append(package_path)

from running.log import get_logger
from model.local_graph import kLocalGraph, load_local_graph, load_local_graph_by_name
from experiment_util.cross_graph import load_node_name, emb_evaluate
from experiment_util.parms_tune import BestScoreParamRecorder, search_grid_generator
from model.graphWave.HSD import MultiHSD
from experiment_util.parms_tune import Params
import numpy as np
from tqdm import tqdm
import datetime


class CurParams(Params):
    def __init__(self) -> None:
        super().__init__()
        self._hop = None
        self._n_scales = None

    def update(self, other):
        self._hop = other._hop
        self._n_scales = other._n_scales

    def to_str(self):
        return f"n_scales:{self._n_scales}_hop:{self._hop}"


class BestScoreParamRecorder:
    def __init__(self, params: Params, score, score_name="") -> None:
        self._params = params
        self._score = score
        self._score_name = score_name

    def update_params(self, other_param, score, logger):
        if score > self._score:
            self._params.update(other_param)
            self._score = score
        logger.info(f"best {self._score_name} score is:")
        logger.info(f"{self._score}")
        logger.info(f"best param is:")
        logger.info(f"{self._params.to_str()}")
        logger.info("=" * 20)
        logger.info(f"cur {self._score_name} score is:")
        logger.info(f"{score}")
        logger.info(f"cur params is:")
        logger.info(f"{other_param.to_str()}")
        logger.info("$" * 20)


if __name__ == "__main__":
    # graph_name_one = kLocalGraph.kdd
    # graph_name_two = kLocalGraph.icdm

    # graph_name_one = kLocalGraph.sigir
    # graph_name_two = kLocalGraph.cikm

    # graph_name_one = kLocalGraph.sigmod
    # graph_name_two = kLocalGraph.icde
    # KDD-ICDM-1 ; SIGIR-CIKM ; SIGMOD-ICDE-1
    graph_name_one = sys.argv[1]
    graph_name_two = sys.argv[2]
    print(f'working on {graph_name_one} and {graph_name_two}')

    graph_one = load_local_graph_by_name(graph_name_one)
    graph_one_node_name = load_node_name(graph_one)

    graph_two = load_local_graph_by_name(graph_name_two)
    graph_two_node_name = load_node_name(graph_two)

    common_node_name = set([k for k in graph_one_node_name if graph_one_node_name[k] in graph_one]).intersection(
        set([k for k in graph_two_node_name if graph_two_node_name[k] in graph_two]))
    graph_one_common_id = [graph_one_node_name[name] for name in sorted(common_node_name)]
    graph_two_common_id = [graph_two_node_name[name] for name in sorted(common_node_name)]

    hops = np.arange(1, 7, 1)
    scales = np.arange(50, 100, 10)
    k_list = [20, 40]  # Top-k

    recorder_20 = BestScoreParamRecorder(CurParams(), -1, "Top" + str(k_list[0]))
    recorder_40 = BestScoreParamRecorder(CurParams(), -1, "Top" + str(k_list[1]))
    params = CurParams()
    logger = get_logger("ex9_cross_graph_HSD_" + graph_name_one + '-' + graph_name_two)
    # logger.info("time = "+str(time))

    logger.info("hops:{}".format(hops))
    logger.info("scales:{}".format(scales))
    # logger.info("step_range:{}".format(step_range))
    start_time = datetime.datetime.now()

    for hop, n_scales in tqdm(list(search_grid_generator(hops, scales))):
        params._hop = hop
        params._n_scales = n_scales

        embed_model_one = MultiHSD(graph_one, graph_one.graph_tag, hop, n_scales)
        embed_model_one.init(is_stable=False)
        embed_model_one.embed()
        graph_one_embed_vec = embed_model_one.get_embedding_vec(graph_one_common_id)

        embed_model_two = MultiHSD(graph_two, graph_two.graph_tag, hop, n_scales)
        embed_model_two.init(is_stable=False)
        embed_model_two.embed()
        graph_two_embed_vec = embed_model_two.get_embedding_vec(graph_two_common_id)

        ans = emb_evaluate(graph_one_embed_vec, graph_two_embed_vec, k_list)
        top_20_score = ans[k_list[0]]
        top_40_score = ans[k_list[1]]

        recorder_20.update_params(params, top_20_score, logger)
        recorder_40.update_params(params, top_40_score, logger)

    time_spent = datetime.datetime.now() - start_time
    min = time_spent.seconds // 60
    sec = time_spent.seconds % 60
    logger.info("time spent: " + str(min) + " min" + str(sec) + " s")
