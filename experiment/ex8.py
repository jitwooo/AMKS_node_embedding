import sys

package_path = '/home/zjh/graph_embedding/src'
if package_path not in sys.path:
    sys.path.append(package_path)

from experiment_util.parms_tune import Params, BestScoreParamRecorder, search_grid_generator
from model.local_graph import kLocalGraph, load_local_graph
from experiment_util.nodes_classify import load_node_label, kLabelType, knn_evaluate
from model.AMKS.amks import MultiHopAMKS, AMKS
from model.WKS.wks import MultiHopWKS
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


if __name__ == "__main__":
    input_map = {
        0: kLocalGraph.amazon_electronics_computers,
        1: kLocalGraph.amazon_electronics_photo,
        2: kLocalGraph.ms_academic_cs,
        3: kLocalGraph.cora,
        4: kLocalGraph.europe,
        5: kLocalGraph.dmela,
        6: kLocalGraph.usa
    }
    # 4 6 3
    # 6 4 3
    # 6
    input_arg = int(sys.argv[1])
    g = load_local_graph(input_map[input_arg])
    label = load_node_label(g, kLabelType.eigencentrality)
    # compute_model = AMKS()
    # compute_model = MultiHopAMKS()
    compute_model = MultiHopWKS()

    sigma_range = np.linspace(0, 8, 10)[1:]
    hop_range = [1, 2, 3]
    step_range = list(range(2, 30, 2))

    time = 0.26
    compute_model.set_g(g)
    if compute_model.model_name != 'MultiHopWKS':
        compute_model.set_time(time)

    node_index = sorted(g.nodes)
    recorder = BestScoreParamRecorder(CurParams(), -1)
    params = CurParams()
    logger = get_logger("ex8_node_classify_" + compute_model.model_name + "_" + input_map[input_arg].value)
    logger.info("dataset:" + input_map[input_arg].value + " find best params of node classify ")
    logger.info("time = " + str(time))

    logger.info("sigma_range:{}".format(sigma_range))
    logger.info("hop_range:{}".format(hop_range))
    logger.info("step_range:{}".format(step_range))

    start_time = datetime.datetime.now()

    for sigma, hop, step in tqdm(list(search_grid_generator(sigma_range, hop_range, step_range))):
        compute_model.set_maxhop(hop)
        compute_model.set_sigma(sigma)
        compute_model.set_step(step)

        compute_model.compute_emb_vec()
        embedding_vec = compute_model.get_embedding_vec(node_index)
        params._hop = hop
        params._sigma = sigma
        params._step = step
        score = knn_evaluate(embedding_vec, label)

        recorder.update_params(params, score, logger)

    time_spent = datetime.datetime.now() - start_time
    min = time_spent.seconds // 60
    sec = time_spent.seconds % 60
    logger.info("time spent: " + str(min) + " min" + str(sec) + " s")
