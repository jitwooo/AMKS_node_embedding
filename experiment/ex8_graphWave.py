import sys
package_path = '/home/zjh/graph_embedding/src'
if package_path not in sys.path:
    sys.path.append(package_path)

from experiment_util.parms_tune import Params, BestScoreParamRecorder, search_grid_generator
from model.local_graph import kLocalGraph, load_local_graph
from experiment_util.nodes_classify import load_node_label, kLabelType, knn_evaluate
import sys
import numpy as np
from running.log import get_logger

from tqdm import tqdm
from model.graphWave.HSD import MultiHSD


class CurParams(Params):
    def __init__(self) -> None:
        super().__init__()
        self._neighbours = None
        self._scales = None

    def update(self, other):
        self._neighbours = other._neighbours
        self._scales = other._scales

    def to_str(self):
        return f"neighbours:{self._neighbours}_scales:{self._scales}"


if __name__ == "__main__":
    base_logger = get_logger("ex8_graphWave")
    input_map = {
        0: kLocalGraph.amazon_electronics_computers,
        1: kLocalGraph.amazon_electronics_photo,
        2: kLocalGraph.ms_academic_cs,
        3: kLocalGraph.cora,
        4: kLocalGraph.europe,
        5: kLocalGraph.dmela,
        6: kLocalGraph.usa
    }
    base_logger.info("input args is" + str(list(sys.argv)))
    # input_arg = int(sys.argv[1])
    # why why why
    # cora usa europe
    # 6 3 4
    input_arg = 4
    base_logger.info("dataset is " + str(input_map[input_arg]))
    g = load_local_graph(input_map[input_arg])
    label = load_node_label(g, kLabelType.eigencentrality)

    hops = np.arange(1, 7, 1)
    scales = np.arange(50, 100, 10)

    node_index = sorted(g.nodes)
    recorder = BestScoreParamRecorder(CurParams(), -1)
    params = CurParams()

    for hop, n_scales in tqdm(list(search_grid_generator(hops, scales))):
        compute_model = MultiHSD(g, g.graph_tag, hop, n_scales)

        compute_model.init(is_stable=False)

        embedding_vec = compute_model.embed()

        X = np.array(embedding_vec)
        score = knn_evaluate(embedding_vec, label)

        recorder.update_params(params, score, base_logger)
