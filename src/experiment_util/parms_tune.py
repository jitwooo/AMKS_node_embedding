import abc
import numpy as np


class Params(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def update(self, other):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_str(self):
        raise NotImplementedError


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
        logger.info(f"current {self._score_name} score is:")
        logger.info(f"{score}")
        logger.info(f"current params is:")
        logger.info(f"{other_param.to_str()}")
        logger.info("$" * 20)


def search_grid_generator(*args):
    all_indices = [list(range(len(arg))) for arg in args]
    search_mesh_axis = np.meshgrid(*all_indices)
    search_indices = [search_mesh_axe.reshape(-1) for search_mesh_axe in search_mesh_axis]
    for indices in zip(*search_indices):
        res = []
        for i, index in enumerate(indices):
            res.append(args[i][index])
        yield res
