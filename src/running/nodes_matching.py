import numpy as np


def is_Top_k(emb_1, emb_2, index_list=None, k_list=None):
    if k_list is None:
        k_list = [5, 10]
    if index_list is None:
        index_list = [0]
    emb_1 /= np.linalg.norm(emb_1, axis=1).reshape(-1, 1)
    emb_2 /= np.linalg.norm(emb_2, axis=1).reshape(-1, 1)

    all_results = {k: {} for k in k_list}
    for index in index_list:
        for k in k_list:
            all_results[k][index] = []

    for index in index_list:
        v = emb_1[index]
        scores = emb_2.dot(v)
        indices = scores.argsort()[::-1]
        temp_list = []
        for k in k_list:
            all_results[k][index].append(index in indices[:k])  # = temp_list.copy()
        # temp_list.clear()
    res = dict(
        (k, all_results[k]) for k in k_list
    )

    return res
