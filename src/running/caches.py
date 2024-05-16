import pickle
import os

import numpy as np
from model.myGraph import MyGraph
from running import rw_directory
from running.lru_recorder import LRURecorder

_registered_caches = {}


class MemCaches:
    def __init__(self, cache_on=False, hash_fn=None) -> None:
        self._cahce_on = cache_on
        self._hash_fn = hash_fn
        self._wrappered_fn = None
        self._cache_dict = {}

    def __call__(self, fn):
        self._wrappered_fn = fn
        return self.wrapper_fn

    def get_cached_data(self, key):
        if self._cahce_on:
            return self._cache_dict.get(key, None)

    def cache_data(self, key, val):
        if self._cahce_on:
            self._cache_dict[key] = val

    def wrapper_fn(self, *args, **kwargs):
        res = None
        # print(args, kwargs)
        key = self._hash_fn(*args, **kwargs)
        if self._cahce_on:
            res = self.get_cached_data(key)
            if res is None:
                res = self._wrappered_fn(*args, **kwargs)
                self.cache_data(key, res)
        return res


class LRUMemCache(MemCaches):
    """
    this is a lru memcache.
    Note: the internal implementation of LRURecorder is not thread safe
    """

    def __init__(self, cache_on=False, hash_fn=None, cache_size=20) -> None:
        super().__init__(cache_on=cache_on, hash_fn=hash_fn)
        self._cache_size = cache_size
        self._recorder = LRURecorder(size=cache_size)

    def get_cached_data(self, key):
        if self._cahce_on:
            return self._recorder.find(key)

    def cache_data(self, key, val):
        if self._cahce_on:
            return self._recorder.insert_record(key, val)


def pickle_dump_helper(path, value):
    with open(path, 'wb') as f:
        pickle.dump(value, f)


def pickle_load_helper(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class DiskCaches:
    def __init__(self, cache_on=False, disk_path_fn=None, load_fn=None, save_fn=None) -> None:
        """
        save_fn's input should be like (path, values)
        """
        self._cache_on = cache_on
        self._disk_path_fn = disk_path_fn
        self._wrappered_fn = None
        self._save_fn = save_fn
        self._load_fn = load_fn

    def __call__(self, fn):
        self._wrappered_fn = fn
        return self.wrapper_fn

    def load_cached_data(self, path):
        if self._cache_on:
            if os.path.exists(path):
                return self._load_fn(path)

    def save_data(self, path, value):
        if self._cache_on:
            self._save_fn(path, value)

    def wrapper_fn(self, *args, **kwargs):
        path = self._disk_path_fn(*args, **kwargs)
        res = None
        if self._cache_on:
            res = self.load_cached_data(path)
            if res is None:
                res = self._wrappered_fn(*args, **kwargs)
                self.save_data(path, res)
        return res


class ClassComputeEigenxxDiskCache:
    def __init__(self, cache_on=False) -> None:
        self._cache_on = cache_on
        self._wappered_fn = None

    def __call__(self, fn):
        self._wappered_fn = fn
        return self.wrapper_fn

    def load_cached_eigenxx(self, path):
        if self._cache_on:
            if os.path.exists(path):
                if os.path.exists(path):
                    res = np.load(path)
                    return res['eigenvalues'], res['eigenvectors']

    def save_eigenxx(self, path, eigenvalues, eigenvectors):
        if self._cache_on:
            np.savez(path, eigenvalues=eigenvalues, eigenvectors=eigenvectors)

    def save_reindex_dict(self, path, value):
        if self._cache_on:
            pickle_dump_helper(path, value)

    def load_reindex_dict(self, path):
        if self._cache_on:
            if os.path.exists(path):
                return pickle_load_helper(path)

    def wrapper_fn(self, g: MyGraph, is_normalized=False):
        eigenxx_path = rw_directory.get_cached_eigenxx_path(
            f"{g.graph_tag}_{is_normalized}.npz")
        reindex_dict_path = rw_directory.get_reindex_dict_path(
            f"{g.graph_tag}.pickle")
        res = [None] * 3
        if self._cache_on:
            reindex_dict = self.load_reindex_dict(reindex_dict_path)
            eigenxx = self.load_cached_eigenxx(eigenxx_path)
            if reindex_dict is None or eigenxx is None:
                res = self._wappered_fn(g, is_normalized)
                self.save_reindex_dict(reindex_dict_path, res[-1])
                self.save_eigenxx(eigenxx_path, res[0], res[1])
            else:
                res[0], res[1], res[2] = eigenxx[0], eigenxx[1], reindex_dict
        return res


from multiprocessing import Manager


class InMemoryPickleCache:
    def __init__(self) -> None:
        self._manager = Manager()
        self._lock = self._manager.Lock()
        self.save_dict = self._manager.dict()

    def save(self, name, obj_pkl_str):
        self._lock.acquire()
        # shm = shared_memory.SharedMemory(name=name,create=True,size=len(obj_pkl_str))
        # shm.buf[:] = obj_pkl_str[:]
        self.save_dict.update({name: obj_pkl_str})
        self._lock.release()

    def get(self, name):
        ret = None
        self._lock.acquire()
        # shm = shared_memory.SharedMemory(name=name,create=False)
        # ret = shm.buf[:]
        print(name)
        ret = self.save_dict[name][:]
        self._lock.release()
        return ret
