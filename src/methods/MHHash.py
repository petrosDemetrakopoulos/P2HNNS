from typing import List
from heapq import heappush, heappop
import numpy as np
from tqdm import tqdm
from ..operations import dot_with_start
from .Hash import Hash
from ..Query import Query
from ..IdxVal import IdxVal
from ..HashBucket import HashBucket

class MHHash(Hash):
    """
    Multilinear Hyperplane (MH) Hash.
    """
    def __init__(self, dimension: int, m: int, l: int, M: int):
        self.m = m
        self.l = l
        self.M = M
        size = m * l * M * dimension
        self.randv = np.array([np.random.normal(0.0, 1.0) for _ in range(size)])
        self.buckets = HashBucket(dimension,l)

    def hash_data(self, data: np.array) -> np.array:
        assert len(self.randv) == self.m * self.l * self.M * len(data)
        sigs = np.zeros(self.l, dtype=int)
        for i in range(self.l):
            sig = 0
            for j in range(self.m):
                pos = (i * self.m + j) * self.M * len(data)
                val = self._hash(data, pos)
                sign = int(0 < val)
                sig = (sig << 1) | sign
            sigs[i] = sig
        return sigs

    def hash_query(self, query: np.array) -> np.array:
        assert len(self.randv) == self.m * self.l * self.M * len(query)
        sigs = np.zeros(self.l, dtype=int)
        for i in range(self.l):
            sig = 0
            for j in range(self.m):
                pos = (i * self.m + j) * self.M * len(query)
                val = self._hash(query, pos)
                sign = int(0 >= val)
                sig = (sig << 1) | sign
            sigs[i] = sig
        return sigs

    def _hash(self, query: np.array, pos: int) -> float:
        val = 1.0
        for k in range(self.M):
            val *= dot_with_start(query, pos + k * len(query), self.randv)
        return val

    def build_index(self, data: np.array):
        n = len(data)
        print("-- Building index... --")
        for i in tqdm(range(n)):
            sig = self.hash_data(data[i])
            self.buckets.insert(i, sig)

    def nns(self, param: Query) -> List[IdxVal]:
        data = param.data
        query = param.query
        top = param.top
        limit = param.limit

        sig = self.hash_query(query)
        heap = []
        dist_fun = param.dist

        def accept(key: int):
            nonlocal heap, query, data, dist_fun, top
            dist = dist_fun.distance(query, data[key])
            heappush(heap, IdxVal(key, dist))
            if top < len(heap):
                heappop(heap)

        self.buckets.search(sig, limit, accept)

        result = sorted(heap, key=lambda x: x.value)
        return result
