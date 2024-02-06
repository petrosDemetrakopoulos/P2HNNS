from typing import List
from heapq import heappush, heappop
from tqdm import tqdm
import numpy as np
from .Hash import Hash
from ..Query import Query
from ..IdxVal import IdxVal
from ..HashBucket import HashBucket

class EHHash(Hash):
    """
    Embedding Hyperplane(EH) Hash.
    """
    def __init__(self, d: int, m: int, l: int):
        self.buckets = HashBucket(d,l)
        self.m = m
        self.l = l
        size = m * l * d * d
        self.randv = np.random.normal(0.0, 1.0, size)

    def hash_data(self, data: np.array):
        assert len(self.randv) == self.m * self.l * len(data) * len(data)
        sigs = np.zeros(self.l, dtype=int)
        for i in range(self.l):
            sig = 0
            for j in range(self.m):
                pos = (i * self.m + j) * len(data) * len(data)
                val = self._hash(data, pos)
                sign = int(0 < val)
                sig = (sig << 1) | sign
            sigs[i] = sig
        return sigs

    def hash_query(self, query: np.array) -> np.array:
        assert len(self.randv) == self.m * self.l * len(query) * len(query)
        sigs = np.zeros(self.l, dtype=int)
        for i in range(self.l):
            sig = 0
            for j in range(self.m):
                pos = (i * self.m + j) * len(query) * len(query)
                val = self._hash(query, pos)
                sign = int(0 >= val)
                sig = (sig << 1) | sign
            sigs[i] = sig
        return sigs

    def _hash(self, data: np.array, pos: int) -> float:
        n = len(data)
        # Reshape data for broadcasting
        data = data.reshape(n, 1)
        # Element-wise multiplication of data with its transpose
        product_matrix = data * data.T
        # Flatten and offset the random vector to match the shape
        randv_subset = self.randv[pos:pos + n * n].reshape(n, n)
        # Element-wise multiplication and sum
        val = np.sum(product_matrix * randv_subset)
        return val

    def build_index(self, data: np.ndarray):
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
