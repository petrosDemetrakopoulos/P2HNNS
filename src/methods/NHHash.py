from queue import PriorityQueue
from typing import List
import numpy as np
from tqdm import tqdm
from ..RandomSampler import RandomSampler
from ..Query import Query
from ..IdxVal import IdxVal
from .Hash import Hash
from ..distance_functions import Dist
from ..SortedLCCS import SortedLCCS

class Signature:
    def __init__(self, value, sig_norm):
        self.value = value
        self.norm = sig_norm

class NHHash(Hash):
    """
    Nearest Hyperplane(NH) Hash.
    """
    def __init__(self, d: int, m: int, s: int, w: float):
        self.m = m
        self.w = w
        self.nhdim = d * (d + 1) // 2 + 1
        self.sampler = RandomSampler(d, s)
        self.called = 0

        projsize = m * self.nhdim
        self.proja = np.array([np.random.normal(0.0, 1.0) for x in range(projsize)])
        self.projb = np.array([np.random.normal(0.0, 1.0) for x in range(m)])

        self.bucketerp = None

    def sampling_signature(self, data: np.array) -> Signature:
        projs = np.zeros(self.m)
        sample = self.sampler.sampling(data)

        for i in range(self.m):
            start = i * self.nhdim
            val = np.sum([self.proja[start + w.idx] * w.value for w in sample])
            projs[i] = val
        self.called += 1
        return Signature(projs, self.norm(sample))

    def norm(self, idxvals: List[IdxVal]) -> float:
        return np.sum([w.value**2 for w in idxvals])

    def hash_data(self, data: np.ndarray) -> np.array:
        n = len(data)
        m = np.finfo(np.float64).min
        sample = [self.sampling_signature(d) for d in data]
        m = max([sig.norm for sig in sample], default=m)

        sigs = np.zeros((n, self.m), dtype=int)
        print("-- Building index... --")
        for i in tqdm(range(n)):
            sampled = sample[i]
            lastcoord = np.sqrt(m - sampled.norm)
            proj = sampled.value
            for j in range(self.m):
                val = proj[j] + lastcoord * self.proja[(j + 1) * self.nhdim - 1]
                v = (val + self.projb[j]) / self.w
                sigs[i, j] = int(v)
        return sigs

    def hash_query(self, query: np.array) -> np.array:
        sig = np.zeros(self.m, dtype=int)
        print("-- Query sampling... --")
        sample = self.sampler.sampling(query)
        for i in range(self.m):
            val = np.sum([self.proja[i * self.nhdim + idx.idx] * idx.value for idx in sample])
            sig[i] = int((val + self.projb[i]) / self.w)
        return sig

    def build_index(self, data: np.ndarray):
        n = len(data)
        sigs = self.hash_data(data)
        arr = np.zeros((n, self.m), dtype=int)

        for i in range(n):
            sig = sigs[i]
            arr[i, :] = sig

        # sort arr data index by value per dim
        self.bucketerp = SortedLCCS(1, arr)

    def nns(self, param: Query) -> List[IdxVal]:
        data = param.data
        query = param.query
        top = param.top
        dist_fun = param.dist

        queue = PriorityQueue()

        sigs = self.hash_query(query)
        step = (top + self.m - 1) // self.m

        # Binary search signature from sorted index.
        # The more similar the signatures, the better the search results.
        self.bucketerp.search(step, sigs, lambda key: self.accept(key, query, data, dist_fun, queue, top))

        result = []
        while not queue.empty():
            w = queue.get()
            result.append(w)

        result.reverse()
        return result

    def accept(self, key: int, query, data, dist_fun: Dist, queue, top):
        dist = dist_fun.distance(query, data[key])
        queue.put(IdxVal(key, dist))
        if top < queue.qsize():
            queue.get()
