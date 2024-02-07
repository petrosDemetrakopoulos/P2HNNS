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
    """
    A class used to represent the signature of a data point after projection.
    
    Attributes:
        value (np.array): The projected values of the data point.
        norm (float): The norm of the signature vector.
    """
    def __init__(self, value, sig_norm):
        self.value = value
        self.norm = sig_norm

class NHHash(Hash):
    """
    Implements the Nearest Hyperplane (NH) Hashing for approximate nearest neighbor search.
    This class extends a general Hash class structure, focusing on generating hash signatures by projecting
    data points onto a set of hyperplanes, and then using these signatures for efficient similarity search.
    
    Attributes:
        m (int): The number of hash functions (hyperplanes) used for hashing.
        w (float): The width of the hash bins.
        nhdim (int): The dimensionality of the space after projecting to the NH space.
        sampler (RandomSampler): Sampler used to select a subset of dimensions for projection.
        called (int): Counter for the number of times sampling has occurred.
        proja (np.array): Coefficients for linear projection in NH space.
        projb (np.array): Bias terms for linear projection in NH space.
        bucketerp (SortedLCCS): Data structure for storing and searching hashed data.
    """
    def __init__(self, d: int, m: int, s: int, w: float):
        """
        Initializes the NHHash object with the given parameters.
        
        Parameters:
            d (int): The original dimensionality of the data.
            m (int): The number of hash functions to use.
            s (int): The size of the sample to take from the original data dimensions.
            w (float): The width of the hash bins.
        """
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
        """
        Projects a data point into the NH space and returns its signature.
        
        Parameters:
            data (np.array): The data point to be hashed.
        
        Returns:
            Signature: The signature of the data point in the NH space.
        """
        projs = np.zeros(self.m)
        sample = self.sampler.sampling(data)

        for i in range(self.m):
            start = i * self.nhdim
            val = np.sum([self.proja[start + w.idx] * w.value for w in sample])
            projs[i] = val
        self.called += 1
        return Signature(projs, self.norm(sample))

    def norm(self, idxvals: List[IdxVal]) -> float:
        """
        Calculates the norm of a vector represented by a list of IdxVal objects.
        
        Parameters:
            idxvals (List[IdxVal]): The vector represented as a list of IdxVal objects.
        
        Returns:
            float: The norm of the vector.
        """
        return np.sum([w.value**2 for w in idxvals])

    def hash_data(self, data: np.ndarray) -> np.array:
        """
        Hashes an array of data points into NH space.
        
        Parameters:
            data (np.ndarray): The data points to be hashed.
        
        Returns:
            np.array: The array of hashed signatures.
        """
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
        """
        Hashes a single query point for searching in the NH space.
        
        Parameters:
            query (np.array): The query point to be hashed.
        
        Returns:
            np.array: The hashed signature of the query.
        """
        sig = np.zeros(self.m, dtype=int)
        print("-- Query sampling... --")
        sample = self.sampler.sampling(query)
        for i in range(self.m):
            val = np.sum([self.proja[i * self.nhdim + idx.idx] * idx.value for idx in sample])
            sig[i] = int((val + self.projb[i]) / self.w)
        return sig

    def build_index(self, data: np.ndarray):
        """
        Builds the index from the provided data points by hashing them into the NH space.
        
        Parameters:
            data (np.ndarray): The data points to index.
        """
        n = len(data)
        sigs = self.hash_data(data)
        arr = np.zeros((n, self.m), dtype=int)

        for i in range(n):
            sig = sigs[i]
            arr[i, :] = sig

        # sort arr data index by value per dim
        self.bucketerp = SortedLCCS(1, arr)

    def nns(self, param: Query) -> List[IdxVal]:
        """
        Performs an approximate nearest neighbor search for a given query.
        
        Parameters:
            param (Query): The query object containing the query hyperplane,
                distance function, and the number of nearest neighbors to find.

        Returns:
            List[IdxVal]: The list of nearest neighbors and their distances to the query hyperplane.
        """
        assert self.bucketerp is not None, "Index not created yet. You need to call create_index() before using nns() to query the index"
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
        """
        Evaluates a candidate point for inclusion in the nearest neighbor priority queue.
        
        Parameters:
            key (int): The index of the candidate data point in the dataset.
            query: The query point.
            data: The dataset.
            dist_fun (Dist): The distance function to use for comparison.
            queue (PriorityQueue): The priority queue to store nearest neighbors.
            top (int): The number of nearest neighbors to find.
        """
        dist = dist_fun.distance(query, data[key])
        queue.put(IdxVal(key, dist))
        if top < queue.qsize():
            queue.get()
