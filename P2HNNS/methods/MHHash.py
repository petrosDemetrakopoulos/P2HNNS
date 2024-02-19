from typing import List
from heapq import heappush, heappop
import numpy as np
from tqdm import tqdm
from ..utils.operations import dot_with_start
from .Hash import Hash
from ..utils.Query import Query
from ..utils.IdxVal import IdxVal
from ..utils.HashBucket import HashBucket

class MHHash(Hash):
    """
    Implements a Multilinear Hyperplane (MH) Hashing method for efficient point-to-hyperplane nearest neighbor searches.
    This class leverages a multi-probe LSH (Locality Sensitive Hashing) technique to index high-dimensional data points,
    enabling approximate nearest neighbor queries with reduced computational cost.

    Attributes:
        m (int): The number of hash functions to be concatenated to form a single bucket key.
        l (int): The number of hash tables to be used.
        M (int): The dimensionality of the hyperplane in each hash function.
        randv (np.array): A randomly generated array of vectors used in hashing.
        buckets (HashBucket): A collection of hash buckets for storing indexed data points.

    Parameters for initialization:
        dimension (int): The dimensionality of the input data points.
        m (int): The number of hash functions per hash table.
        l (int): The number of hash tables.
        M (int): The projection dimension for each hash function.
        n (int): The expected size of the dataset, used to determine the range of hash codes.

    Methods:
        hash_data(data: np.array) -> np.array: Hashes the input data into `l` hash signatures, one for each hash table.

        hash_query(query: np.array) -> np.array: Similar to `hash_data`, but hashes a query for searching the index.
    
        build_index(data): Indexes the provided dataset by hashing and storing the data points in the hash buckets.

        nns(param: Query) -> List[IdxVal]: Performs a nearest neighbor search for the given query,
                                           returning the closest points based on the query parameters.
    """
    def __init__(self, dimension: int, m: int, l: int, M: int, n: int):
        self.m = m
        self.l = l
        self.M = M
        size = m * l * M * dimension
        self.randv = np.array([np.random.normal(0.0, 1.0) for _ in range(size)])
        self.buckets = HashBucket(n,l)

    def hash_data(self, data: np.array) -> np.array:
        """
        Hashes the input data using the MH hashing method to generate bucket keys.

        Paramters:
            data (np.array): A single data point to hash.

        Returns:
            np.array: An array of integers representing the hash signatures for the data point across `l` hash tables.
        """
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
        """
        Hashes the query data using the MH hashing method, with a modification for query purposes to generate bucket keys.

        Paramters:
            query (np.array): A single query point to hash.

        Returns:
            np.array: An array of integers representing the hash signatures for the query point across `l` hash tables.
        """
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
        """
        Internal method to compute the hash value of a query/data point at a specific position
        using the stored random vectors.

        Paramters:
            query (np.array): The query or data point to be hashed.
            pos (int): The starting position in the `randv` array for hashing.

        Returns:
            float: The computed hash value.
        """
        val = 1.0
        for k in range(self.M):
            val *= dot_with_start(query, pos + k * len(query), self.randv)
        return val

    def build_index(self, data: np.array):
        """
        Builds the index by hashing all data points and inserting them into the appropriate hash buckets.

        Paramters:
            data (np.array): An array of data points to be indexed.
        
        Returns:
            None
        """
        n = len(data)
        print("-- Building index... --")
        for i in tqdm(range(n)):
            sig = self.hash_data(data[i])
            self.buckets.insert(i, sig)

    def nns(self, param: Query) -> List[IdxVal]:
        """
        Performs a nearest neighbor search for the given query.

        Paramters:
            param (Query): A `Query` object containing the query hyperplane, distance function, and other query parameters.

        Returns:
            List[IdxVal]: A sorted list of `IdxVal` objects representing the nearest neighbors and their distances.
        """
        assert self.buckets.is_empty() is False, "Index not created yet. You need to call create_index() before using nns() to query the index"
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
