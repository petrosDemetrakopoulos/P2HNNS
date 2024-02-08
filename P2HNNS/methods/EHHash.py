from typing import List
from heapq import heappush, heappop
from tqdm import tqdm
import numpy as np
from .Hash import Hash
from ..utils.Query import Query
from ..utils.IdxVal import IdxVal
from ..utils.HashBucket import HashBucket

class EHHash(Hash):
    """
    Implements the Embedding Hyperplane (EH) hashing method for efficient point-to-hyperplane nearest neighbour search.
    This class provides functionality to hash data points and queries into binary signatures and perform fast
    nearest neighbour searches using these signatures.

    Attributes:
        buckets (HashBucket): An object to manage the storage and retrieval of hashed data points.
        m (int): The number of hyperplanes used for hashing each layer.
        l (int): The number of layers in the hash function.
        randv (np.array): A numpy array of random vectors used for hashing.

    Parameters for initialization:
        d (int): The dimensionality of the input data.
        m (int): The number of hyperplanes used for hashing in each layer.
        l (int): The number of hash layers used to increase the robustness of the search.

    Methods:
        hash_data(data: np.array) -> np.array: Hashes a given data point into a series of binary signatures
                                               based on the embedding hyperplane method.

        hash_query(query: np.array) -> np.array: Hashes a query into binary signatures,
                                                 analogous to `hash_data` but tailored for query handling.

        build_index(data: np.ndarray): Constructs the hash index for a dataset,
                                       enabling efficient nearest neighbour searches.

        nns(param: Query) -> List[IdxVal]: Performs a nearest neighbour search for a given query using the pre-built index.
    """
    def __init__(self, d: int, m: int, l: int):
        self.buckets = HashBucket(d,l)
        self.m = m
        self.l = l
        size = m * l * d * d
        self.randv = np.random.normal(0.0, 1.0, size)

    def hash_data(self, data: np.array):
        """
        Hashes the input data into binary signatures based on the EH hashing method.

        Parameters:
            data (np.array): The data point to hash, represented as a numpy array.

        Returns:
            np.array: An array of integers representing the binary signatures of the hashed data.
        """
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
        """
        Hashes the query data into binary signatures, similar to `hash_data` but tailored for queries.

        Parameters:
            query (np.array): The query data point to hash, represented as a numpy array.

        Returns:
            np.array: An array of integers representing the binary signatures of the hashed query.
        """
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
        """
        A helper method to compute the hash value of a data point or query at a specified position.

        Parameters:
            data (np.array): The data point or query to hash.
            pos (int): The position in the random vector to start hashing.

        Returns:
            float: The hash value of the data.
        """
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
        """
        Builds the hash index for a given dataset.

        Parameters:
            data (np.ndarray): The dataset to index, where each row is a data point.
        """
        n = len(data)
        print("-- Building index... --")
        for i in tqdm(range(n)):
            sig = self.hash_data(data[i])
            self.buckets.insert(i, sig)

    def nns(self, param: Query) -> List[IdxVal]:
        """
        Performs a nearest neighbour search (NNS) for a given query.

        Parameters:
            param (Query): A Query object containing the query hyperplane, distance function, and search parameters.

        Returns:
            List[IdxVal]: A list of IdxVal objects, where each IdxVal contains the index 
                          of a data point and its distance to the query, sorted by distance.
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
