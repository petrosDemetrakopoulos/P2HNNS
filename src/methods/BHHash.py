from typing import List
from heapq import heappush, heappop
from tqdm import tqdm
import numpy as np
from .Hash import Hash
from ..Query import Query
from ..IdxVal import IdxVal
from ..HashBucket import HashBucket

class BHHash(Hash):
    """
    Implements the Bilinear Hyperplane (BH) hashing method for efficient point-to-hyperplane nearest neighbor searches.
    This class generates hash codes by projecting data points onto random hyperplanes generated from
    Gaussian distributions. The hash code for a data point is determined by the sign of the dot product
    between the data point and each pair of random vectors, allowing for the capture of bilinear interactions.

    Attributes:
        m (int): The number of hyperplanes used for each hash function.
        l (int): The number of hash functions (or hash tables).
        randu (np.array): A numpy array of random vectors from a Gaussian distribution, first part of bilinear projection.
        randv (np.array): A numpy array of random vectors from a Gaussian distribution, second part of bilinear projection.

    Methods:
        data(data: np.array) -> np.array: Hashes the input data array into binary hash codes.
        query(query: np.array) -> np.array: Hashes the query data array into binary hash codes.

    Parameters for initialization:
        d (int): The dimensionality of the input data vectors.
        m (int): The number of hyperplanes for each hash function.
        l (int): The number of hash functions to generate.
    """
    def __init__(self, d: int, m: int, l: int):
        """
        Initializes the BHHash instance with given dimensions and generates two sets of random vectors (randu and randv)
        for projecting the data points onto the bilinear hyperplanes.

        Parameters:
            d (int): The dimensionality of the input data.
            m (int): The number of hyperplanes per hash function.
            l (int): The number of hash functions.
        """
        self.buckets = HashBucket(d,l)
        self.m = m
        self.l = l
        size = m * l * d
        self.randu = np.random.normal(0.0, 1.0, size)
        self.randv = np.random.normal(0.0, 1.0, size)

    def hash_data(self, data: np.array) -> np.array:
        """
        Generates binary hash codes for the input data using the bilinear hyperplane approach.
        The method projects the data onto the generated random hyperplanes and computes the sign
        of the bilinear product for each projection to form the hash code.

        Parameters:
            data (np.array): The input data array to be hashed.

        Returns:
            np.array: An array of integers representing the binary hash codes of the input data.
        """
        assert len(self.randu) == self.m * self.l * len(data)
        sigs = np.zeros(self.l, dtype=int)
        for i in range(self.l):
            sig = 0
            for j in range(self.m):
                pos = (i * self.m + j) * len(data)
                val1 = np.dot(data, self.randu[pos:pos+len(data)])
                val2 = np.dot(data, self.randv[pos:pos+len(data)])
                sign = int(0 < val1 * val2)
                sig = (sig << 1) | sign
            sigs[i] = sig
        return sigs

    def hash_query(self, query) -> np.array:
        """
        Generates binary hash codes for a query, similar to the `data` method but intended for query vectors.
        This allows the hashing process to be applied specifically to query points, potentially with different processing.

        Parameters:
            query (np.array): The query data array to be hashed.

        Returns:
            np.array: An array of integers representing the binary hash codes of the query data.
        """
        assert len(self.randu) == self.m * self.l * len(query)
        sigs = np.zeros(self.l, dtype=int)
        for i in range(self.l):
            sig = 0
            for j in range(self.m):
                pos = (i * self.m + j) * len(query)
                val1 = np.dot(query, self.randu[pos:pos+len(query)])
                val2 = np.dot(query, self.randv[pos:pos+len(query)])
                sign = int(0 >= val1 * val2)
                sig = (sig << 1) | sign
            sigs[i] = sig
        return sigs

    def build_index(self, data: np.ndarray):
        """
        Inserts the given dataset into the hash buckets by generating binary hash codes for each data point and
        storing them in the appropriate buckets.

        Parameters:
            data (np.ndarray): The dataset to be inserted into the hash table. Each row represents a data point.

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
        Performs a nearest neighbor search (NNS) for the given query hyperplane, using the binary hash codes to efficiently
        narrow down the search space. This method retrieves a list of the nearest neighbors sorted by their distance
        from the query hyperplane, up to a specified limit.

        The method uses a max heap to maintain the top closest points encountered during the search. The search space
        is limited by traversing only those buckets that match the query's hash code up to a specified Hamming distance.

        Parameters:
        - param (Query): A Query object containing the query hyperplane, the number of nearest neighbors to return (top),
                        and the search limit (limit). The Query object also includes the dataset to search against,
                        although not directly used here.

        Returns:
        - List[IdxVal]: A list of IdxVal objects, each containing the index of a data point in the dataset and its
                        distance to the query. The list is sorted by distance, ascending.
        """
        assert self.buckets.is_empty() is False, "Index not created yet. You need to call create_index() before using nns() to query the index"
        data = param.data
        query = param.query
        top = param.top
        limit = param.limit
        data_dim = data.shape[1]
        sig = self.hash_query(query)
        heap = []

        def accept(key: int):
            nonlocal heap, query, data, top, data_dim
            dist = np.abs(np.dot(query[:data_dim], data[key])) #dist_fun.distance(query, data[key])
            heappush(heap, IdxVal(key, dist))
            if top < len(heap):
                heappop(heap)

        self.buckets.search(sig, limit, accept)

        result = sorted(heap, key=lambda x: x.value)
        return result
