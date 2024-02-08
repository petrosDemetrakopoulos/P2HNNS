from typing import List
import heapq
import numpy as np
from tqdm import tqdm
from ..utils.RandomSampler import RandomSampler
from ..utils.IdxVal import IdxVal
from ..utils.FHQuery import FHQuery
from .Hash import Hash
from ..utils.RQALSH import RQALSH

class Transform:
    """
    A class representing the transformation of data for nearest neighbor search using the Furthest Hyperplane method.

    Attributes:
        norm (np.array): An array holding the L2 norms of the data points.
        centroid (np.array): The centroid of the transformed data points in the hyperplane space.
        samples (List[List[IdxVal]]): Sampled representation of the data points in the hyperplane space.
        M (float): The maximum norm among all data points, used for normalization.
        dist (List[IdxVal]): Sorted distances of the data points from a reference point in the hyperplane space.
    """
    def __init__(self, norm: np.array, centroid: np.array, samples: List[List[IdxVal]], M: float, dist: List[IdxVal]):
        self.norm = norm
        self.centroid = centroid
        self.samples = samples
        self.M = M
        self.dist = dist

class FHHash(Hash):
    """
    Implements the Furthest Hyperplane (FH) hashing method for efficient nearest neighbor search in high-dimensional data.

    Attributes:
        hashs (List): Container for RQALSH instances, each corresponding to a block of the dataset.
        fhdim (int): The dimensionality of the data after transformation to the furthest hyperplane space.
        sampler (RandomSampler): An instance of RandomSampler for sampling high-dimensional data points.
        M (float): Maximum norm value among all data points, used in normalization.
        b (float): The base radius for dividing the dataset into blocks.
        m (int): The number of hash functions to be used in RQALSH.
        max_blocks (int): Maximum number of blocks into which the dataset is divided.

    Methods:
        

    Parameters for initalization:
        d (int): Original dimensionality of the data.
        s (int): Number of samples to be taken from each data point.
        b (float): Base radius parameter for dataset division.
        m (int): Number of hash functions for RQALSH.
        max_blocks (int): Maximum number of data blocks.
    """
    def __init__(self, d, s, b,m, max_blocks):
        """
        Initializes the FHHash instance.

        Parameters:
            d (int): Original dimensionality of the data.
            s (int): Number of samples to be taken from each data point.
            b (float): Base radius parameter for dataset division.
            m (int): Number of hash functions for RQALSH.
            max_blocks (int): Maximum number of data blocks.
        """
        if b > 1.0:
            raise ValueError("b must be less than or equal to 1.0")
        self.hashs = []
        self.fhdim = d * (d + 1) // 2 + 1
        self.sampler = RandomSampler(d, s)
        self.M = 0.0
        self.b = b
        self.m = m
        self.max_blocks = max_blocks

    def hash_data(self, data: np.ndarray) -> Transform:
        """
        Transforms and hashes the input data into the furthest hyperplane space for indexing.

        Parameters:
            data (np.ndarray): The high-dimensional data to be indexed.

        Returns:
            Transform: A Transform object containing the transformation results.
        """
        n = len(data)
        norm = np.zeros(n)
        centroid = np.zeros(self.fhdim)
        samples = [None] * n

        M = float('-inf')
        for i in range(n):
            sample = self.sampler.sampling(data[i])
            l2 = 0.0
            for w in sample:
                l2 += w.value ** 2
                centroid[w.idx] += w.value
            norm[i] = l2
            samples[i] = sample
            M = max(M, norm[i])

        l2centroid = 0.0
        for i in range(self.fhdim - 1):
            centroid[i] /= n
            l2centroid += centroid[i] ** 2
        last = sum(np.sqrt(M - norm[i]) for i in range(n)) / n
        centroid[self.fhdim - 1] = last
        l2centroid += last ** 2

        arr = [None] * n
        for i in range(n):
            val = self.calc_transform_dist(self.fhdim, np.sqrt(M - norm[i]), l2centroid, samples[i], centroid)
            arr[i] = IdxVal(i, val)
        arr.sort(key=lambda x: x.value)
        return Transform(norm, centroid, samples, M, arr)

    def calc_transform_dist(self, fhdim: int, last: float, l2centroid: float, sample: List[IdxVal], centroid: np.array):
        """
        Calculates the transformed distance between a data point and the centroid in the hyperplane space.

        Parameters:
            fhdim (int): Dimensionality of the furthest hyperplane space.
            last (float): The component of the centroid corresponding to the added dimension.
            l2centroid (float): The L2 norm of the centroid in the hyperplane space.
            sample (List[IdxVal]): The sampled representation of a data point.
            centroid (np.array): The centroid of the data in the hyperplane space.

        Returns:
            float: The transformed distance.
        """
        dist = l2centroid

        for i in range(len(sample)):
            idx = sample[i].idx
            tmp = centroid[idx]
            diff = sample[i].value - tmp
            dist -= tmp ** 2
            dist += diff ** 2

        tmp = centroid[fhdim - 1]
        dist -= tmp ** 2
        dist += (last - tmp) ** 2
        return np.sqrt(dist)

    def build_index(self, data: np.ndarray):
        """
        Builds the indexing structure for the given data using the Furthest Hyperplane hashing method.

        Parameters:
            data (np.ndarray): The high-dimensional data to be indexed.
        """
        n = len(data)
        norm = self.hash_data(data)
        self.M = norm.M

        dists = norm.dist
        index = []

        for i in range(n):
            index.append(dists[i].idx)

        #  divide datasets into blocks and build hash tables for each block
        start = 0
        pbar = tqdm(total=n)
        print("-- Building index... --")
        while start < n:
            minradius = self.b * dists[start].value
            block = start
            cnt = 0

            while block < n and minradius < dists[block].value:
                block += 1
                if self.max_blocks <= (cnt+1):
                    pbar.update(n)
                    break
                cnt += 1

            hashidx = index[start:]

            rqa = RQALSH(cnt,self.fhdim, self.m, hashidx, norm.norm, norm.samples)
            self.hashs.append(rqa)
            start += cnt
            pbar.update(cnt)
        pbar.close()
        assert start == n

    def query(self, query):
        """
        Samples a query point for nearest neighbor search.

        Parameters:
            query: The query point in the original high-dimensional space.

        Returns:
            The sampled representation of the query point.
        """
        return self.sampler.sampling(query)

    def get_sample_query(self,query: np.array) -> List[IdxVal]:
        """
        Transforms a query point into the furthest hyperplane space for querying the index.

        Parameters:
            query (np.array): The query point in the original high-dimensional space.

        Returns:
            List[IdxVal]: The transformed and sampled representation of the query point.
        """
        # calculate sampleQuery with query transformation
        sample = self.query(query)
        norm = self.norm(sample)

        lamda = np.sqrt(self.M / norm)

        for i in range(len(sample)):
            w = sample[i]
            sample[i] = IdxVal(w.idx, w.value * lamda)
        return sample

    def nns(self, param: FHQuery) -> List[IdxVal]:
        """
        Performs the nearest neighbor search for a given query in the indexed data.

        Parameters:
            param (FHQuery): An object containing the query details and search parameters.

        Returns:
            List[IdxVal]: The nearest neighbors to the query point.
        """
        data, query = param.data, param.query
        l, top = param.l, param.top

        sample = self.get_sample_query(query)

        # point-to-hyperplane NNS
        limit = param.limit + top - 1
        fixval = 2 * self.M
        queue = []
        distance_func = param.dist
        print("-- Searching... --")
        for crn_hash in tqdm(self.hashs):
            # check candidates returned by rqalsh
            kfndist = -1.0
            if top <= len(queue):
                kdist = queue[0][1]
                kfndist = np.sqrt(fixval - 2 * kdist * kdist)

            # scan range search by distance between query and data
            res = crn_hash.fns(l, limit, kfndist, len(sample), sample)
            for idx in res:
                dist = distance_func.distance(query, data[idx])
                heapq.heappush(queue, (-dist, IdxVal(idx, dist)))
                if top < len(queue):
                    heapq.heappop(queue)

            size = len(res)
            limit -= size
            if limit <= 0:
                break

        result = []
        while queue:
            w = heapq.heappop(queue)[1]
            result.insert(0, w)

        return result

    def norm(self, idxvals: List[IdxVal]) -> float:
        """
        Calculates the L2-norm squared of the feature vector represented by IdxVal objects.

        Parameters:
            idxvals (List[IdxVal]): The feature vector represented as a list of IdxVal objects.

        Returns:
            float: The L2-norm squared of the feature vector.
        """
        norm = 0.0
        for w in idxvals:
            norm += w.value ** 2
        return norm

    def hash_query(self, query: np.array):
        pass
