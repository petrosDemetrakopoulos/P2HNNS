from typing import List
import numpy as np
from .IdxVal import IdxVal
from .operations import dot

class SearchPosition:
    """
    Represents the positions within hash tables for a query.

    Attributes:
        query_val (np.array): The hash value of the query.
        left_pos (np.array): The left boundary positions in the hash tables for the query.
        right_pos (np.array): The right boundary positions in the hash tables for the query.
    """
    def __init__(self, query_val: np.array, left_pos: np.array, right_pos: np.array):
        self.query_val = query_val
        self.left_pos = left_pos
        self.right_pos = right_pos

class CountParam:
    """
    Encapsulates parameters for the counting process in RQALSH.

    Attributes:
        n (int): The number of data points.
        m (int): The number of hash tables.
        freq (List[int]): A frequency list to track the number of occurrences for each data point.
        rangeflag (List[bool]): Flags to indicate if a data point is within the search range in each hash table.
        bucketflag (List[bool]): Flags to indicate if a data point is within the current bucket being scanned in each hash table.
        checked (List[bool]): Flags to indicate if a data point has been checked.
        rangeval (float): The value representing the search range.
        bucket (int): The number of buckets.
        radius (float): The search radius.
        width (float): The bucket width.
        cands (List): A list of candidate data points.
    """
    def __init__(self, n: int, m: int, radius: float, width: float):
        self.m = m
        # separation frequency for n data points
        self.freq = [0] * n
        # range flag for m hash tables
        self.rangeflag = [True] * m
        # bucket flag for m hash tables
        self.bucketflag = [False] * m
        self.checked = [False] *n
        # number of search range flag
        self.rangeval = 0
        # number of bucket
        self.bucket = 0
        # search radius
        self.radius = radius
        # bucket width
        self.width = width
        # candidate list
        self.cands = []


class RQALSH:
    """
    Implements the Reverse Query-Aware Locality Sensitive Hashing algorithm.

    Attributes:
        n (int): The number of data points.
        dim (int): The dimensionality of the data points.
        m (int): The number of hash tables.
        index (np.array): The indices of the data points.
        scan_size (int): The number of points to scan in each hash table.
        check_error (float): The error tolerance for checking candidates.
        a (np.array): The generated hash functions.
        tables (List): The hash tables storing data points and their hash values.

    Methods:
        calc_hash_value, calc_hash_value_1, calc_hash_value_2: Calculate hash values for data points.
        get_search_position: Determine the search positions for a given query in the hash tables.
        find_radius: Find the search radius based on the query and the data distribution.
        dynamic_separation_counting: Dynamically count and identify candidate data points close to the query.
        fns: Find the nearest neighbors for a given query.
    """
    def __init__(self, n:int , dim: int, m: int, index: np.array, norm: np.array, data:List[List[IdxVal]], scan_size=1600, check_error=10e-6):
        self.n = n
        self.dim = dim
        self.m = m
        self.index = index
        self.scan_size = scan_size
        self.check_error = check_error

        # generate hash functions
        self.a = np.random.normal(0,1,m* dim)

        # allocate space for tables
        self.tables = [None] * (m * n)

        for i in range(n):
            idx = index[i]
            for j in range(m):
                w = data[idx]
                val = self.calc_hash_value_1(len(w),j, norm[idx], w)
                self.tables[j * n + i] = IdxVal(i, val)

        # sort hash tables in ascending order of hash values
        for i in range(m):
            start = i * n
            end = start + n
            self.tables[start:end] = sorted(self.tables[start:end], key=lambda x: x.value)

    def calc_hash_value(self, tid: int, data: np.array) -> float:
        return dot(data,self.a, tid * self.dim)

    def calc_hash_value_1(self, d: int, tid: int, last: float, data: List[IdxVal]) -> float:
        start = tid * self.dim
        val = 0.0
        for i in range(d):
            idx = data[i].idx
            val += self.a[start + idx] * data[i].value
        return val + self.a[start + self.dim - 1] * last

    def calc_hash_value_2(self, d: int, tid: int, data: List[IdxVal]) -> float:
        start = tid * self.dim
        val = 0.0
        for i in range(d):
            idx = data[i].idx
            val += self.a[start + idx] * data[i].value
        return val

    def get_search_position(self,sample_dim:int , query: List[IdxVal]) -> SearchPosition:
        """
        Determines the initial search positions for a given query in the hash tables.

        Parameters:
            sample_dim (int): The dimensionality of the sample data points.
            query (List[IdxVal]): The query data points.

        Returns:
            SearchPosition: An object representing the positions within hash tables for the given query.
        """
        query_val = np.zeros(self.m)
        left_pos = np.zeros(self.m, dtype=int)
        right_pos = np.zeros(self.m, dtype=int)
        for i in range(self.m):
            query_val[i] = self.calc_hash_value_1(sample_dim, i, 0.0, query)
            left_pos[i] = 0
            right_pos[i] = self.n - 1

        return SearchPosition(query_val, left_pos, right_pos)

    def find_radius(self, w: float, position: SearchPosition) -> float:
        # find projected distance closest to the query in each hash tables
        arr = np.zeros(self.m * 2)
        num = 0
        for i in range(self.m):
            lpos = position.left_pos[i]
            rpos = position.right_pos[i]
            if lpos < rpos:
                query_val = position.query_val[i]
                arr[num] = np.abs(self.tables[i * self.n + lpos].value - query_val)
                arr[num + 1] = np.abs(self.tables[i * self.n + rpos].value - query_val)
                num += 2

        arr = np.sort(arr[:num])

        # find the median distance and return the new radius
        if num % 2 == 0:
            dist = (arr[num // 2 - 1] + arr[num // 2]) / 2.0
        else:
            dist = arr[num // 2]

        kappa = int(np.ceil(np.log(2.0 * dist / w) / np.log(2.0)))

        return 2.0 ** kappa

    def dynamic_separation_counting(self,l: int, limit: int, R: float, position: SearchPosition):
        num_range = 0
        num_bucket = 0
        cnt = 0
        lpos = 0
        rpos = 0
        idx = 0

        w = 1.0
        radius = self.find_radius(w, position)
        width = radius * w / 2.0
        range_val = 0.0 if R < self.check_error else R * w / 2.0

        param = CountParam(self.n, self.m, radius, width)
        while True:
            num_bucket = 0
            param.bucketflag = [True] * self.m

            while num_bucket < self.m and num_range < self.m:
                for j in range(self.m):
                    if not param.bucketflag[j]:
                        continue
                    start = j * self.n
                    q_v = position.query_val[j]
                    # step 2.1: scan the left part of the bucket
                    cnt = 0
                    lpos, rpos = position.left_pos[j], position.right_pos[j]
                    while cnt < self.scan_size:
                        ldist = float('-inf')
                        if lpos < rpos:
                            ldist = abs(q_v - self.tables[start+lpos].value)
                        else:
                            break
                        if ldist < param.width or ldist < range_val:
                            break
                        idx = self.tables[start+lpos].idx
                        param.freq[idx] += 1
                        if param.freq[idx] >= l and not param.checked[idx]:
                            param.checked[idx] = True
                            if self.index:
                                param.cands.append(self.index[idx])
                            else:
                                param.cands.append(idx)
                            if len(param.cands) >= limit:
                                break
                        lpos += 1
                        cnt += 1
                    if len(param.cands) >= limit:
                        break
                    position.left_pos[j] = lpos

                    #step 2.2 scan the right part of the bucket
                    cnt = 0
                    while cnt < self.scan_size:
                        rdist = float('-inf')
                        if lpos < rpos:
                            rdist = abs(q_v - self.tables[start+rpos].value)
                        else:
                            break
                        if rdist < param.width or rdist < range_val:
                            break
                        idx = self.tables[start+rpos].idx
                        param.freq[idx] += 1
                        if param.freq[idx] >= l and not param.checked[idx]:
                            param.checked[idx] = True
                            if self.index:
                                param.cands.append(self.index[idx])
                            else:
                                param.cands.append(idx)
                            if len(param.cands) >= limit:
                                break
                        rpos -= 1
                        cnt += 1
                    if len(param.cands) >= limit:
                        break
                    position.right_pos[j] = rpos

                    # step 2.3 check whether this bucket is finished scanned
                    if lpos >= rpos or (ldist < param.width and rdist < param.width):
                        if param.bucketflag[j]:
                            param.bucketflag[j] = False
                            num_bucket += 1
                    if lpos >= rpos or (ldist < range_val and rdist < range_val):
                        if param.bucketflag[j]:
                            param.bucketflag[j] = False
                            num_bucket += 1
                        if param.rangeflag[j]:
                            param.rangeflag[j] = False
                            num_range += 1
                    if num_bucket >= self.m or num_range >= self.m:
                        break
                if num_bucket >= self.m or num_range >= self.m:
                    break
                if len(param.cands) >= limit:
                    break
            # step 3
            if num_range >= self.m or len(param.cands) >= limit:
                break
            # step 4, update radius and width
            param.radius = param.radius / 2.0
            param.width = param.radius * w / 2.0
        return param.cands


    def fns(self, l: int, limit: int, R: float, sampledim: int, query: List[IdxVal]):
        # simply check all data if #candidates is equal to the cardinality
        if self.n <= limit:
            cands = []
            for i in range(self.n):
                if self.index is not None and -1 < self.index[i]:
                    cands.append(self.index[i])
                else:
                    cands.append(i)
            return cands

        # dynamic separation counting
        pos = self.get_search_position(sampledim, query)
        return self.dynamic_separation_counting(l, limit, R, pos)
