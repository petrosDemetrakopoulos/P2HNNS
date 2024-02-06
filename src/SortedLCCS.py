from functools import cmp_to_key
from typing import List, Callable, Dict
from collections import defaultdict
import numpy as np

# helper classes
class Locs:
    def __init__(self, idxes: List[int], lowlens: List[int], highlens: List[int]):
        self.idxes = idxes
        self.lowlens = lowlens
        self.highlens = highlens

class Loc:
    def __init__(self, idx: int, lowlen: int, highlen: int):
        self.idx = idx
        self.lowlen = lowlen
        self.highlen = highlen

class CmpLoc:
    def __init__(self, walked: int, cmp: int):
        self.walked = walked
        self.cmp = cmp


class SortedLCCS:
    """
    Sorted Longest Circular Co-Substring (LCCS)
    """
    def __init__(self, step: int, data: np.ndarray, scan_size:int=4):
        self.SCAN_SIZE = scan_size 
        self.n = data.shape[0]
        self.dim = data.shape[1]
        self.step = step
        self.searchdim = ((self.dim - 1) // self.step) + 1
        self.data = data

        self.sortedidx = self.get_sort_idx(self.dim, self.n)
        self.nextidx = self.get_next_link(self.step)

    def compare_dim(self,xs: np.array, ys: np.array, start: int, walked: int) -> CmpLoc:
        size = self.dim - walked
        for i in range(size):
            idx = (start + walked + i) % self.dim
            x = xs[idx]
            y = ys[idx]
            if x != y:
                return CmpLoc(i + walked, int(x > y) - int(x < y))
        return CmpLoc(self.dim, 0)

    def compare_dim_indices(self,idx1: int, idx2: int, start: int, walked: int) -> CmpLoc:
        xp = self.data[idx1]
        yp = self.data[idx2]
        return self.compare_dim(xp, yp, start, walked)

    def get_sort_idx(self,dim:int, n:int) -> List[List[int]]:
        sorted_idx = [[i for i in range(n)] for _ in range(dim)]
        for d in range(dim):
            cmp = cmp_to_key(lambda idx1, idx2, d=d: self.compare_dim_indices(idx1, idx2, d, 0).cmp)
            sorted_idx[d].sort(key=cmp)
        return sorted_idx

    def sorted_idx(self,dim: int, idx: int) -> int:
        return self.sortedidx[dim][idx]

    def get_next_link(self,step: int) -> List[List[int]]:
        next_link = [[0] * self.n for _ in range(self.dim)]
        for d in range(self.dim - 1, -1, -1):
            next_dim = (d + step) % self.dim
            next_order = [0] * self.n
            for i in range(self.n):
                next_order[self.sorted_idx(next_dim, i)] = i

            for i in range(self.n):
                idx = self.sorted_idx(d, i)
                next_link[d][i] = next_order[idx]

        return next_link

    def get_data(self,query_dim: int, idx: int) -> np.array:
        row = self.sortedidx[query_dim][idx]
        return self.data[row]

    def scan_loc(self,query: np.array, query_dim: int, low: int, low_len: int, high: int, high_len: int) -> Loc:
        last_dim = low_len
        min_dim = min(low_len, high_len)

        for i in range(low + 1, high):
            dpi = self.get_data(query_dim, i)
            loc = self.compare_dim(query, dpi, query_dim, min_dim)

            if loc.cmp == -1:
                return Loc(i - 1, last_dim, loc.walked)

            last_dim = loc.walked

        # Reach the end
        return Loc(high - 1, last_dim, high_len)

    # Define the equivalent of BiIntFunction
    def cmp_function(self,i, prev, query_dim, query):
        dpa = self.get_data(query_dim, i)
        return self.compare_dim(query, dpa, query_dim, prev)

    def binary_search_loc(self,query: np.array, query_dim: int, low: int, low_len: int, high: int, high_len: int) -> Loc:
        next_low = low
        next_low_len = low_len
        next_high = high
        next_high_len = high_len

        while next_low < next_high - self.SCAN_SIZE:
            cur_dim = min(next_low_len, next_high_len)

            # Binary search
            mid = (next_low + next_high) // 2
            loc = self.cmp_function(mid, cur_dim, query_dim, query)

            # If query < mid
            if loc.cmp == -1:
                # Mid < query
                next_high = mid
                next_high_len = loc.walked
            else:
                next_low = mid
                next_low_len = loc.walked

        return self.scan_loc(query, query_dim, next_low, next_low_len, next_high, next_high_len)

    def get_loc(self,query: np.array, query_dim: int) -> Loc:
        low = 0
        high = self.n - 1
        datalow = self.get_data(query_dim, low)
        datahigh = self.get_data(query_dim, high)
        low_loc = self.compare_dim(query, datalow, query_dim, 0)
        high_loc = self.compare_dim(query, datahigh, query_dim, 0)

        return self.binary_search_loc(query, query_dim, low, low_loc.walked,high, high_loc.walked)

    def find_matched_locs(self,query: np.array) -> Locs:
        # Binary search
        loc = self.get_loc(query, 0)
        idx, lowlen, highlen = loc.idx, loc.lowlen, loc.highlen

        idxes = [0] * self.dim
        lowlens = [0] * self.dim
        highlens = [0] * self.dim

        # Store the result for multi-probe lsh
        idxes[0] = idx
        lowlens[0] = lowlen
        highlens[0] = highlen

        for i in range(1, self.searchdim):
            d = i * self.step
            lowidx = self.nextidx[d - self.step][idx]
            highidx = self.nextidx[d - self.step][idx + 1]

            # Set lowLen, highLen range within step size
            if lowlen < self.step:
                lowlen = 0
                lowidx = 0
            elif lowlen != self.dim:
                lowlen -= self.step

            if highlen < self.step:
                highlen = 0
                highidx = self.n - 1
            elif highlen != self.dim:
                highlen -= self.step

            found = self.binary_search_loc(query, d, lowidx, lowlen, highidx, highlen)
            # Store the result for multi-probe lsh
            idxes[d] = found.idx
            lowlens[d] = found.lowlen
            highlens[d] = found.highlen

        return Locs(idxes, lowlens, highlens)

    def search(self,scan_step:int, query: np.array, f: Callable) -> Dict:
        return self.candidates_by_scan(scan_step, query, f)

    def candidates_by_scan(self,scan_step: int, query: np.array, f: Callable) -> Dict:
        locs = self.find_matched_locs(query)

        checked = defaultdict(int)

        def check(idx):
            cnt = checked[idx]
            if cnt == 0:
                f(idx)
            checked[idx] = cnt + 1

        def check_loc(cur_idx, d):
            for i in range(cur_idx, -1, -1):
                if cur_idx - i < scan_step:
                    match_idx = self.sorted_idx(d, i)
                    check(match_idx)

            for i in range(cur_idx + 1, self.n):
                if i - cur_idx - 1 < scan_step:
                    match_idx = self.sorted_idx(d, i)
                    check(match_idx)

        for i in range(self.dim):
            check_loc(locs.idxes[i], i)

        return checked
