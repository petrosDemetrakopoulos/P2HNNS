from functools import cmp_to_key
from typing import List, Callable, Dict
from collections import defaultdict
import numpy as np

# helper classes
class Locs:
    """
    A container for storing indices and lengths of locations in a search space.

    Attributes:
        idxes (List[int]): Indices of locations.
        lowlens (List[int]): The lengths at the lower end of the locations.
        highlens (List[int]): The lengths at the higher end of the locations.
    """
    def __init__(self, idxes: List[int], lowlens: List[int], highlens: List[int]):
        self.idxes = idxes
        self.lowlens = lowlens
        self.highlens = highlens

class Loc:
    """
    Represents a single location in the search space.

    Attributes:
        idx (int): The index of the location.
        lowlen (int): The length at the lower end of the location.
        highlen (int): The length at the higher end of the location.
    """
    def __init__(self, idx: int, lowlen: int, highlen: int):
        self.idx = idx
        self.lowlen = lowlen
        self.highlen = highlen

class CmpLoc:
    """
    Holds the result of a comparison and the distance walked during the comparison.

    Attributes:
        walked (int): The distance walked in the comparison process.
        cmp (int): The result of the comparison, indicating equality, less than, or greater than.
    """
    def __init__(self, walked: int, cmp: int):
        self.walked = walked
        self.cmp = cmp


class SortedLCCS:
    """
    Implements Sorted Longest Circular Co-Substring (LCCS) search algorithm for multi-dimensional data.

    Attributes:
        scan_size (int): The size of the range to perform linear scans in binary search fallback.
        n (int): The number of elements in the dataset.
        dim (int): The dimensionality of the dataset.
        step (int): The step size for dimensionality reduction in search.
        searchdim (int): The reduced number of dimensions to be searched.
        data (np.ndarray): The dataset.
        sortedidx (List[List[int]]): Indices of elements sorted per dimension.
        nextidx (List[List[int]]): Links to next elements for dimensionality-reduced search.
    """
    def __init__(self, step: int, data: np.ndarray, scan_size:int=4):
        self.scan_size = scan_size
        self.n = data.shape[0]
        self.dim = data.shape[1]
        self.step = step
        self.searchdim = ((self.dim - 1) // self.step) + 1
        self.data = data

        self.sortedidx = self.get_sort_idx(self.dim, self.n)
        self.nextidx = self.get_next_link(self.step)

    def compare_dim(self,xs: np.array, ys: np.array, start: int, walked: int) -> CmpLoc:
        """
        Compares two arrays element-wise in a circular manner starting from a given index.

        Parameters:
            xs (np.array): The first array to compare.
            ys (np.array): The second array to compare.
            start (int): The starting index for comparison.
            walked (int): The distance already walked in the comparison.

        Returns:
            CmpLoc: A `CmpLoc` object containing the distance walked and comparison result.
        """
        size = self.dim - walked
        for i in range(size):
            idx = (start + walked + i) % self.dim
            x = xs[idx]
            y = ys[idx]
            if x != y:
                return CmpLoc(i + walked, int(x > y) - int(x < y))
        return CmpLoc(self.dim, 0)

    def compare_dim_indices(self,idx1: int, idx2: int, start: int, walked: int) -> CmpLoc:
        """
        Compares elements at given indices of the dataset in a circular manner.

        Parameters:
            idx1 (int): The index of the first element in the dataset.
            idx2 (int): The index of the second element in the dataset.
            start (int): The starting dimension for comparison.
            walked (int): The initial offset in the dimension.

        Returns:
            CmpLoc: The result of the comparison and the total distance walked in the dimension.
        """
        xp = self.data[idx1]
        yp = self.data[idx2]
        return self.compare_dim(xp, yp, start, walked)

    def get_sort_idx(self,dim:int, n:int) -> List[List[int]]:
        """
        Sorts indices of the dataset for each dimension based on the comparison of elements.

        Parameters:
            dim (int): The dimension of the dataset.
            n (int): The number of elements in the dataset.

        Returns:
            List[List[int]]: A list of lists containing sorted indices for each dimension.
        """
        sorted_idx = [[i for i in range(n)] for _ in range(dim)]
        for d in range(dim):
            cmp = cmp_to_key(lambda idx1, idx2, d=d: self.compare_dim_indices(idx1, idx2, d, 0).cmp)
            sorted_idx[d].sort(key=cmp)
        return sorted_idx

    def get_next_link(self,step: int) -> List[List[int]]:
        """
        Creates links for each element to its successor in a dimensionally reduced search space.

        Parameters:
            step (int): The step size for reducing dimensionality in the search process.

        Returns:
            List[List[int]]: A list of lists where each sublist contains links to the next elements in the search space.
        """
        next_link = [[0] * self.n for _ in range(self.dim)]
        for d in range(self.dim - 1, -1, -1):
            next_dim = (d + step) % self.dim
            next_order = [0] * self.n
            for i in range(self.n):
                next_order[self.sortedidx[next_dim][i]] = i

            for i in range(self.n):
                idx =self.sortedidx[d][i]
                next_link[d][i] = next_order[idx]

        return next_link

    def get_data(self,query_dim: int, idx: int) -> np.array:
        """
        Retrieves data from the dataset at a given index sorted by a specific dimension.

        Parameters:
            query_dim (int): The dimension of interest.
            idx (int): The index in the sorted order.

        Returns:
            np.array: The data row at the specified index and dimension.
        """
        row = self.sortedidx[query_dim][idx]
        return self.data[row]

    def scan_loc(self,query: np.array, query_dim: int, low: int, low_len: int, high: int, high_len: int) -> Loc:
        """
        Performs a linear scan to refine the location of a query within a given range.

        Parameters:
            query (np.array): The query array.
            query_dim (int): The dimension to query.
            low (int): The lower bound of the scan range.
            low_len (int): The length at the lower bound.
            high (int): The upper bound of the scan range.
            high_len (int): The length at the upper bound.

        Returns:
            Loc: The location of the query after the scan.
        """
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
        """
        Compares data at a given index with the query, used as a callback for binary search.

        Parameters:
            i (int): The index in the sorted data to compare.
            prev (int): The previously walked distance in the comparison.
            query_dim (int): The dimension of the query.
            query (np.array): The query array.

        Returns:
            CmpLoc: The result of the comparison and the distance walked.
        """
        dpa = self.get_data(query_dim, i)
        return self.compare_dim(query, dpa, query_dim, prev)

    def binary_search_loc(self,query: np.array, query_dim: int, low: int, low_len: int, high: int, high_len: int) -> Loc:
        """
        Uses binary search to find the approximate location of a query in the dataset.

        Parameters:
            query (np.array): The query array.
            query_dim (int): The dimension of the query.
            low (int): The lower bound of the search range.
            low_len (int): The length at the lower bound.
            high (int): The upper bound of the search range.
            high_len (int): The length at the upper bound.

        Returns:
            Loc: The located position of the query in the search space.
        """
        next_low = low
        next_low_len = low_len
        next_high = high
        next_high_len = high_len

        while next_low < next_high - self.scan_size:
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
        """
        Finds the location of a query within the dataset using a combination of binary search and linear scan.

        Parameters:
            query (np.array): The query array.
            query_dim (int): The dimension to start the search.

        Returns:
            Loc: The location of the query.
        """
        low = 0
        high = self.n - 1
        datalow = self.get_data(query_dim, low)
        datahigh = self.get_data(query_dim, high)
        low_loc = self.compare_dim(query, datalow, query_dim, 0)
        high_loc = self.compare_dim(query, datahigh, query_dim, 0)

        return self.binary_search_loc(query, query_dim, low, low_loc.walked,high, high_loc.walked)

    def find_matched_locs(self,query: np.array) -> Locs:
        """
        Finds locations in the dataset that match the query across various dimensions.

        Parameters:
            query (np.array): The query array.

        Returns:
            Locs: A `Locs` object containing indices and lengths of matched locations.
        """
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
                    match_idx = self.sortedidx[d][i]
                    check(match_idx)

            for i in range(cur_idx + 1, self.n):
                if i - cur_idx - 1 < scan_step:
                    match_idx = self.sortedidx[d][i]
                    check(match_idx)

        for i in range(self.dim):
            check_loc(locs.idxes[i], i)

        return checked
