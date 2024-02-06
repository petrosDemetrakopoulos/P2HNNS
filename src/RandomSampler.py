from typing import List
import bisect
import numpy as np
from .IdxVal import IdxVal

class RandomSampler:
    def __init__(self, dim: int, s: int):
        self.dim = dim
        self.afterdim = dim * (dim + 1) // 2 + 1
        self.sampledim = dim * s

    def sampling(self, data: np.array) -> List[IdxVal]:
        sample = [None] * self.sampledim

        prob = self.probability_vector(self.dim, data)

        cnt = 0
        checked = [False] * self.afterdim

        sid = self.dim - 1
        checked[sid] = True
        sample[cnt] = IdxVal(sid, data[sid] ** 2)
        cnt += 1
        for _ in range(1, self.sampledim):
            idx = self.search_idx_from(self.dim - 1, prob)
            idy = self.search_idx_from(self.dim, prob)
            if idx > idy:
                idx, idy = idy, idx
            if idx == idy:
                sid = idx
                if not checked[sid]:
                    checked[sid] = True
                    sample[cnt] = IdxVal(sid, data[idx] ** 2)
                    cnt += 1
            else:
                sid = self.dim + (idx * self.dim - idx * (idx + 1) // 2) + (idy - idx - 1)
                if not checked[sid]:
                    checked[sid] = True
                    sample[cnt] = IdxVal(sid, data[idx] * data[idy])
                    cnt += 1
        return sample[:cnt]

    def probability_vector(self, dim: int, data: np.array) -> np.array:
        prob = np.zeros(dim)
        prob[0] = data[0] ** 2
        for i in range(1, dim):
            prob[i] = prob[i - 1] + data[i] * data[i]
        return prob

    def search_idx_from(self, d: int, prob: np.array) -> int:
        end = prob[d - 1]
        assert 0 < end, f"must 0 < sigma({end})"
        # Generate a random Gaussian number with mean 0 and standard deviation 'end'
        rnd = np.random.normal(0.0, end)
        # Perform a binary search for 'rnd' in the slice of 'prob' up to 'd'
        idx = bisect.bisect_left(prob, rnd, 0, d)
        # Python's bisect_left returns the insertion point which can be used directly
        return max(0, idx - 1) if idx < d else max(0, d - 2)
