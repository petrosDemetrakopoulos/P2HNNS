from typing import List
import bisect
import numpy as np
from .IdxVal import IdxVal

class RandomSampler:
    """
    A class performing random sampling from a dataset using a strategy combining probabilistic and deterministic
    features to ensure a diverse and representative selection of samples. This sampler is particularly designed to
    handle high-dimensional data and is capable of capturing both individual and interactive effects between
    data dimensions through a unique sampling mechanism.
    
    The sampling strategy involves:
    1. Creating a cumulative probability vector based on the squared values of the data, emphasizing 
       regions with higher magnitude for selection.
    2. Deterministically selecting the initial sample from the end of the dimension space, ensuring the
       inclusion of edge data.
    3. Probabilistically selecting subsequent samples based on the cumulative probability distribution,
       with adjustments to explore cross-dimension interactions.
    4. Preventing the selection of duplicate samples by maintaining a record of already-selected samples.

    Attributes:
        dim (int): The dimension of the input data.
        afterdim (int): The computed dimension to consider after processing, accommodating cross-dimension sampling.
        sampledim (int): The dimension of the sample to be drawn, calculated as the product of `dim` and a scaling factor `s`.

    Parameters for initialization:
        dim (int): The dimension of the input data.
        s (int): A scaling factor to determine the size of the sample relative to the input data's dimension.
    """
    def __init__(self, dim: int, s: int):
        self.dim = dim
        self.afterdim = dim * (dim + 1) // 2 + 1
        self.sampledim = dim * s

    def sampling(self, data: np.array) -> List[IdxVal]:
        """
        Perform random sampling from the given data utilizing a strategy that combines probabilistic
        selection with deterministic checks to ensure a diverse and comprehensive sample set. This method
        specifically targets capturing both the individual and interactive effects between data dimensions
        through a nuanced sampling mechanism.
        
        Parameters:
            data (np.array): The input data array from which to sample.
            
        Returns:
            List[IdxVal]: A list of IdxVal objects representing the sampled indices and their corresponding
                          values squared (for individual dimensions) or multiplied (for cross dimensions),
                          ensuring a varied representation of the dataset's characteristics.
        """
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
        """
        Generate a probability vector from the input data.
        
        Parameters:
            dim (int): The dimension of the input data.
            data (np.array): The input data array.
            
        Returns:
            np.array: A probability vector for the input data.
        """
        prob = np.zeros(dim)
        prob[0] = data[0] ** 2
        for i in range(1, dim):
            prob[i] = prob[i - 1] + data[i] * data[i]
        return prob

    def search_idx_from(self, d: int, prob: np.array) -> int:
        """
        Generate a probability vector from the input data, which is used to guide the probabilistic
        selection of samples. This vector is a cumulative sum of the squared values of the data,
        emphasizing areas of higher magnitude for selection, facilitating a biased sampling towards
        significant features in the data.
        
        Parameters:
            dim (int): The dimension of the input data.
            data (np.array): The input data array.
            
        Returns:
            np.array: A probability vector for the input data, used to guide the sampling process.
        """
        end = prob[d - 1]
        assert 0 < end, f"must 0 < sigma({end})"
        # Generate a random Gaussian number with mean 0 and standard deviation 'end'
        rnd = np.random.normal(0.0, end)
        # Perform a binary search for 'rnd' in the slice of 'prob' up to 'd'
        idx = bisect.bisect_left(prob, rnd, 0, d)
        # Python's bisect_left returns the insertion point which can be used directly
        return max(0, idx - 1) if idx < d else max(0, d - 2)
