import numpy as np
import pytest
from P2HNNS.utils.RandomSampler import RandomSampler
from P2HNNS.utils.IdxVal import IdxVal

class TestRandomSampler:
    @pytest.fixture
    def sampler(self):
        dim = 5
        s = 2
        return RandomSampler(dim, s)

    def test_init(self, sampler):
        assert sampler.dim == 5
        assert sampler.afterdim == 5 * (5 + 1) // 2 + 1
        assert sampler.sampledim == 5 * 2

    def test_probability_vector(self, sampler):
        data = np.array([1, 2, 3, 4, 5])
        expected_prob = np.array([1, 5, 14, 30, 55])
        np.testing.assert_array_equal(sampler.probability_vector(5, data), expected_prob)

    def test_sampling(self, sampler):
        np.random.seed(42)  # Set seed for reproducibility
        data = np.array([1, 2, 3, 4, 5])
        samples = sampler.sampling(data)
        # Test properties that should always be true rather than specific outcomes due to randomness
        assert len(samples) <= sampler.sampledim
        assert all(isinstance(sample, IdxVal) for sample in samples)
        assert all(sample.idx < sampler.afterdim for sample in samples)

    def test_search_idx_from(self, sampler):
        np.random.seed(42)  # Set seed for reproducibility
        data = np.array([1, 2, 3, 4, 5])
        prob = sampler.probability_vector(5, data)
        idx = sampler.search_idx_from(5, prob)
        # Due to randomness, we focus on the validity of the index rather than its specific value
        assert 0 <= idx < 5

# Run tests with pytest
if __name__ == "__main__":
    pytest.main()
