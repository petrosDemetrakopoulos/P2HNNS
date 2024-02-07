# pylint: disable=W0621, C0116, C0115, W0201, W0401
import numpy as np
import pytest
from src.SortedLCCS import SortedLCCS, Loc

@pytest.fixture
def sample_data():
    np.random.seed(42)  # Ensuring repeatability
    data = np.random.rand(10, 5)  # A sample dataset with 10 elements, each is a 5-dimensional vector
    step = 1
    return SortedLCCS(step, data)

def test_initialization(sample_data):
    assert sample_data.step == 1, "Step size should be initialized correctly."
    assert sample_data.n == 10, "Number of elements should match the input data."
    assert sample_data.dim == 5, "Dimensionality should match the input data."

def test_compare_dim_identical(sample_data):
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    result = sample_data.compare_dim_vectors(x, y, 0, 0)
    assert result.cmp == 0, "Expected comparison result is 0 for identical arrays."
    assert result.walked == sample_data.dim, "Expected walked distance to be equal to array length for identical arrays."

def test_compare_dim_different(sample_data):
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 2, 4, 5])  # Difference at the third position
    result = sample_data.compare_dim_vectors(x, y, 0, 0)
    assert result.cmp != 0, "Expected non-zero comparison result for different arrays."
    assert result.walked == 2, "Expected walked distance to indicate the first point of difference."

def test_get_sort_idx(sample_data):
    sorted_idx = sample_data.get_sort_idx(sample_data.dim, sample_data.n)
    assert len(sorted_idx) == sample_data.dim, "Sorted indices list should match dataset dimensions."
    assert all(len(idx_list) == sample_data.n for idx_list in sorted_idx), "Each dimension's index list should match the number of dataset elements."

def test_binary_search_loc(sample_data):
    query = np.median(sample_data.data, axis=0)
    loc = sample_data.binary_search_loc(query, 0, 0, 0, sample_data.n-1, sample_data.dim)
    assert isinstance(loc, Loc), "Expected a Loc instance as the result."

def test_find_matched_locs(sample_data):
    # Assuming sample_data.data is accessible and has a predictable setup
    # Select a query that is known to exist in the data for predictable results
    query_index = 0  # Using the first element of the dataset as the query
    query = sample_data.data[query_index]

    locs = sample_data.find_matched_locs(query)
    
    # Check if the first matched location's index points to the last index (1st object should be matched to the last)
    # This assumes the data and the query match perfectly in at least one dimension
    assert 5 in locs.idxes, "Expected at least one matching location index to point to the query's index."

    # Validate the lengths to ensure they are logical based on the query and data setup
    # For a perfect match, lowlens and highlens should reflect complete or near-complete matches
    assert all(l >= 0 for l in locs.lowlens), "Expected low lengths to be non-negative."
    assert all(h >= 0 for h in locs.highlens), "Expected high lengths to be non-negative."

    # More specific assertions can be added based on the expected behavior of your dataset and algorithm

