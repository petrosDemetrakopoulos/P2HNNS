# pylint: disable=W0621, C0116, C0115, W0201, W0401
import numpy as np
import pytest
from P2HNNS.utils.RQALSH import RQALSH, SearchPosition
from P2HNNS.utils.IdxVal import IdxVal

@pytest.fixture
def setup_rqalsh():
    # Create a simple setup for testing
    n = 10
    dim = 2
    m = 3
    index = np.arange(n)
    norm = np.ones(n)
    data = [[IdxVal(idx, val) for idx, val in enumerate(np.random.rand(dim))] for _ in range(n)]
    rqalsh = RQALSH(n=n, dim=dim, m=m, index=index, norm=norm, data=data)
    return rqalsh

def test_init_rqalsh(setup_rqalsh):
    rqalsh = setup_rqalsh
    assert rqalsh.n == 10 and rqalsh.dim == 2 and rqalsh.m == 3, "Initialization parameters do not match."

def test_calc_hash_value(setup_rqalsh):
    rqalsh = setup_rqalsh
    tid = 1
    data = np.random.rand(rqalsh.dim)
    # Just testing if it executes without error as actual value depends on random a
    assert isinstance(rqalsh._calc_hash_value(tid, data), float), "Hash value should be a float."

def test_hash_value_with_zero_vector(setup_rqalsh):
    rqalsh = setup_rqalsh
    zero_vector = np.zeros(rqalsh.dim)
    tid = 0
    hash_value = rqalsh._calc_hash_value(tid, zero_vector)
    assert isinstance(hash_value, float), "Hash value should be a float for zero vector input."

def test_hash_value_with_negative_values(setup_rqalsh):
    rqalsh = setup_rqalsh
    negative_vector = -np.ones(rqalsh.dim)
    tid = 0
    hash_value = rqalsh._calc_hash_value(tid, negative_vector)
    assert isinstance(hash_value, float), "Hash value should be a float for negative vector input."

def test_get_search_position(setup_rqalsh):
    rqalsh = setup_rqalsh
    sample_dim = 2
    query = [IdxVal(idx, val) for idx, val in enumerate(np.random.rand(sample_dim))]
    pos = rqalsh.get_search_position(sample_dim, query)
    assert isinstance(pos, SearchPosition), "Should return an instance of SearchPosition."
    assert len(pos.query_val) == rqalsh.m, "Length of query_val should match the number of hash tables."

def test_find_radius(setup_rqalsh):
    rqalsh = setup_rqalsh
    w = 1.0
    sample_dim = 2
    query = [IdxVal(idx, val) for idx, val in enumerate(np.random.rand(sample_dim))]
    pos = rqalsh.get_search_position(sample_dim, query)
    radius = rqalsh.find_radius(w, pos)
    assert isinstance(radius, float), "Radius should be a float."

def test_calc_hash_value_1(setup_rqalsh):
    rqalsh = setup_rqalsh
    d = 2
    tid = 1
    last = 0.0
    data = [IdxVal(0, 0.5), IdxVal(1, 0.5)]
    # Test calculates without error, actual value depends on a
    assert isinstance(rqalsh._calc_hash_value_1(d, tid, last, data), float), "Hash value should be a float."

def test_calc_hash_value_2(setup_rqalsh):
    rqalsh = setup_rqalsh
    d = 2
    tid = 0
    data = [IdxVal(0, 0.5), IdxVal(1, 0.5)]
    # Test calculates without error, actual value depends on a
    assert isinstance(rqalsh._calc_hash_value_2(d, tid, data), float), "Hash value should be a float."

def test_dynamic_separation_counting(setup_rqalsh):
    rqalsh = setup_rqalsh
    l = 1
    limit = 5
    R = 0.1
    sample_dim = 2
    query = [IdxVal(idx, val) for idx, val in enumerate(np.random.rand(sample_dim))]
    pos = rqalsh.get_search_position(sample_dim, query)
    candidates = rqalsh.dynamic_separation_counting(l, limit, R, pos)
    # This test assumes some mock setup that would produce candidates
    assert isinstance(candidates, list), "Should return a list of candidates."

def test_fns(setup_rqalsh):
    rqalsh = setup_rqalsh
    l = 1
    limit = 5
    R = 0.1
    sample_dim = 2
    query = [IdxVal(idx, val) for idx, val in enumerate(np.random.rand(sample_dim))]
    candidates = rqalsh.fns(l, limit, R, sample_dim, query)
    # This test checks for the return type and assumes fns method logic correctness
    assert isinstance(candidates, list), "Should return a list of candidates."

def test_search_position_with_corner_cases(setup_rqalsh):
    rqalsh = setup_rqalsh
    corner_case_query = [IdxVal(idx, 0) for idx in range(rqalsh.dim)]  # All-zero case
    position = rqalsh.get_search_position(rqalsh.dim, corner_case_query)
    assert all(pos >= 0 for pos in position.left_pos), "Left positions should be non-negative."
    assert all(pos <= rqalsh.n - 1 for pos in position.right_pos), "Right positions should be within bounds."

def test_dynamic_separation_counting_with_various_limits(setup_rqalsh):
    rqalsh = setup_rqalsh
    query = [IdxVal(idx, val) for idx, val in enumerate(np.random.rand(rqalsh.dim))]
    position = rqalsh.get_search_position(rqalsh.dim, query)

    for limit in [1, rqalsh.n // 2, rqalsh.n * 2]:  # Testing various limits
        candidates = rqalsh.dynamic_separation_counting(1, limit, 0.1, position)
        assert len(candidates) <= limit, f"Number of candidates should not exceed the limit ({limit})."

def test_dynamic_separation_counting_with_small_radius(setup_rqalsh):
    rqalsh = setup_rqalsh
    query = [IdxVal(idx, val) for idx, val in enumerate(np.random.rand(rqalsh.dim))]
    position = rqalsh.get_search_position(rqalsh.dim, query)
    candidates = rqalsh.dynamic_separation_counting(1, 10, 0.0001, position)  # Very small radius
    assert isinstance(candidates, list), "Should return a list of candidates."

def test_rqalsh_with_invalid_dimensions():
    with pytest.raises(ValueError):
        RQALSH(n=10, dim=-2, m=3, index=np.arange(10), norm=np.ones(10), data=[], scan_size=100, check_error=1e-6)
