# pylint: disable=W0621, C0116
from copy import deepcopy
import pytest
import numpy as np
from src import FHQuery
#set seed for stability of random numbers
np.random.seed(0)
# Fixture to initialize and return a FHQuery object for each test
@pytest.fixture
def fh_query():
    query = np.array([x for x in range(51)])
    data = np.array([np.array([x for x in range(50)]) for _ in range(50)])
    top = 5
    limit = 10
    l = 2
    dist = 0.5
    return FHQuery(query, data, top, limit, l, dist)

def test_initialization(fh_query):
    np.testing.assert_array_equal(fh_query.query, np.array([x for x in range(51)]))
    np.testing.assert_array_equal(fh_query.data, np.array([np.array([x for x in range(50)]) for _ in range(50)]))
    assert fh_query.top == 5
    assert fh_query.limit == 10
    assert fh_query.l == 2
    assert fh_query.dist == 0.5

def test_copy(fh_query):
    new_dist = 0.75
    fh_query_copy = fh_query.copy(new_dist)

    assert fh_query_copy is not fh_query
    np.testing.assert_array_equal(fh_query_copy.query, deepcopy(fh_query.query), "Query arrays are not equal")
    np.testing.assert_array_equal(fh_query_copy.data, deepcopy(fh_query.data), "Data arrays are not equal")
    assert fh_query_copy.top == fh_query.top
    assert fh_query_copy.limit == fh_query.limit
    assert fh_query_copy.l == fh_query.l
    assert fh_query_copy.dist == new_dist
