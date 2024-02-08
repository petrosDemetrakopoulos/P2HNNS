# pylint: disable=W0621, C0116
import numpy as np
import pytest
from src.utils.distance_functions import Dist
from src.utils import Query

class MockDist(Dist):
    """A mock distance function for testing purposes."""
    def distance(self, query, data):
        # Mock distance calculation
        return np.dot(query, data)

@pytest.fixture
def sample_data():
    query = np.array([1, 2, 3])
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    top = 2
    limit = 100
    dist = MockDist()
    return query, data, top, limit, dist

def test_query_initialization(sample_data):
    query, data, top, limit, dist = sample_data
    q = Query(query, data, top, limit, dist)

    np.testing.assert_array_equal(q.query, query)
    np.testing.assert_array_equal(q.data, data)
    assert q.top == top
    assert q.limit == limit
    assert q.dist == dist

def test_query_copy_with_new_dist_function(sample_data):
    query, data, top, limit, dist = sample_data
    q = Query(query, data, top, limit, dist)

    new_dist = MockDist()  # Assuming a new instance for the sake of test differentiation
    copied_query = q.copy(new_dist)

    # Check that the copied query shares the same attributes except for the dist
    np.testing.assert_array_equal(copied_query.query, q.query)
    np.testing.assert_array_equal(copied_query.data, q.data)
    assert copied_query.top == q.top
    assert copied_query.limit == q.limit
    assert copied_query.dist == new_dist
    assert copied_query.dist is not q.dist  # Ensure the dist instance is indeed different
