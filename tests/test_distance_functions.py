# pylint: disable=W0621, C0116, C0115, W0201, E0110
import numpy as np
import pytest
from src.utils.distance_functions import Dist, DistAbsDot, DistCos, DistDP2H

def test_abs_dot_distance():
    query = np.array([1, 2, 3])
    data = np.array([4, 5, 6])
    dist = DistAbsDot()
    expected_result = np.abs(np.dot(query, data))
    assert dist.distance(query, data) == expected_result

def test_cos_distance():
    query = np.array([1, 0, 0])
    data = np.array([0, 1, 0])
    dist = DistCos()
    expected_result = 1  # Cosine distance between orthogonal vectors
    assert dist.distance(query, data) == expected_result

def test_dp2h_distance():
    query = np.array([1, 1, -1])  # Hyperplane x + y = 1
    data = np.array([0, 0])  # Origin
    dist = DistDP2H()
    expected_result = 1 / np.sqrt(2)  # Distance to the hyperplane
    assert np.isclose(dist.distance(query, data), expected_result)

def test_dist_value_of():
    assert isinstance(Dist.value_of("ABS_DOT"), DistAbsDot)
    assert isinstance(Dist.value_of("COS"), DistCos)
    assert isinstance(Dist.value_of("DP2H"), DistDP2H)

# Optional: Test the abstract class behavior (not instantiable, etc.)
def test_dist_abstract_class():
    with pytest.raises(TypeError):
        Dist()  # Attempting to instantiate should raise TypeError
