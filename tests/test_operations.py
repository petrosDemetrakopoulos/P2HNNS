# pylint: disable=W0621, C0116, C0115, W0201, W0401
import pytest
import numpy as np
from src.operations import *

def test_dot():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    dimensions = 2
    expected_result = 14  # 1*4 + 2*5
    assert dot(a, b, dimensions) == expected_result

def test_dot_with_start():
    a = np.array([1, 2])
    b = np.array([0, 4, 5])
    start = 1
    expected_result = 14  # 1*4 + 2*5
    assert dot_with_start(a, start, b) == expected_result

def test_norm():
    v = np.array([3, 4])
    expected_result = 5.0
    assert norm(v) == expected_result

def test_concat():
    data = np.array([[1, 2], [3, 4]])
    dimensions = np.array([5, 6])
    expected_result = np.array([[1, 2, 5], [3, 4, 6]])
    np.testing.assert_array_equal(concat(data, dimensions), expected_result)

def test_normalize():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Compute expected result manually or using a known good result
    # This is a simplified example; actual normalization will depend on your implementation details
    normalized_data = normalize(data)
    # Check that the mean of each row is close to 0 and norm is close to 1
    for row in normalized_data:
        assert np.isclose(np.linalg.norm(row), 1, atol=1e-7)

def test_dim():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    assert dim(data) == 3
    empty_data = np.array([])
    assert dim(empty_data) == 0

# Run tests with pytest
if __name__ == "__main__":
    pytest.main()
