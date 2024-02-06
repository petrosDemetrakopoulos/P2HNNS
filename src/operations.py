import numpy as np
# Implementation based on
# https://github.com/stepping1st/hyperplane-hash/blob/master/src/main/java/io/github/stepping1st/hh/Op.java
def dot(a: np.array, b: np.array, dimensions: int) -> float:
    """
    Computes the dot product of two vectors up to a specified number of dimensions.

    Parameters:
        a (np.array): The first vector.
        b (np.array): The second vector.
        dimensions (int): The number of dimensions to consider in the dot product calculation.

    Returns:
        float: The dot product of the two vectors considering only the specified dimensions.
    """
    return np.dot(a[:dimensions], b[:dimensions])

def dot_with_start(a: np.array, start: int, b: np.array) -> float:
    """
    Computes the dot product of two vectors, with the second vector's relevant section starting from a specified index.

    Parameters:
        a (np.array): The first vector.
        start (int): The starting index in the second vector to begin the dot product calculation.
        b (np.array): The second vector.

    Returns:
        float: The dot product of the two vectors, with the second vector's calculation starting from 'start'.
    """
    return np.dot(a, b[start:start+len(a)])

def norm(v: np.array) -> float:
    """
    Computes the Euclidean norm (magnitude) of a vector.

    Parameters:
        v (np.array): The vector to compute the norm of.

    Returns:
        float: The Euclidean norm of the vector.
    """
    return np.sqrt(np.dot(v, v))

def concat(data: np.ndarray, dimensions: np.array) -> np.ndarray:
    """
    Concatenates a 1D array to each row of a 2D array, effectively adding a new dimension to the data.

    Parameters:
        data (np.ndarray): The original 2D array.
        dimensions (np.array): The 1D array to concatenate to each row of 'data'.

    Returns:
        np.ndarray: The concatenated array.

    Raises:
        AssertionError: If the length of 'dimensions' does not match the number of rows in 'data'.
    """
    assert data.shape[0] == len(dimensions), "data and dim must have the same length in the first dimension"
    dim_expanded = np.expand_dims(dimensions, axis=1)  # Convert dim to a column vector
    result = np.hstack((data, dim_expanded))    # Horizontally stack the arrays
    return result

def normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalizes the rows of a 2D array such that each row has a mean of 0 and a norm of 1.

    Parameters:
        data (np.ndarray): The data to normalize.

    Returns:
        np.ndarray: The normalized data.
    """
    dimensions = len(data[0])
    result = np.zeros_like(data)
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    for i in range(len(data)):
        sumnorm = 0.0
        for d in range(dimensions):
            center = (mins[d] + maxs[d]) / 2.0
            diff = data[i, d] - center
            result[i, d] = diff
            sumnorm += diff ** 2
        norm_val = np.sqrt(sumnorm)
        result[i, :] /= norm_val
    return result

def dim(data: np.ndarray) -> int:
    """
    Determines the dimensionality of the first row of a 2D array.

    Parameters:
        data (np.ndarray): The data array.

    Returns:
        int: The number of dimensions (columns) in the first row of 'data', or 0 if 'data' is empty.
    """
    if len(data) > 0:
        return len(data[0])
    else:
        return 0
