from abc import ABC, abstractmethod
import numpy as np
from .operations import norm, dot

class Dist(ABC):
    """
    Distance Extractor.
    """

    @abstractmethod
    def distance(self, query: np.array, data: np.array) -> float:
        """
        Calculate the distance between query and data.
        """
        pass

    @staticmethod
    def value_of(value):
        """
        Get Dist implementation based on the provided name.
        """
        if value == "ABS_DOT":
            return DistAbsDot()
        elif value == "COS":
            return DistCos()
        else:
            return DistDP2H()

class DistAbsDot(Dist):
    """
    Absolute Dot Product distance.
    """

    def distance(self, query: np.array, data: np.array) -> float:
        return np.abs(np.dot(query, data))

class DistCos(Dist):
    """
    Cosine distance.
    """

    def distance(self, query: np.array, data: np.array) -> float:
        return 1 - np.dot(query, data) / (norm(query) * norm(data))

class DistDP2H(Dist):
    """
    Distance from a data point p to the hyperplane query q.
    DP2H aims to find the k closest data points to the hyperplane query.
    """

    def distance(self, query: np.array, data: np.array) -> float:
        last = len(query) - 1
        return np.abs(query[last] + dot(data, query,last)) / np.sqrt(dot(query, query, last))
