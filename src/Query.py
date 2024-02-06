import copy
import numpy as np
from .distance_functions import Dist

class Query:
    """
    Represents a query for point-to-hyperplane nearest neighbor searches. This class encapsulates the details
    of the query hyperplane, the dataset against which the query is executed, and parameters such as the number
    of top results to return and a limit for search optimization.

    Attributes:
        query (np.array): The hyperplane coefficients representing the query.
        data (np.ndarray): The dataset array against which the query will be executed.
        top (int): The number of top nearest neighbors to return.
        limit (int): A limit for search optimization, possibly affecting the search algorithm's performance or accuracy.
        dist (Dist): The distance function used to calculate distances between points and the hyperplane.

    Parameters:
        query (np.array): An array representing the query hyperplane.
        data (np.ndarray): The dataset to search against.
        top (int): The number of nearest neighbors to find.
        limit (int): A search optimization parameter.
        dist (Dist): An instance of a distance function class used for calculating distances.

    Methods:
        copy(self, dist: Dist) -> 'Query':
            Creates a deep copy of the current Query instance with a new distance function.
    """

    def __init__(self, query: np.array, data: np.ndarray, top: int, limit: int, dist: Dist):
        """
        Initializes a new Query instance with the specified query hyperplane, dataset, top results limit,
        search limit, and distance function.

        Parameters:
            query (np.array): Coefficients defining the query hyperplane.
            data (np.ndarray): The dataset in which to perform the search.
            top (int): The number of nearest neighbors to retrieve.
            limit (int): A limit for optimizing the search process.
            dist (Dist): The distance function to use for calculating point-to-hyperplane distances.
        """
        self.query = query
        self.data = data
        self.top = top
        self.limit = limit
        self.dist = dist

    def copy(self, dist: Dist) -> 'Query':
        """
        Creates a deep copy of this Query instance, allowing for the specification of a new distance function,
        while preserving the query, data, top, and limit parameters.

        Parameters:
            dist (Dist): The new distance function to use in the copied Query instance.

        Returns:
            Query: A new Query instance with the same query, data, top, and limit parameters, but using the
                   specified distance function.
        """
        return Query(copy.deepcopy(self.query), copy.deepcopy(self.data), self.top, self.limit, dist)
