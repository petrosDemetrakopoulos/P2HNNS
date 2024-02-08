from copy import deepcopy
from .Query import Query

class FHQuery(Query):
    """
    Represents a query for point-to-hyperplane nearest neighbor searches specifically for the Furthest Hyperplane (FH) hashing method. 
    This class encapsulates the details of the query hyperplane, the dataset against which the query is executed, 
    and parameters such as the number of top results to return and a limit for search optimization.

    Attributes / Parameters for initialization:
        query (np.array): The hyperplane coefficients representing the query.
        data (np.ndarray): The dataset array against which the query will be executed.
        top (int): The number of top nearest neighbors to return.
        limit (int): A limit for search optimization, possibly affecting the search algorithm's performance or accuracy.
        l (int): Separation threshold
        dist (Dist): The distance function used to calculate distances between points and the hyperplane.

    Methods:
        copy(self, dist: Dist) -> 'Query':
            Creates a deep copy of the current Query instance with a new distance function.
    """
    def __init__(self, query, data, top, limit, l, dist):
        super().__init__(query, data, top, limit, dist)
        self.l = l

    def copy(self, dist):
        return FHQuery(deepcopy(self.query), deepcopy(self.data), self.top, self.limit, self.l, dist)
