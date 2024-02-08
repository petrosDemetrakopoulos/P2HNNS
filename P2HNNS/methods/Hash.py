from abc import abstractmethod
from typing import TypeVar, List
import numpy as np
from ..utils.IdxVal import IdxVal
from ..utils.Query import Query

T = TypeVar('T', bound=Query)

class Hash():
    """
    Abstract base class for implementing hash-based nearest neighbors search algorithms. This class defines
    the interface for hashing data points and hyperplane queries, building the index for fast point-to-hyperplane 
    nearest neighbor searches, and performing the actual search. 
    Concrete implementations (BHHash, EHHash, MHHash, NHHash and FHHash) provide functionality for these operations,
    tailored to specific hashing techniques.

    Methods:
        hash_data(self, data: np.array) -> np.array:
            Hashes the input data into a lower-dimensional space for efficient similarity search.

        hash_query(self, query: np.array) -> np.array:
            Hashes the query point(s) into the same lower-dimensional space as the data for comparison.

        nns(self, param: T) -> List[IdxVal]:
            Performs a nearest neighbor search, returning the closest data points to the query.

        build_index(self, data: np.array):
            Constructs the index from the input dataset to enable fast nearest neighbor searches.
    """
    @abstractmethod
    def hash_data(self, data: np.array) -> np.array:
        """
        Hashes the input data points into a lower-dimensional representation, which is used for
        building the index and performing efficient similarity searches.

        Parameters:
            data (np.array): The high-dimensional data points to be hashed, typically as a 2D numpy array
                             where each row is a data point.

        Returns:
            np.array: The hashed representation of the input data, usually in a lower-dimensional space.
        """
        pass

    @abstractmethod
    def hash_query(self, query: np.array) -> np.array:
        """
        Hashes the query point(s) into the same lower-dimensional space as the dataset to facilitate
        comparison and nearest neighbor search.

        Parameters:
            query (np.array): The query point(s) to be hashed, which may be a single data point or multiple
                              points in a 2D array format.

        Returns:
            np.array: The hashed representation of the query point(s).
        """
        pass

    @abstractmethod
    def nns(self, param: T) -> List[IdxVal]:
        """
        Performs a nearest neighbor search to find the closest points in the dataset to the given query.

        Parameters:
            param (T): The query parameter, typically encapsulating the query point(s) and possibly
                       additional search parameters specific to the concrete implementation.

        Returns:
            List[IdxVal]: A list of IdxVal objects, each representing a data point (by index) and its
                          distance or similarity measure to the query. The list is usually ordered by
                          increasing distance or decreasing similarity.
        """
        pass

    @abstractmethod
    def build_index(self, data: np.array):
        """
        Constructs an index from the provided dataset to enable fast nearest neighbor searches. This
        method processes the entire dataset, typically hashing it and organizing the hashed values
        in a way that optimizes the search performance.

        Parameters:
            data (np.array): The dataset to build the index from, generally as a 2D array where each
                             row represents a data point.
        """
        pass
