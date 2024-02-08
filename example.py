# pylint: disable=W0621, C0116, C0103
import numpy as np
from P2HNNS.methods import FHHash, BHHash, MHHash, NHHash, EHHash
from P2HNNS.utils import FHQuery, Query
from P2HNNS.utils.distance_functions import DistDP2H, DistAbsDot

# initialize 20 points in the 2D space
example_data = np.array([[1.5,0.5],[0.5,0.2],[2,2],[1,1.5],[0.8,0.5],[2,2.5],[3,2],[5,8],[4,6],[5,5],[10,10],[12,7],[8,6],[5,2],[16,32],[10,5],[50,45],[-13,15],[7,6],[9,8]])
# we add a last dimension filled with 1s to the data, this is to bring the ararys into the same dimentionality
# it does not affect relative distance between the points and it is performed for technical reasons
example_data = np.append(example_data, np.full((20,1), fill_value=1.0), axis=1)

# initialize a random hyperplane in the 2D space (aka a line)
# recall that a line is defined by 3 contsants (thus a 3-dimensional vector):
# A x coefficient, a y coefficient and a constant.
example_query_hyperplane = np.array([1,1,-1]) # y + x -1 = 0

def example_FH():
    fh_index = FHHash(d=3,s=10,b = 0.5,m=10,max_blocks=1600)
    fh_index.build_index(example_data)

    query = FHQuery(query=example_query_hyperplane, data=example_data, top=5, limit=1000,l=2, dist=DistDP2H())
    results = fh_index.nns(query)
    print("FH Results:")
    print(results)

def example_BH():
    bh_index = BHHash(d=3, m=2,l=10)
    bh_index.build_index(example_data)
    query = Query(query=example_query_hyperplane, data=example_data, top=5, limit=1000, dist=DistDP2H())
    results = bh_index.nns(query)

    print("BH Results:")
    print(results)

def example_EH():
    eh_index = EHHash(d=3, m=2, l=10)
    eh_index.build_index(example_data)
    query = Query(query=example_query_hyperplane, data=example_data, top=5, limit=1000, dist=DistDP2H())
    results = eh_index.nns(query)

    print("EH Results:")
    print(results)

def example_MH():
    mh_index = MHHash(dimension=3, m=10, l=2, M=2)
    mh_index.build_index(example_data)
    query = Query(query=example_query_hyperplane, data=example_data, top=5, limit=1000, dist=DistAbsDot())
    results = mh_index.nns(query)

    print("MH Results:")
    print(results)

def example_NH():
    nh_index = NHHash(d=3, m=10, w=2, s=5)
    nh_index.build_index(example_data)
    query = Query(query=example_query_hyperplane, data=example_data, top=5, limit=1000, dist=DistAbsDot())
    results = nh_index.nns(query)

    print("NH Results:")
    print(results)

if __name__ == "__main__":
    example_FH()
    example_BH()
    example_EH()
    example_MH()
    example_NH()
