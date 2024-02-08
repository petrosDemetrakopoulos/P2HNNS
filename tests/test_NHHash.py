# pylint: disable=W0621, C0116
import numpy as np
import pytest
from P2HNNS.methods.NHHash import NHHash, Signature
from P2HNNS.utils.RandomSampler import RandomSampler
from P2HNNS.utils.distance_functions import DistAbsDot
from P2HNNS.utils import Query,IdxVal

@pytest.fixture
def nh_hash_setup():
    d = 5  # Dimensionality of the data
    m = 10  # Number of hash functions
    s = 3   # Sample size
    w = 1.0 # Width of hash bins
    nh_hash = NHHash(d=d, m=m, s=s, w=w)
    return nh_hash

def test_initialization(nh_hash_setup):
    nh_hash = nh_hash_setup
    assert nh_hash.m == 10
    assert nh_hash.w == 1.0
    assert isinstance(nh_hash.sampler, RandomSampler)
    assert len(nh_hash.proja) == nh_hash.m * nh_hash.nhdim
    assert len(nh_hash.projb) == nh_hash.m

def test_sampling_signature(nh_hash_setup):
    nh_hash = nh_hash_setup
    data_point = np.random.rand(5)
    signature = nh_hash.sampling_signature(data_point)
    assert isinstance(signature, Signature)
    assert signature.value.shape[0] == nh_hash.m
    assert signature.norm >= 0

def test_hash_data(nh_hash_setup):
    nh_hash = nh_hash_setup
    data = np.random.rand(100, 5)  # 100 random data points
    hashed_data = nh_hash.hash_data(data)
    assert hashed_data.shape == (100, nh_hash.m)

def test_hash_query(nh_hash_setup):
    nh_hash = nh_hash_setup
    query_point = np.random.rand(5)
    hashed_query = nh_hash.hash_query(query_point)
    assert hashed_query.shape[0] == nh_hash.m

def test_nns(nh_hash_setup):
    nh_hash = nh_hash_setup
    mock_data = np.random.rand(100, 5)  # 100 random data points
    query = Query(query=np.random.rand(5), data=mock_data, top=5, limit=1000, dist=DistAbsDot())
    nh_hash.build_index(data=mock_data)
    nns_result = nh_hash.nns(query)
    assert isinstance(nns_result, list)
    assert len(nns_result) <= query.top
    assert len(nns_result) <= query.limit
    assert all(isinstance(result, IdxVal) for result in nns_result)
