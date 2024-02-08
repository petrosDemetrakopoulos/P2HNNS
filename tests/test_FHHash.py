# pylint: disable=W0621, C0116, C0103, W0613
from unittest.mock import patch
import numpy as np
import pytest
from P2HNNS.methods import FHHash
from P2HNNS.methods.FHHash import Transform
from P2HNNS.utils import IdxVal, FHQuery
from P2HNNS.utils.distance_functions import DistDP2H

@pytest.fixture
def sample_data():
    return np.random.rand(10, 5)  # 10 data points in 5 dimensions

@pytest.fixture
def transform_instance():
    norm = np.array([1, 2, 3])
    centroid = np.array([0.5, 0.5, 0.5])
    samples = [[(0, 0.1), (1, 0.2)], [(2, 0.3), (3, 0.4)], [(4, 0.5), (5, 0.6)]]
    M = 3
    dist = [(0, 0.1), (1, 0.2), (2, 0.3)]
    return Transform(norm, centroid, samples, M, dist)

@pytest.fixture
def fhhash_instance():
    return FHHash(d=5,s=10,b = 0.5,m=10,max_blocks=1600)

def test_transform_initialization(transform_instance):
    assert transform_instance.norm is not None
    assert transform_instance.centroid is not None
    assert transform_instance.samples is not None
    assert transform_instance.M == 3.0
    assert transform_instance.dist is not None

@patch('P2HNNS.utils.RandomSampler')
def test_fhhash_initialization(MockRandomSampler, fhhash_instance):
    assert fhhash_instance.hashs == []
    assert fhhash_instance.fhdim == 16  # For d=5, calculated as 5*(5+1)/2 + 1
    assert fhhash_instance.sampler is not None
    assert fhhash_instance.M == 0.0
    assert fhhash_instance.b == 0.5
    assert fhhash_instance.m == 10
    assert fhhash_instance.max_blocks == 1600

def test_nns(fhhash_instance):
    fh_hash = fhhash_instance
    mock_data = np.random.rand(10, 5)  # 100 random data points
    query = FHQuery(query=np.random.rand(5), data=mock_data, top=5, limit=1000,l=2, dist=DistDP2H())
    fh_hash.build_index(data=mock_data)
    nns_result = fh_hash.nns(query)
    assert isinstance(nns_result, list)
    assert len(nns_result) <= query.top
    assert len(nns_result) <= query.limit
    assert all(isinstance(result, IdxVal) for result in nns_result)
