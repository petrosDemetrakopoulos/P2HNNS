# pylint: disable=W0621, C0116
import numpy as np
import pytest
from P2HNNS.utils import Query
from P2HNNS.methods import BHHash
from P2HNNS.utils.distance_functions import DistDP2H

@pytest.fixture
def bhhash_setup():
    d = 10  # Dimensionality of the data.
    m = 5   # Number of hyperplanes per hash function.
    l = 3   # Number of hash functions.
    bhhash = BHHash(d=d, m=m, l=l)
    return bhhash

def test_initialization(bhhash_setup):
    bhhash = bhhash_setup
    assert bhhash.m == 5
    assert bhhash.l == 3
    assert hasattr(bhhash, 'randu')
    assert hasattr(bhhash, 'randv')
    assert bhhash.randu.shape == (bhhash.m * bhhash.l * 10,)
    assert bhhash.randv.shape == (bhhash.m * bhhash.l * 10,)

def test_hash_data(bhhash_setup):
    bhhash = bhhash_setup
    data = np.random.rand(10)
    hash_codes = bhhash.hash_data(data)
    assert len(hash_codes) == bhhash.l
    assert np.issubdtype(hash_codes.dtype, np.integer)

def test_hash_query(bhhash_setup):
    bhhash = bhhash_setup
    query = np.random.rand(10)
    hash_codes = bhhash.hash_query(query)
    assert len(hash_codes) == bhhash.l
    assert np.issubdtype(hash_codes.dtype, np.integer)

def test_build_index(bhhash_setup):
    bhhash = bhhash_setup
    data = np.random.rand(10, 10)
    bhhash.build_index(data)
    assert bhhash.buckets.is_empty() is False

def test_nns(bhhash_setup):
    bhhash = bhhash_setup
    data = np.random.rand(20, 10)
    query = np.random.rand(10)
    bhhash.build_index(data)
    param = Query(query=query, data=data, top=5, limit=10, dist=DistDP2H())
    results = bhhash.nns(param)
    assert len(results) <= 5
