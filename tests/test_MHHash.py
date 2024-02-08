# pylint: disable=W0621, C0116
from unittest.mock import patch
import pytest
import numpy as np
from src.methods import MHHash
from src.utils.HashBucket import HashBucket
from src.utils.distance_functions import DistCos
from src.utils import Query

@pytest.fixture
def setup_mhhash():
    dimension = 128
    m = 5
    l = 10
    M = 2
    mhhash = MHHash(dimension, m, l, M)
    return mhhash

def test_initialization(setup_mhhash):
    mhhash = setup_mhhash
    assert mhhash.m == 5
    assert mhhash.l == 10
    assert mhhash.M == 2
    assert len(mhhash.randv) == 5 * 10 * 2 * 128
    assert isinstance(mhhash.buckets, HashBucket)

def test_hash_data(setup_mhhash):
    data = np.random.rand(128)
    mhhash = setup_mhhash
    sigs = mhhash.hash_data(data)
    assert len(sigs) == mhhash.l

def test_hash_query(setup_mhhash):
    query = np.random.rand(128)
    mhhash = setup_mhhash
    sigs = mhhash.hash_query(query)
    assert len(sigs) == mhhash.l

def test_build_index(setup_mhhash):
    mhhash = setup_mhhash
    data = np.random.rand(20, 128)  # 20 data points
    with patch.object(mhhash.buckets, 'insert', autospec=True) as mock_insert:
        mhhash.build_index(data)
        assert mock_insert.call_count == 20

def test_nns(setup_mhhash):
    mhhash = setup_mhhash
    rand_data = np.random.rand(20, 128)  # 20 data points
    mhhash.build_index(rand_data)
    query = Query(query=np.random.rand(128), data=rand_data, top=5, limit=10, dist=DistCos())
    # Assume buckets are already populated
    with patch.object(mhhash.buckets, 'search', autospec=True) as mock_search:
        mhhash.nns(query)
        mock_search.assert_called_once()
