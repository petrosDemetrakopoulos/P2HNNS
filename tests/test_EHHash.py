# pylint: disable=W0621, C0116
from unittest.mock import MagicMock
import numpy as np
import pytest
from src.methods import EHHash
from src import Query, IdxVal
from src.distance_functions import DistAbsDot


@pytest.fixture
def setup_ehhash():
    # Fixture to initialize EHHash object before each test
    d, m, l = 10, 5, 3  # Example dimensions and parameters
    ehhash = EHHash(d, m, l)
    return ehhash

def test_initialization():
    d, m, l = 10, 5, 3
    ehhash = EHHash(d, m, l)
    assert ehhash.m == m
    assert ehhash.l == l
    assert ehhash.randv.shape == (m * l * d * d,)

def test_hash_data(setup_ehhash):
    data = np.random.rand(10)  # Example data
    sigs = setup_ehhash.hash_data(data)
    assert len(sigs) == setup_ehhash.l
    assert isinstance(sigs, np.ndarray)
    assert sigs.dtype == int

def test_hash_query(setup_ehhash):
    query = np.random.rand(10)  # Example query
    sigs = setup_ehhash.hash_query(query)
    assert len(sigs) == setup_ehhash.l
    assert isinstance(sigs, np.ndarray)
    assert sigs.dtype == int

def test_build_index(setup_ehhash):
    data = np.random.rand(5, 10)  # Generate random dataset with 5 samples and 10 features each

    # Mock `insert` method of the `buckets` attribute to verify it's called correctly
    setup_ehhash.buckets = MagicMock()

    setup_ehhash.build_index(data)

    # Verify that `insert` was called 5 times, once for each data point
    assert setup_ehhash.buckets.insert.call_count == 5

def test_nns(setup_ehhash):
    eh_hash = setup_ehhash
    mock_data = np.random.rand(100, 10)  # 100 random data points
    query = Query(query=np.random.rand(10), data=mock_data, top=5, limit=1000, dist=DistAbsDot())
    eh_hash.build_index(data=mock_data)
    nns_result = eh_hash.nns(query)
    assert isinstance(nns_result, list)
    assert len(nns_result) <= query.top
    assert len(nns_result) <= query.limit
    assert all(isinstance(result, IdxVal) for result in nns_result)
