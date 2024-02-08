# pylint: disable=W0621, C0116, C0115, W0201
import pytest
import numpy as np
from src.utils.HashBucket import HashBucket

class TestHashBucket:
    def setup_method(self):
        self.n = 1024
        self.l = 10
        self.hash_bucket = HashBucket(n=self.n, l=self.l)

    def test_initialization(self):
        assert self.hash_bucket.l == self.l
        assert len(self.hash_bucket.buckets) == self.l
        assert self.hash_bucket.mask == (1 << (self.n-1).bit_length()) - 1

    def test_insert_and_search(self):
        key = 1
        dcode = np.array([np.random.randint(0, 1024) for _ in range(self.l)])
        self.hash_bucket.insert(key, dcode)
        # Use a simple consumer function that does nothing for testing
        consumer = lambda x: None
        candidates = self.hash_bucket.search(dcode, limit=1, consumer=consumer)
        assert key in candidates
        assert candidates[key] >= 1

    def test_get_or_empty(self):
        bucket = self.hash_bucket.buckets[0]
        hashcode32 = np.random.randint(0, 1024) & self.hash_bucket.mask
        items = self.hash_bucket.get_or_empty(bucket, hashcode32)
        assert isinstance(items, list)

    def test_swap(self):
        list1 = [1, 2, 3]
        list2 = [4, 5, 6]
        self.hash_bucket.swap(list1, list2, 1, 2)
        assert list1 == [1, 6, 3]
        assert list2 == [4, 5, 2]

    def test_get_or_insert(self):
        map_obj = {}
        idx = 0
        returned_list = self.hash_bucket.get_or_insert(map_obj, idx)
        assert returned_list == []
        assert idx in map_obj

@pytest.fixture
def hash_bucket_fixture():
    n = 2048
    l = 5
    hash_bucket = HashBucket(n=n, l=l)
    return hash_bucket

def test_with_fixture(hash_bucket_fixture):
    assert hash_bucket_fixture.l == 5
    assert len(hash_bucket_fixture.buckets) == 5

def test_hashbucket_is_empty_when_new():
    hash_bucket = HashBucket(n=100, l=5)
    assert hash_bucket.is_empty() is True, "Expected HashBucket to be empty when newly created."

def test_hashbucket_is_not_empty_after_insert():
    hash_bucket = HashBucket(n=100, l=5)
    # Creating a dummy hash code array to insert.
    dcode = np.array([1, 2, 3, 4, 5])
    hash_bucket.insert(key=1, dcode=dcode)
    assert hash_bucket.is_empty() is False, "Expected HashBucket to not be empty after inserting an item."
