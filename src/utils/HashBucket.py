from typing import Callable, List, Dict
from collections import defaultdict
import numpy as np
from tqdm import tqdm

class HashBucket:
    """
    Implements a hash bucket system for efficiently managing and querying data using multiple hash tables. 
    The class supports inserting data items and searching for items based on a query code, utilizing bitwise 
    operations for hash code generation and management.

    Attributes:
        l (int): The number of hash tables (buckets) to be used.
        mask (int): A bitmask used to ensure hash codes fit within a certain range.
        buckets (List[defaultdict[list]]): A list of hash tables for storing data items, where each hash table 
                                           is implemented as a defaultdict(list) for flexibility and efficiency.

    Parameters:
        n (int): The expected size of the dataset, used to determine the range of hash codes.
        l (int): The number of hash tables (buckets) to create for data storage.

    Methods:
        insert(self, key: int, dcode: np.array): Inserts a data item into the hash buckets.
        search(self, qcode, limit, consumer): Searches for data items that match a query code.
        get_or_empty(self, bucket, hashcode32): Retrieves items from a bucket based on a hash code, 
                                                 returning an empty list if no items are found.
        swap(self, as_list, bs_list, aidx, bidx): Swaps items between two lists at specified indices.
        get_or_insert(self, map_obj, idx): Retrieves or initializes a list in a map for a given index.
        is_empty(self): Returns True if no data has been added to the HashBucket, otherwise False.
    """
    def __init__(self, n: int, l: int):
        """
        Initializes the HashBucket with the specified number of hash tables and configures the bitmask 
        based on the expected size of the dataset.

        Parameters:
            n (int): The expected size of the dataset.
            l (int): The number of hash tables to use.
        """
        self.l = l
        max_val = 1
        while max_val < n:
            max_val <<= 1
        max_val -= 1
        self.mask = max_val
        self.buckets = [defaultdict(list) for _ in range(l)]

    def insert(self, key: int, dcode: np.array):
        """
        Inserts a data item identified by a key into the hash buckets using the provided hash codes.

        Parameters:
            key (int): The identifier for the data item to be inserted.
            dcode (np.array): An array of hash codes for the data item, one for each hash table.
        """
        for j in range(self.l):
            hashcode32 = dcode[j] & self.mask
            found = self.buckets[j][hashcode32]
            found.append(key)
            if len(found) > 1:
                n = np.random.randint(0, len(found))
                found[n], found[-1] = found[-1], found[n]

    def search(self, qcode, limit: int, consumer: Callable) -> Dict:
        """
        Searches for data items that match the query code, using a consumer function to process each found item.

        Parameters:
            qcode: The query code used to search for matching data items.
            limit (int): The maximum number of candidate items to find before stopping the search.
            consumer (Callable): A function to be called for each data item found during the search.

        Returns:
            defaultdict[int, int]: A dictionary with keys as item identifiers and values as the count of matches.
        """
        candidate = defaultdict(int)
        print("-- Searching... --")
        for j in tqdm(range(self.l)):
            hashcode32 = qcode[j] & self.mask
            bucket = self.get_or_empty(self.buckets[j], hashcode32)

            for key in bucket:
                cnt = candidate[key]
                if cnt == 0:
                    consumer(key)
                candidate[key] = cnt + 1
                if limit <= len(candidate):
                    return candidate

        return candidate

    def get_or_empty(self, bucket: Dict, hashcode32: int) -> List:
        """
        Retrieves the list of items associated with a hash code from a bucket, or an empty list if no items are found.

        Parameters:
            bucket (defaultdict[list]): The bucket from which to retrieve items.
            hashcode32 (int): The hash code used to look up items in the bucket.

        Returns:
            list: The list of items associated with the hash code, or an empty list if none are found.
        """
        return bucket.get(hashcode32, [])

    def swap(self, as_list: List, bs_list: List, aidx: int, bidx: int):
        """
        Swaps items between two lists at specified indices.

        Parameters:
            as_list (list): The first list from which an item will be swapped.
            bs_list (list): The second list with which an item from the first list will be swapped.
            aidx (int): The index in the first list of the item to swap.
            bidx (int): The index in the second list of the item to swap.
        """
        if aidx != bidx:
            as_list[aidx], bs_list[bidx] = bs_list[bidx], as_list[aidx]

    def get_or_insert(self, map_obj: Dict, idx: int) -> List:
        """
        Retrieves a list from a map object for a given index. If the index does not exist, a new list is initialized.

        Parameters:
            map_obj (defaultdict[list]): The map from which to retrieve or initialize the list.
            idx (int): The index for which to retrieve or initialize the list.

        Returns:
            list: The list associated with the given index in the map object.
        """
        if idx not in map_obj:
            map_obj[idx] = []
        return map_obj[idx]

    def is_empty(self) -> bool:
        """
        Checks if the HashBucket is empty.

        Returns:
            bool: True if no data has been added to any of the hash tables, otherwise False.
        """
        return all(len(bucket) == 0 for bucket in self.buckets)
