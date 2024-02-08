from functools import total_ordering

@total_ordering
class IdxVal:
    """
    Represents an ordered pair consisting of an index and a value, where ordering is primarily based on the value, 
    and secondarily (in case of equal values) on the index. This class supports comparison operations allowing 
    objects of this class to be sorted or compared directly.

    Attributes / Parameters for initialization:
        idx (int): The index part of the ordered pair.
        value (Any): The value part of the ordered pair, can be of any type that supports comparison operations.

    Methods:
        __eq__(self, other): Checks equality between this object and another based on their values and indices.
        __lt__(self, other): Defines the less-than ordering between this object and another.
        __repr__(self): Returns an unambiguous string representation of this object.
    """
    def __init__(self, idx, value):
        self.idx = idx
        self.value = value

    def __eq__(self, other):
        """
        Checks if this IdxVal object is equal to another, based on both the value and the index. Equality is 
        achieved when both the value and the index are equal.

        Parameters:
            other (IdxVal): The other IdxVal object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        return (self.value, self.idx) == (other.value, other.idx)

    def __lt__(self, other):
        """
        Compares this IdxVal object to another to establish a less-than relationship based on their values,
        and secondarily on their indices if their values are equal.

        Parameters:
            other (IdxVal): The other IdxVal object to compare with.

        Returns:
            bool: True if this object is considered less than the other, False otherwise.
        """
        if self.value == other.value:
            return self.idx < other.idx
        else:
            return self.value < other.value

    def __repr__(self):
        """
        Returns an unambiguous string representation of this IdxVal object, which is useful for debugging and logging.

        Returns:
            str: The string representation of the IdxVal object in the format "IdxVal{idx=X, value=Y}".
        """
        return f"IdxVal{{idx={self.idx}, value={self.value}}}"
