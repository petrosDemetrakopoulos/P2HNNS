from P2HNNS.utils.IdxVal import IdxVal

def test_initialization():
    """Test initialization of IdxVal objects."""
    iv = IdxVal(1, 10)
    assert iv.idx == 1
    assert iv.value == 10

def test_equality():
    """Test equality comparison of IdxVal objects."""
    iv1 = IdxVal(1, 10)
    iv2 = IdxVal(1, 10)
    iv3 = IdxVal(2, 10)
    iv4 = IdxVal(1, 11)

    assert iv1 == iv2
    assert iv1 != iv3
    assert iv1 != iv4

def test_less_than():
    """Test less-than comparison of IdxVal objects."""
    iv1 = IdxVal(1, 10)
    iv2 = IdxVal(2, 10)
    iv3 = IdxVal(1, 9)
    iv4 = IdxVal(2, 11)

    assert iv1 < iv2  # Secondary ordering based on index
    assert iv3 < iv1  # Primary ordering based on value
    assert not iv4 < iv1  # iv4 has a greater value

def test_repr():
    """Test the string representation of IdxVal objects."""
    iv = IdxVal(1, 10)
    expected_repr = "IdxVal{idx=1, value=10}"
    assert repr(iv) == expected_repr
