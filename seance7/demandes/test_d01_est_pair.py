from d01_est_pair import est_pair


def test_pairs():
    assert est_pair(0) is True
    assert est_pair(4) is True
    assert est_pair(-8) is True


def test_impairs():
    assert est_pair(1) is False
    assert est_pair(-7) is False
