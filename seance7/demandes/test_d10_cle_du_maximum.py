from d10_cle_du_maximum import cle_du_maximum


def test_exemple():
    assert cle_du_maximum({"None": 0.487, "not": 0.174, "in": 0.062}) == "None"


def test_autres():
    assert cle_du_maximum({"a": 1}) == "a"
    assert cle_du_maximum({"x": -3, "y": -1, "z": -2}) == "y"
