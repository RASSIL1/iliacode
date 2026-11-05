from d08_fusionner_tries import fusionner_tries


def test_exemple():
    assert fusionner_tries([1, 4, 9], [2, 3, 10]) == [1, 2, 3, 4, 9, 10]


def test_listes_vides_et_doublons():
    assert fusionner_tries([], [1, 2]) == [1, 2]
    assert fusionner_tries([3], []) == [3]
    assert fusionner_tries([1, 1], [1]) == [1, 1, 1]
