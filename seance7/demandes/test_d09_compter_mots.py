from d09_compter_mots import compter_mots


def test_exemple():
    assert compter_mots("a b a") == {"a": 2, "b": 1}


def test_vide_et_repetitions():
    assert compter_mots("") == {}
    assert compter_mots("test test test todo") == {"test": 3, "todo": 1}
