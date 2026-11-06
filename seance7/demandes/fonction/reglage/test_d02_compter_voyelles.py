from d02_compter_voyelles import compter_voyelles


def test_minuscules():
    assert compter_voyelles("ticket") == 2
    assert compter_voyelles("rythme") == 2


def test_majuscules_et_vide():
    assert compter_voyelles("ILIACode") == 5
    assert compter_voyelles("") == 0
    assert compter_voyelles("xyz") == 1
