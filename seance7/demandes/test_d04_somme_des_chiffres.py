from d04_somme_des_chiffres import somme_des_chiffres


def test_exemples():
    assert somme_des_chiffres(2048) == 14
    assert somme_des_chiffres(7) == 7
    assert somme_des_chiffres(0) == 0
    assert somme_des_chiffres(999999) == 54
