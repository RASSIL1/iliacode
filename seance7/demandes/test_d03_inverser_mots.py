from d03_inverser_mots import inverser_mots


def test_trois_mots():
    assert inverser_mots("le test échoue") == "échoue test le"


def test_un_mot_et_vide():
    assert inverser_mots("ticket") == "ticket"
    assert inverser_mots("") == ""
