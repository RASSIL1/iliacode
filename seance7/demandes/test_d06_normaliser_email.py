from d06_normaliser_email import normaliser_email


def test_espaces_et_casse():
    assert normaliser_email("  Amine.Alami@Exemple.MA ") == "amine.alami@exemple.ma"


def test_deja_propre():
    assert normaliser_email("test@ensa.ma") == "test@ensa.ma"
