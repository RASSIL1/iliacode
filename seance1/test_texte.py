"""Les 13 tests de la séance 1. Lancer avec  pytest -v  dans ce dossier.
Chaque bloc de la séance fait passer ses tests au vert."""
from texte import lire, reparer, normaliser, tokeniser, mesurer


# ---------- Bloc Lire (2 tests) ----------
def test_lire_utf8(tmp_path):
    f = tmp_path / "ligne.txt"
    f.write_bytes(b"cr\xc3\xa9\xc3\xa9")            # six octets
    assert lire(f) == "créé"                         # quatre caractères


def test_lire_ancien_encodage(tmp_path):
    f = tmp_path / "ticket.txt"
    f.write_bytes(b"\xe3\xe1\xdd")                   # écrit en Windows-1256
    assert lire(f, regle="cp1256") == "ملف"


# ---------- Bloc Réparer (2 tests) ----------
def test_reparer_mojibake():
    assert reparer("# crÃ©Ã© par K. Alaoui") == "# créé par K. Alaoui"


def test_reparer_laisse_le_texte_sain():
    assert reparer("déjà vérifié, surface en m²") == "déjà vérifié, surface en m²"


# ---------- Bloc Normaliser (4 tests) ----------
def test_normaliser_double_ecriture():
    assert normaliser("cre\u0301e\u0301") == "créé"


def test_normaliser_compatibilite():
    assert normaliser("surface en m²") == "surface en m2"


def test_normaliser_arabe():
    assert normaliser("كَتَبَ") == "كتب"              # diacritiques
    assert normaliser("مـهـم") == "مهم"              # tatweel
    assert normaliser("٢٠٢٦") == "2026"              # chiffres


def test_normaliser_forme_nfc_garde_l_exposant():
    assert normaliser("10² mg", forme="NFC") == "10² mg"


# ---------- Bloc Tokeniser (4 tests) ----------
def test_tokeniser_ligne_de_test():
    assert tokeniser("assert prix != 3.14") == ["assert", "prix", "!=", "3.14"]


def test_tokeniser_ligne_de_code():
    assert tokeniser("def is_valid_email(adresse):") == [
        "def", "is_valid_email", "is", "valid", "email", "(", "adresse", ")", ":"]


def test_tokeniser_ticket():
    assert tokeniser("l'app t7bes, surface en m2 fausse") == [
        "l", "'", "app", "t7bes", ",", "surface", "en", "m2", "fausse"]


def test_tokeniser_protege_commentaires_et_chaines():
    assert tokeniser('x = "total_ht calcule"  # créé par K. Alaoui') == [
        "x", "=", '"total_ht calcule"', "# créé par K. Alaoui"]


# ---------- Bloc Mesurer (1 test) ----------
def test_mesurer():
    frequences = mesurer(["def", "if", "def", "return", "def"])
    assert frequences.most_common(1) == [("def", 3)]
