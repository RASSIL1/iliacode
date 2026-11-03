# ============================================================
#  Séance 2 --- les 10 tests du tokeniseur appris
#  Lancer : pytest -v          (tous)      pytest -v -k octets   (une partie)
# ============================================================
from collections import Counter
from bpe import (FIN, en_octets, hexa, pretokeniser, compter_paires, fusionner_mot,
                 entrainer, table_des_rangs, encoder_mot, encoder, decoder, fertilite)

# Le mini-corpus de la séance, les mots du champ « validation » avec leurs fréquences
MINI = Counter({"valid": 6, "valide": 3, "valider": 2, "email": 4})
TEXTE_MINI = " ".join(["valid"] * 6 + ["valide"] * 3 + ["valider"] * 2 + ["email"] * 4)
DOUZE_FUSIONS = ["va", "val", "vali", "valid", "valid" + FIN, "valide",
                 "em", "ema", "emai", "email", "email" + FIN, "valide" + FIN]

# Les trois textes de la production, jamais vus
LIGNE_DEV = "def calculerRistourneCNSS(salaire):"
LIGNE_TEST = "def test_ristourne_plafonnee():"
LIGNE_CLIENT = "l'app t7bes mnin kanseivi le contrat"


# --- Partie 1, la base des octets --------------------------------------------
def test_octets_ascii():
    assert en_octets("valid") == ["v", "a", "l", "i", "d", FIN]


def test_octets_arabe():
    assert en_octets("معطل") == [
        "Ù", "\x85",    # م  = octets D9 85
        "Ø", "¹",       # ع  = octets D8 B9
        "Ø", "·",       # ط  = octets D8 B7
        "Ù", "\x84",    # ل  = octets D9 84
        FIN,
    ]


# --- Partie 2, la pré-tokenisation --------------------------------------------
def test_pretokeniser_camelcase_et_tirets():
    assert pretokeniser(LIGNE_DEV) == ["def", " ", "calculer", "Ristourne", "CNSS",
                                       "(", "salaire", ")", ":"]
    assert pretokeniser(LIGNE_TEST) == ["def", " ", "test", "_", "ristourne", "_",
                                        "plafonnee", "(", ")", ":"]
    assert pretokeniser("t7bes 3.14 !=") == ["t7bes", " ", "3.14", " ", "!="]


# --- Partie 3, l'entraînement -------------------------------------------------
def test_compter_paires():
    corpus = {tuple(en_octets(mot)): freq for mot, freq in MINI.items()}
    paires = compter_paires(corpus)
    assert paires[("v", "a")] == 11
    assert paires[("e", "m")] == 4
    assert paires[("d", FIN)] == 6
    assert paires[("d", "e")] == 5


def test_fusionner_mot():
    assert fusionner_mot(("v", "a"), ("v", "a", "l", "i", "d", FIN)) == ("va", "l", "i", "d", FIN)
    assert fusionner_mot(("a", "b"), ("a", "b", "a", "b", "c")) == ("ab", "ab", "c")


def test_entrainer_douze_fusions():
    fusions, V = entrainer(MINI, 12)
    assert [a + b for a, b in fusions] == DOUZE_FUSIONS
    assert len(V) == 256 + 1 + 12


# --- Partie 4, l'encodage de l'inconnu ---------------------------------------
def test_encoder_mots_proches_du_corpus():
    fusions, _ = entrainer(MINI, 12)
    rangs = table_des_rangs(fusions)
    assert encoder_mot("valids", rangs) == ["valid", "s", FIN]
    assert encoder_mot("invalidation", rangs) == ["i", "n", "valid", "a", "t", "i", "o", "n", FIN]


def test_encoder_mot_jamais_vu_sans_inconnu():
    fusions, V = entrainer(MINI, 12)
    tokens = encoder(LIGNE_CLIENT, fusions)
    assert encoder_mot("t7bes", table_des_rangs(fusions)) == ["t", "7", "b", "e", "s", FIN]
    assert all(t in V for t in tokens)             # aucun token hors de V, donc aucun <UNK>


# --- Partie 5, la réversibilité -----------------------------------------------
def test_decoder_reversible():
    fusions, _ = entrainer(MINI, 12)
    for texte in [LIGNE_DEV, LIGNE_TEST, LIGNE_CLIENT, "  if valid:\n    return معطل"]:
        assert decoder(encoder(texte, fusions)) == texte


# --- Partie 6, la fertilité -----------------------------------------------------
def test_fertilite_mini():
    fusions, _ = entrainer(MINI, 12)
    assert abs(fertilite(TEXTE_MINI, []) - 97 / 15) < 1e-9         # niveau octet, 97 tokens pour 15 mots
    assert abs(fertilite(TEXTE_MINI, fusions) - 19 / 15) < 1e-9    # après douze fusions, 19 tokens
