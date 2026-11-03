# ============================================================
#  Séance 3 --- les tests du premier moteur et de sa mesure
#  Lancer : pytest -v          (tous)      pytest -v -k p1   (la partie 1)
# ============================================================
import math
import re
import pytest
from bigramme import (DEBUT, FIN_SEQ, encadrer, compter, proba, predire,
                      proba_lissee, probas_du_texte)
from perplexite import surprise, entropie_croisee, perplexite
from hygiene import masquer_secrets, MASQUE

# Le mini-corpus de la séance, cinq lignes du dépôt découpées en mots
MINI = [["for", "ticket", "in", "tickets"],
        ["for", "ticket", "in", "queue"],
        ["for", "ticket", "in", "tickets"],
        ["for", "user", "in", "users"],
        ["if", "ticket", "is", "None"]]
TAILLE_V_MINI = 11          # dix mots distincts et le token de fin


def tables():
    """Phase 1 rejouée dans chaque test, les deux tables du mini-corpus."""
    return compter(MINI)


def produit(nombres):
    resultat = 1.0
    for x in nombres:
        resultat *= x
    return resultat


# --- Partie 1, compter ---------------------------------------------------------
def test_p1_encadrer():
    assert encadrer(["if", "ticket"]) == [DEBUT, "if", "ticket", FIN_SEQ]


def test_p1_compter_la_table():
    paires, contextes = tables()
    assert contextes[DEBUT] == 5 and contextes["in"] == 4 and contextes["tickets"] == 2
    assert paires[("in", "tickets")] == 2 and paires[("in", "queue")] == 1
    assert paires[("ticket", "queue")] == 0          # jamais vu
    assert contextes[FIN_SEQ] == 0                   # rien ne suit le token de fin


# --- Partie 2, prédire ---------------------------------------------------------
def test_p2_proba_lue_dans_la_table():
    assert proba("for", "ticket", *tables()) == 3 / 4
    assert proba(DEBUT, "for", *tables()) == 4 / 5
    assert proba("ticket", "queue", *tables()) == 0     # le zéro du jamais vu


def test_p2_predire_apres_in():
    assert predire("in", *tables()) == [("tickets", 0.5), ("queue", 0.25), ("users", 0.25)]


def test_p2_proba_d_une_ligne():
    vue = probas_du_texte([["for", "ticket", "in", "tickets"]], *tables(), TAILLE_V_MINI, 0)
    assert produit(vue) == pytest.approx(0.225)
    nouvelle = probas_du_texte([["for", "user", "in", "queue"]], *tables(), TAILLE_V_MINI, 0)
    assert produit(nouvelle) == pytest.approx(0.05)   # jamais écrite, pourtant possible
    cassee = probas_du_texte([["if", "user", "is", "None"]], *tables(), TAILLE_V_MINI, 0)
    assert produit(cassee) == 0                          # un seul zéro détruit le produit


# --- Partie 3, lisser ----------------------------------------------------------
def test_p3_lissage_supprime_le_zero():
    assert proba_lissee("if", "user", *tables(), TAILLE_V_MINI, 1) == pytest.approx(1 / 12)
    lissee = probas_du_texte([["if", "user", "is", "None"]], *tables(), TAILLE_V_MINI, 1)
    assert produit(lissee) > 0


def test_p3_lissage_reste_une_probabilite():
    vocabulaire = {w for ligne in MINI for w in ligne} | {FIN_SEQ}
    assert len(vocabulaire) == TAILLE_V_MINI
    for k in (1, 0.1):
        total = sum(proba_lissee("in", w, *tables(), TAILLE_V_MINI, k) for w in vocabulaire)
        assert total == pytest.approx(1)


# --- Partie 4, mesurer ---------------------------------------------------------
def test_p4_surprise():
    assert surprise(1) == 0
    assert surprise(0.5) == 1 and surprise(0.25) == 2 and surprise(0.125) == 3
    assert surprise(0) == math.inf


def test_p4_entropie_et_perplexite_de_l_exemple():
    probas = [0.5, 0.25, 0.5, 0.125]
    assert entropie_croisee(probas) == 1.75
    assert perplexite(probas) == pytest.approx(3.364, abs=0.001)


def test_p4_perplexite_d_un_modele_uniforme():
    assert perplexite([1 / 8] * 20) == pytest.approx(8)    # hésiter entre 8, perplexité 8
    assert perplexite([1.0] * 20) == 1                     # aucune hésitation


def test_p4_un_zero_rend_la_perplexite_infinie():
    cassee = probas_du_texte([["if", "user", "is", "None"]], *tables(), TAILLE_V_MINI, 0)
    assert perplexite(cassee) == math.inf


# --- Partie 5, mesurer un modèle -----------------------------------------------
def test_p5_perplexite_du_bigramme_lisse():
    vue = probas_du_texte([["for", "ticket", "in", "tickets"]], *tables(), TAILLE_V_MINI, 1)
    nouvelle = probas_du_texte([["if", "user", "is", "None"]], *tables(), TAILLE_V_MINI, 1)
    assert perplexite(vue) == pytest.approx(3.96, abs=0.01)
    assert perplexite(nouvelle) == pytest.approx(8.39, abs=0.01)
    assert perplexite(vue) < perplexite(nouvelle)          # le déjà-vu surprend moins


# --- Partie 6, protéger --------------------------------------------------------
LIGNE_SECRETE = 'API_KEY = "sk-4f9a2c71"'


def decouper(ligne):
    return re.findall(r"\w+|\S", ligne)


def test_p6_masquer_secrets():
    assert masquer_secrets(LIGNE_SECRETE) == 'API_KEY = "' + MASQUE + '"'
    assert masquer_secrets("mot_de_passe: 'azerty123'") == "mot_de_passe: '" + MASQUE + "'"
    assert masquer_secrets('titre = "Ticket ouvert"') == 'titre = "Ticket ouvert"'


def test_p6_le_modele_recite_puis_ne_recite_plus():
    depot = ["for ticket in tickets", LIGNE_SECRETE, "if ticket is None"]
    brut = compter([decouper(l) for l in depot])
    assert predire("-", *brut) == [("4f9a2c71", 1.0)]            # le secret est récité
    propre = compter([decouper(masquer_secrets(l)) for l in depot])
    assert predire("-", *propre) == []                           # il n'est plus dans la table
    assert all("4f9a2c71" not in paire for paire in propre[0])
