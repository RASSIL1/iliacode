# ============================================================
#  Séance 3 --- le premier moteur, un modèle bigramme par comptage
#  Phase 1 : compter (une fois, hors ligne), l'artefact figé est la table
#  Phase 2 : prédire (à chaque demande), une lecture de la table
#  Ce moteur sera remplacé par le Transformer, le contrat qu'il remplit
#  (une histoire entre, une probabilité par token suivant sort) restera.
# ============================================================
from collections import Counter

DEBUT = "<s>"      # token de bord, sert de contexte au premier token, jamais prédit
FIN_SEQ = "</s>"   # token de bord, prédit comme les autres, dit au modèle de s'arrêter


# --- Partie 1, compter (phase 1) ---------------------------------------------
def encadrer(tokens):
    """Ajoute les deux tokens de bord autour d'une séquence de tokens."""
    return [DEBUT] + list(tokens) + [FIN_SEQ]


def compter(sequences):
    """Phase 1. sequences = liste de listes de tokens. Renvoie deux tables,
    paires[(v, w)] = nombre de fois où w suit v, contextes[v] = nombre de
    fois où v est suivi d'un token."""
    paires, contextes = Counter(), Counter()
    for tokens in sequences:
        bordee = encadrer(tokens)
        for v, w in zip(bordee, bordee[1:]):
            paires[(v, w)] += 1
            contextes[v] += 1
    return paires, contextes


# --- Partie 2, prédire (phase 2) ---------------------------------------------
def proba(v, w, paires, contextes):
    """Phase 2. La probabilité que w suive v, lue dans la table, C(v,w) / C(v).
    Un contexte jamais vu renvoie 0."""
    if contextes[v] == 0:
        return 0.0
    return paires[(v, w)] / contextes[v]


def predire(v, paires, contextes, combien=5):
    """Les candidats les plus probables après v, sous la forme d'une liste
    de couples (token, probabilité), du plus probable au moins probable."""
    candidats = [(w, proba(v, w, paires, contextes))
                 for (contexte, w) in paires if contexte == v]
    candidats.sort(key=lambda couple: (-couple[1], couple[0]))
    return candidats[:combien]


# --- Partie 3, lisser ----------------------------------------------------------
def proba_lissee(v, w, paires, contextes, taille_V, k):
    """Le lissage additif, (C(v,w) + k) / (C(v) + k x taille_V). Avec k > 0,
    aucune probabilité ne vaut zéro. taille_V compte les tokens prédictibles,
    ceux du tokeniseur et le token de fin."""
    return (paires[(v, w)] + k) / (contextes[v] + k * taille_V)


# --- Partie 5, les probabilités données à un texte -----------------------------
def probas_du_texte(sequences, paires, contextes, taille_V, k):
    """La liste des probabilités que le modèle donne aux tokens réellement
    écrits dans sequences, le token de fin compris. C'est l'entrée de
    perplexite(). Avec k = 0, c'est le comptage brut."""
    probas = []
    for tokens in sequences:
        bordee = encadrer(tokens)
        for v, w in zip(bordee, bordee[1:]):
            if k == 0:
                probas.append(proba(v, w, paires, contextes))
            else:
                probas.append(proba_lissee(v, w, paires, contextes, taille_V, k))
    return probas
