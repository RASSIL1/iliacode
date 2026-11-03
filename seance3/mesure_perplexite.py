# ============================================================
#  Séance 3 --- la mesure du premier moteur sur le dépôt, script fourni
#  Il réutilise la brique de la séance 2 (bpe.py, merges.json, donnees)
#  et vos trois fichiers de la séance 3.
#  Usage, depuis le dossier seance3 :   python mesure_perplexite.py
# ============================================================
import glob
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)

SEANCE2 = Path(__file__).resolve().parent.parent / "seance2"
if not (SEANCE2 / "merges.json").exists():
    sys.exit(f"Introuvable : {SEANCE2 / 'merges.json'}. Le dossier seance3 doit être "
             "à côté du dossier seance2, et merges.json doit y avoir été créé.")
sys.path.insert(0, str(SEANCE2))

from bpe import pretokeniser, encoder_mot, table_des_rangs      # la brique de la séance 2
from bigramme import compter, probas_du_texte                    # le moteur de la séance 3
from perplexite import perplexite                                # l'instrument de mesure
from hygiene import masquer_secrets                              # l'hygiène avant le comptage

VALEURS_DE_K = [1, 0.1, 0.01, 0.001]

# La partie 6 du TP écrit masquer_secrets. Tant qu'elle n'est pas faite, les
# lignes sont comptées telles quelles.
try:
    masquer_secrets("")
    nettoyer = masquer_secrets
except NotImplementedError:
    print("hygiène pas encore écrite (partie 6), les lignes sont comptées telles quelles")
    nettoyer = lambda ligne: ligne

# --- Le tokeniseur figé de la séance 2 ----------------------------------------
fusions = [tuple(p) for p in json.load(open(SEANCE2 / "merges.json", encoding="utf-8"))]
rangs = table_des_rangs(fusions)
TAILLE_V = 257 + len(fusions) + 1      # 256 octets, le marqueur, les fusions, et </s>
memoire = {}                           # un mot déjà encodé n'est pas réencodé


def en_tokens(ligne):
    """Une ligne de texte devient la liste de ses tokens BPE, blancs exclus."""
    tokens = []
    for mot in pretokeniser(nettoyer(ligne)):
        if mot.isspace():
            continue
        if mot not in memoire:
            memoire[mot] = encoder_mot(mot, rangs)
        tokens.extend(memoire[mot])
    return tokens


def lire_lignes(chemins):
    """Toutes les lignes non vides des fichiers, chacune devenue une séquence."""
    sequences = []
    for chemin in sorted(chemins):
        for ligne in open(chemin, encoding="utf-8").read().splitlines():
            if ligne.strip():
                sequences.append(en_tokens(ligne))
    return [s for s in sequences if s]


def fichiers(dossier):
    return glob.glob(str(SEANCE2 / "donnees" / dossier / "**" / "*.*"), recursive=True)


# --- Les trois blocs du protocole ---------------------------------------------
tout = lire_lignes(fichiers("entrainement"))
train = [s for i, s in enumerate(tout) if i % 10 != 9]      # neuf lignes sur dix
dev = [s for i, s in enumerate(tout) if i % 10 == 9]        # une ligne sur dix, pour régler k
test = {
    "code": lire_lignes(fichiers("test/code")),
    "français": lire_lignes([SEANCE2 / "donnees/test/tickets_fr.txt"]),
    "anglais": lire_lignes([SEANCE2 / "donnees/test/tickets_en.txt"]),
    "darija": lire_lignes([SEANCE2 / "donnees/test/tickets_darija.txt"]),
}

# --- Phase 1, compter sur le bloc train ----------------------------------------
paires, contextes = compter(train)
nb_tokens = sum(contextes.values())
print(f"train : {len(train)} lignes, {nb_tokens} tokens prédits, |V| = {TAILLE_V}")
print(f"bigrammes distincts vus : {len(paires)} sur {TAILLE_V ** 2} possibles, "
      f"soit {100 * len(paires) / TAILLE_V ** 2:.2f} %")

# --- Le comptage brut, sans lissage ----------------------------------------------
print()
print(f"comptage brut   train {perplexite(probas_du_texte(train, paires, contextes, TAILLE_V, 0)):8.1f}"
      f"   dev {perplexite(probas_du_texte(dev, paires, contextes, TAILLE_V, 0)):8.1f}")

# --- Le réglage de k sur le bloc dev ---------------------------------------------
print()
print("k        " + f"{'train':>10}{'dev':>10}")
meilleur_k, meilleure_pp = None, float("inf")
for k in VALEURS_DE_K:
    pp_train = perplexite(probas_du_texte(train, paires, contextes, TAILLE_V, k))
    pp_dev = perplexite(probas_du_texte(dev, paires, contextes, TAILLE_V, k))
    print(f"{k:<9}{pp_train:10.1f}{pp_dev:10.1f}")
    if pp_dev < meilleure_pp:
        meilleur_k, meilleure_pp = k, pp_dev
print(f"\nk retenu sur le dev : {meilleur_k}")

# --- Le chiffre final, une seule lecture du bloc test ---------------------------
print()
print("perplexité sur le test, k =", meilleur_k)
for nom, sequences in test.items():
    pp = perplexite(probas_du_texte(sequences, paires, contextes, TAILLE_V, meilleur_k))
    print(f"  {nom:<10}{pp:10.1f}")
print(f"  {'hasard':<10}{TAILLE_V:10.1f}   (un modèle qui tirerait au sort parmi |V| tokens)")