# ============================================================
#  Séance 2 --- mesure de la fertilité en fonction de la taille du vocabulaire
#  Entraîne BPE sur donnees/entrainement, mesure sur donnees/test
#  (des textes que l'entraînement n'a jamais vus, comme en production).
#  Usage : python mesure_fertilite.py
# ============================================================
import glob
import time
import json
from collections import Counter
from bpe import pretokeniser, fertilite, encoder
from outils import entrainer_rapide

TAILLES = [500, 1000, 2000, 4000, 8000]       # tailles cibles de V
BASE_ET_MARQUEUR = 257                          # 256 octets + le marqueur de fin de mot


def lire_dossier(dossier):
    """Concatène tous les fichiers texte d'un dossier et de ses sous-dossiers."""
    textes = []
    for chemin in sorted(glob.glob(dossier + "/**/*.*", recursive=True)):
        textes.append(open(chemin, encoding="utf-8").read())
    return "".join(textes)


def lire(chemin):
    return open(chemin, encoding="utf-8").read()


# --- Phase 1, l'entraînement sur le corpus d'entraînement ------------------
entrainement = (lire_dossier("donnees/entrainement/code")
                + lire("donnees/entrainement/tickets_fr.txt")
                + lire("donnees/entrainement/tickets_en.txt")
                + lire("donnees/entrainement/tickets_darija.txt"))
mots = Counter(pretokeniser(entrainement))
print(f"corpus d'entraînement : {sum(mots.values())} mots, {len(mots)} mots distincts")

debut = time.time()
fusions, V = entrainer_rapide(mots, max(TAILLES) - BASE_ET_MARQUEUR)
print(f"{len(fusions)} fusions apprises en {time.time() - debut:.1f} s, |V| = {len(V)}")

# --- Phase 2, la mesure sur le corpus de test ------------------------------
test = {
    "code": lire_dossier("donnees/test/code"),
    "français": lire("donnees/test/tickets_fr.txt"),
    "anglais": lire("donnees/test/tickets_en.txt"),
    "darija": lire("donnees/test/tickets_darija.txt"),
}
print()
print("|V|      " + "".join(f"{nom:>10}" for nom in test))
print("octets   " + "".join(f"{fertilite(t, []):10.2f}" for t in test.values()))
for taille in TAILLES:
    f = fusions[:taille - BASE_ET_MARQUEUR]     # les premières fusions suffisent, l'ordre est le même
    print(f"{taille:<9}" + "".join(f"{fertilite(t, f):10.2f}" for t in test.values()))

# --- Sauvegarde du tokeniseur retenu ---------------------------------------
RETENU = 2000
json.dump([list(p) for p in fusions[:RETENU - BASE_ET_MARQUEUR]],
          open("merges.json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"\nliste ordonnée des {RETENU - BASE_ET_MARQUEUR} fusions du tokeniseur retenu "
      f"(|V| = {RETENU}) écrite dans merges.json")
