# ============================================================
#  Séance 7 --- fixer le réglage d'un type de demande (déjà écrit)
#  ILIACode répond à trois types de demande, chacun avec son juge.
#    fonction     écrire le corps d'une fonction, jugé par ses tests
#    tests        écrire les tests d'une fonction, jugés par la version
#                 juste (ils doivent y passer) et par deux versions
#                 fautées, les mutants (ils doivent les faire échouer)
#    explication  expliquer un code, jugé par la lecture humaine
#  Pour fonction et tests, le script suit le chemin du cours sur le jeu
#  de réglage : mesure 1, pass@1 à T = 0,2 ; mesure 2, pass@1 à des T plus
#  hautes ; mesure 3, pass@k au meilleur réglage, pour le diagnostic.
#  Il vérifie ensuite le réglage retenu sur le jeu de test, et affiche la
#  ligne à recopier dans reglages.json.
#  Pour explication, il affiche les explications à deux températures, et
#  c'est vous qui jugez.
#  Usage : python regler.py fonction
#          python regler.py tests
#          python regler.py explication
#          python regler.py fonction --seuil 0.8 --n 5 --k 10
# ============================================================
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent / "genai_seance1"))

from moteur import MOTEUR_PAR_DEFAUT, charger_moteur  # noqa: E402
from generer import generer, pass_at_k  # noqa: E402

DEMANDES = ICI / "demandes"
T_DEPART, T_PLUS_HAUTES, P_NOYAU, DELAI = 0.2, (0.5, 0.8), 0.95, 30
N_MAX = {"fonction": 160, "tests": 200, "explication": 80}
SUITE_EXPLICATION = {
    "fr": "\n\n# Explication de la fonction ci-dessus, en français :\n#",
    "darija": "\n\n# Charh dyal had l fonction b darija :\n#",
}


# ------------------------------------------------------------
#  Les demandes et leurs juges
# ------------------------------------------------------------
def lire_fonction(jeu):
    """Les demandes du type fonction d'un jeu, (nom, en-tête, tests)."""
    dossier = DEMANDES / "fonction" / jeu
    demandes = []
    for chemin in sorted(dossier.glob("d*.py")):
        demandes.append((chemin.stem, chemin.read_text(encoding="utf-8"),
                         dossier / f"test_{chemin.name}"))
    return demandes


def lire_tests(jeu):
    """Les demandes du type tests d'un jeu, (nom, version juste, mutants)."""
    demandes = []
    for dossier in sorted((DEMANDES / "tests" / jeu).iterdir()):
        juste = (dossier / "juste.py").read_text(encoding="utf-8")
        mutants = [m.read_text(encoding="utf-8") for m in sorted(dossier.glob("mutant_*.py"))]
        demandes.append((dossier.name, juste, mutants))
    return demandes


def couper_fonction(texte):
    """Garde le corps de la fonction, les lignes vides ou indentées, et
    s'arrête à la première ligne qui repart à la marge."""
    corps = []
    for ligne in texte.split("\n"):
        if ligne.strip() and not ligne[0].isspace():
            break
        corps.append(ligne)
    return "\n".join(corps).rstrip() + "\n"


def couper_tests(texte):
    """Garde les fonctions de test, les lignes vides, indentées ou qui
    commencent par def test_, et s'arrête à toute autre ligne en marge."""
    garde = []
    for ligne in ("def test_" + texte).split("\n"):
        if ligne.strip() and not ligne[0].isspace() and not ligne.startswith("def test_"):
            break
        garde.append(ligne)
    return "\n".join(garde).rstrip() + "\n"


def lancer_pytest(fichiers):
    """Écrit les fichiers dans un dossier à part, lance pytest, et rend
    True si au moins un test a tourné et que tous passent."""
    dossier = Path(tempfile.mkdtemp(prefix="juge_"))
    try:
        for nom, contenu in fichiers.items():
            (dossier / nom).write_text(contenu, encoding="utf-8")
        resultat = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "."],
            cwd=dossier, capture_output=True, text=True, timeout=DELAI)
        return resultat.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        shutil.rmtree(dossier, ignore_errors=True)


def juger_fonction(demande, reponse):
    """Le corps écrit passe-t-il tous les tests de la fonction ?"""
    nom, en_tete, tests = demande
    return lancer_pytest({f"{nom}.py": en_tete + couper_fonction(reponse),
                          tests.name: tests.read_text(encoding="utf-8")})


def juger_tests(demande, reponse):
    """Les tests écrits passent-ils sur la version juste et font-ils
    échouer chaque mutant ?"""
    _, juste, mutants = demande
    tests = couper_tests(reponse)
    if not lancer_pytest({"test_reponse.py": juste + "\n\n" + tests}):
        return False
    return all(not lancer_pytest({"test_reponse.py": m + "\n\n" + tests}) for m in mutants)


def amorce(type_, demande):
    """Le début de texte donné au moteur, que sa suite doit compléter."""
    if type_ == "fonction":
        return demande[1]
    return demande[1] + "\n\n\ndef test_"


# ------------------------------------------------------------
#  Le chemin du cours
# ------------------------------------------------------------
def reussites(modele, tokeniseur, type_, demandes, temperature, n, graine0=0):
    """Pour chaque demande, la liste des verdicts de n réponses."""
    juger = juger_fonction if type_ == "fonction" else juger_tests
    verdicts = []
    for demande in demandes:
        v = []
        for i in range(n):
            reponse = generer(modele, tokeniseur, amorce(type_, demande), n_max=N_MAX[type_],
                              temperature=temperature, p_noyau=P_NOYAU, graine=graine0 + i)
            v.append(juger(demande, reponse))
        verdicts.append(v)
    return verdicts


def pass1(verdicts):
    """Le pass@1, la part moyenne des réponses justes par demande."""
    return sum(sum(v) / len(v) for v in verdicts) / len(verdicts)


def regler(modele, tokeniseur, type_, seuil, n, k):
    lire = lire_fonction if type_ == "fonction" else lire_tests
    reglage, test = lire("reglage"), lire("test")
    print(f"type {type_}, {len(reglage)} demandes de réglage, {len(test)} de test, "
          f"noyau p = {P_NOYAU}, seuil {seuil}\n")
    debut = time.time()

    v = reussites(modele, tokeniseur, type_, reglage, T_DEPART, 1)
    p = pass1(v)
    print(f"  mesure 1   pass@1 à T = {T_DEPART}            {p:.2f}")
    retenu, diagnostic = T_DEPART, None
    if p < seuil:
        meilleur, meilleurs_verdicts = None, None
        for T in T_PLUS_HAUTES:
            vt = reussites(modele, tokeniseur, type_, reglage, T, n, graine0=1)
            pt = pass1(vt)
            print(f"  mesure 2   pass@1 à T = {T}, {n} réponses   {pt:.2f}")
            if meilleur is None or pt > meilleur[1]:
                meilleur, meilleurs_verdicts = (T, pt), vt
        retenu = meilleur[0] if meilleur[1] > p else T_DEPART
        if max(p, meilleur[1]) < seuil:
            T3 = meilleur[0]
            plus = reussites(modele, tokeniseur, type_, reglage, T3, k - n, graine0=1 + n)
            vk = [a + b for a, b in zip(meilleurs_verdicts, plus)]
            pk = pass_at_k(vk)
            print(f"  mesure 3   pass@{k} à T = {T3}             {pk:.2f}")
            diagnostic = ("le moteur sait, aucun tirage unique n'est fiable"
                          if pk >= seuil else "le moteur ne sait pas")
    n_test = 1 if retenu == T_DEPART else n
    pv = pass1(reussites(modele, tokeniseur, type_, test, retenu, n_test, graine0=100))
    print(f"\n  réglage retenu, T = {retenu}, p = {P_NOYAU}")
    if diagnostic:
        print(f"  diagnostic : {diagnostic}")
    print(f"  vérification, pass@1 sur le jeu de test   {pv:.2f}")
    print(f"  durée {time.time() - debut:.0f} s")
    print("\nligne à recopier dans reglages.json :")
    print(f'  "{type_}": {json.dumps({"temperature": retenu, "p_noyau": P_NOYAU})}')


def expliquer(modele, tokeniseur, temperatures):
    """Affiche l'explication de chaque code à chaque température."""
    for chemin in sorted((DEMANDES / "explication").glob("e*.py")):
        langue = chemin.stem.split("_")[1]
        code = chemin.read_text(encoding="utf-8").rstrip("\n")
        print(f"=== {chemin.stem}, explication en {langue}")
        for T in temperatures:
            suite = generer(modele, tokeniseur, code + SUITE_EXPLICATION[langue],
                            n_max=N_MAX["explication"], temperature=T, p_noyau=P_NOYAU,
                            graine=0, arret="\n\n")
            lignes = ("#" + suite).split("\n")
            garde = [l for l in lignes if l.startswith("#")]
            print(f"  T = {T}")
            for l in garde[:6]:
                print("    " + l)
        print()
    print("Jugez chaque explication, juste et lisible ou non, puis recopiez dans")
    print('reglages.json la ligne  "explication": {"temperature": T, "p_noyau": 0.95}')
    print("avec la température que vous retenez.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fixe le réglage de décodage d'un type de demande.")
    p.add_argument("type", choices=["fonction", "tests", "explication"])
    p.add_argument("--seuil", type=float, default=0.8, help="pass@1 jugé élevé à partir de")
    p.add_argument("--n", type=int, default=5, help="réponses par demande aux T plus hautes")
    p.add_argument("--k", type=int, default=10, help="réponses par demande pour le pass@k")
    p.add_argument("--moteur", default=MOTEUR_PAR_DEFAUT)
    a = p.parse_args()
    modele, tokeniseur = charger_moteur(a.moteur)
    if a.type == "explication":
        expliquer(modele, tokeniseur, (T_DEPART, 0.7))
    else:
        regler(modele, tokeniseur, a.type, a.seuil, a.n, a.k)
