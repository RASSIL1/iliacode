# ============================================================
#  Séance 7 --- la mesure du code écrit (déjà écrit)
#  Pour chaque demande du dossier demandes (un en-tête de fonction avec sa
#  docstring, et son fichier de tests), fait écrire le corps de la
#  fonction par le moteur, une fois à basse température puis k fois à
#  haute température, exécute les tests sur chaque réponse dans un dossier
#  à part, et affiche le pass@1 et le pass@k.
#  Usage : python mesurer_code.py                (k = 10)
#          python mesurer_code.py --k 5
#          python mesurer_code.py --seulement d07_verifier_ticket
#          python mesurer_code.py --moteur Qwen/Qwen2.5-Coder-1.5B
# ============================================================
import argparse
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
T_SURE, T_VARIEE, P_NOYAU, N_MAX, DELAI = 0.2, 0.8, 0.95, 160, 30


def lire_demandes(seulement=None):
    """Les couples (nom, en-tête, fichier de tests) du dossier demandes."""
    couples = []
    for chemin in sorted(DEMANDES.glob("d*.py")):
        if chemin.name.startswith("test_"):
            continue
        if seulement and seulement not in chemin.stem:
            continue
        tests = DEMANDES / f"test_{chemin.name}"
        if not tests.exists():
            sys.exit(f"Introuvable : {tests}")
        couples.append((chemin.stem, chemin.read_text(encoding="utf-8"), tests))
    if not couples:
        sys.exit(f"Aucune demande dans {DEMANDES}")
    return couples


def couper_fonction(texte):
    """Garde le corps de la fonction, c'est-à-dire les lignes vides ou
    indentées, et s'arrête à la première ligne qui repart à la marge."""
    corps = []
    for ligne in texte.split("\n"):
        if ligne.strip() and not ligne[0].isspace():
            break
        corps.append(ligne)
    return "\n".join(corps).rstrip() + "\n"


def executer_tests(nom, en_tete, corps, tests):
    """Écrit la fonction complète et ses tests dans un dossier temporaire,
    lance pytest, et rend True si tous les tests passent."""
    dossier = Path(tempfile.mkdtemp(prefix="mesure_"))
    try:
        (dossier / f"{nom}.py").write_text(en_tete + corps, encoding="utf-8")
        shutil.copy(tests, dossier / tests.name)
        resultat = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", tests.name],
            cwd=dossier, capture_output=True, text=True, timeout=DELAI)
        return resultat.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        shutil.rmtree(dossier, ignore_errors=True)


def mesurer(modele, tokeniseur, couples, k):
    print(f"{len(couples)} demandes, moteur {modele.config._name_or_path}, "
          f"une réponse à T = {T_SURE} puis {k} réponses à T = {T_VARIEE}, noyau {P_NOYAU}\n")
    print(f"  {'demande':26s}{'pass@1':>8s}{'réussites sur ' + str(k):>18s}{'temps':>8s}")
    verdicts_1, verdicts_k = [], []
    for nom, en_tete, tests in couples:
        debut = time.time()
        corps = couper_fonction(generer(modele, tokeniseur, en_tete, n_max=N_MAX,
                                        temperature=T_SURE, p_noyau=P_NOYAU, graine=0))
        v1 = [executer_tests(nom, en_tete, corps, tests)]
        vk = []
        for i in range(k):
            corps = couper_fonction(generer(modele, tokeniseur, en_tete, n_max=N_MAX,
                                            temperature=T_VARIEE, p_noyau=P_NOYAU, graine=1 + i))
            vk.append(executer_tests(nom, en_tete, corps, tests))
        verdicts_1.append(v1)
        verdicts_k.append(vk)
        print(f"  {nom:26s}{'oui' if v1[0] else 'non':>8s}{sum(vk):>18d}{time.time() - debut:>7.0f} s")
    n = len(couples)
    print(f"\n  pass@1  = {sum(any(v) for v in verdicts_1)}/{n} = {pass_at_k(verdicts_1):.2f}")
    print(f"  pass@{k} = {sum(any(v) for v in verdicts_k)}/{n} = {pass_at_k(verdicts_k):.2f}")
    print("\nle pass@1 note une seule réponse sûre, le pass@k note la chance qu'une "
          "réponse au moins passe quand les tests tranchent")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Le pass@k du moteur sur les demandes du dossier.")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seulement", default=None, help="un morceau du nom d'une demande")
    p.add_argument("--moteur", default=MOTEUR_PAR_DEFAUT)
    a = p.parse_args()
    modele, tokeniseur = charger_moteur(a.moteur)
    mesurer(modele, tokeniseur, lire_demandes(a.seulement), a.k)
