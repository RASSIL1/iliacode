# ============================================================
#  Séance 7 --- le moteur écrit (déjà écrit)
#  Charge le moteur de la séance 1 d'IA Générative, lui donne une
#  demande et affiche ce qu'il écrit à la suite, une ou plusieurs fois,
#  avec le réglage choisi.
#  Usage : python ecrire.py "if user is"
#          python ecrire.py "if user is" --temperature 0.8 --p 0.95 --n 3
#          python ecrire.py --fichier demandes/d07_verifier_ticket.py --temperature 0.2
#          python ecrire.py "..." --max 120 --graine 4 --moteur Qwen/Qwen2.5-Coder-1.5B
#  --temperature 0 donne le décodage glouton. --arret "\n\n" coupe le
#  texte à la première ligne vide.
# ============================================================
import argparse
import sys
import time
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent / "genai_seance1"))

from moteur import MOTEUR_PAR_DEFAUT, charger_moteur  # noqa: E402
from generer import generer  # noqa: E402

CADRE = "-" * 72


def lire_arguments():
    p = argparse.ArgumentParser(description="Le moteur écrit à la suite d'une demande.")
    p.add_argument("demande", nargs="*", help="le texte à prolonger")
    p.add_argument("--fichier", help="un fichier dont le contenu est la demande")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--p", type=float, default=0.95, help="le noyau, part de probabilité gardée")
    p.add_argument("--n", type=int, default=1, help="le nombre de réponses écrites")
    p.add_argument("--max", type=int, default=80, help="le nombre maximal de tokens par réponse")
    p.add_argument("--graine", type=int, default=0, help="la graine du premier tirage")
    p.add_argument("--arret", default=None, help="une séquence d'arrêt, \\n pour une fin de ligne")
    p.add_argument("--moteur", default=MOTEUR_PAR_DEFAUT)
    a = p.parse_args()
    if a.fichier:
        a.texte = Path(a.fichier).read_text(encoding="utf-8")
    elif a.demande:
        a.texte = " ".join(a.demande)
    else:
        p.error("donnez une demande entre guillemets ou un --fichier")
    if a.arret:
        a.arret = a.arret.replace("\\n", "\n")
    return a


if __name__ == "__main__":
    a = lire_arguments()
    modele, tokeniseur = charger_moteur(a.moteur)
    n_tokens = len(tokeniseur(a.texte)["input_ids"])
    print(f"moteur {a.moteur}, température {a.temperature}, noyau {a.p}, "
          f"au plus {a.max} tokens, graine {a.graine}")
    print(f"demande, {n_tokens} tokens\n{CADRE}\n{a.texte.rstrip()}\n{CADRE}")
    for i in range(a.n):
        debut = time.time()
        texte = generer(modele, tokeniseur, a.texte, n_max=a.max,
                        temperature=a.temperature, p_noyau=a.p,
                        graine=a.graine + i, arret=a.arret)
        duree = time.time() - debut
        ecrits = len(tokeniseur(texte)["input_ids"])
        print(f"\nréponse {i + 1}, graine {a.graine + i}, {ecrits} tokens "
              f"en {duree:.1f} s, soit {1000 * duree / max(ecrits, 1):.0f} ms par token")
        print(CADRE)
        print(texte)
        print(CADRE)
