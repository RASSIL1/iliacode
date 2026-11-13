# ============================================================
#  IA Générative, séance 2 --- essayer le serveur (déjà écrit)
#  Envoie au serveur une demande de chacun des trois types d'ILIACode,
#  avec la fonction demander de client.py et le réglage du type lu dans
#  reglages.json de la séance 7, puis affiche ce que le moteur a écrit.
#  Avec --sans-cle, la demande part sans la clé d'accès, le serveur doit
#  la refuser.
#  Usage : python essayer.py fonction
#          python essayer.py tests
#          python essayer.py explication
#          python essayer.py fonction --sans-cle
# ============================================================
import argparse
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))

from client import demander  # noqa: E402

EN_TETE = ('def verifier_ticket(ticket):\n'
           '    """Rend True si le dictionnaire ticket possède une clé \'titre\' dont la\n'
           '    valeur est une chaîne non vide et une clé \'priorite\' dont la valeur est\n'
           '    1, 2 ou 3. Rend False sinon."""\n')
CORPS = ('    titre = ticket.get("titre")\n'
         '    return isinstance(titre, str) and titre != "" and ticket.get("priorite") in (1, 2, 3)\n')
DEMANDES = {
    "fonction": (EN_TETE, 160, None),
    "tests": (EN_TETE + CORPS + "\n\n\ndef test_", 200, None),
    "explication": (EN_TETE + CORPS + "\n\n# Explication de la fonction ci-dessus, en français :\n#",
                    80, "\n\n"),
}


def principal():
    p = argparse.ArgumentParser(description="Envoie une demande d'un type au serveur.")
    p.add_argument("type", choices=list(DEMANDES))
    p.add_argument("--sans-cle", action="store_true", help="envoie la demande sans la clé")
    p.add_argument("--graine", type=int, default=0)
    args = p.parse_args()

    texte, n_max, arret = DEMANDES[args.type]
    cle = "" if args.sans_cle else None
    print(f"demande de type {args.type}, {'sans' if args.sans_cle else 'avec'} la clé")
    print("  " + "-" * 64)
    try:
        suite = demander(texte, args.type, n_max=n_max, graine=args.graine, arret=arret, cle=cle)
    except PermissionError as erreur:
        print(f"  demande refusée par le serveur, {erreur}")
        return
    except ValueError as erreur:
        print(f"  {erreur}, remplissez reglages.json dans seance7 (TP N7, partie 3)")
        return
    debut = "def test_" if args.type == "tests" else ("#" if args.type == "explication" else "")
    for ligne in (texte.split("\n")[0] + "\n    ...\n" + debut + suite).splitlines():
        print("  " + ligne)
    print("  " + "-" * 64)


if __name__ == "__main__":
    principal()
