# ============================================================
#  IA Générative, séance 2 --- mesurer le service (déjà écrit)
#  Mesure le serveur lancé et met chaque mesure en face de la prédiction
#  du cours, calculée sur la fiche du L4 et la fiche du moteur.
#    temps du premier token   une demande à qui l'on ne fait écrire qu'un
#                             token, médiane de cinq essais
#    une demande seule        200 tokens écrits, temps par token
#    trente demandes          envoyées en même temps, 200 tokens chacune,
#                             débit total et durée jusqu'à la dernière
#  Les réponses sont forcées à 200 tokens, le token de fin est ignoré,
#  pour que chaque mesure porte sur le même nombre de pas.
#  Avec --attendre, le script attend seulement que le serveur soit prêt.
#  Usage : python mesurer_service.py --attendre
#          python mesurer_service.py
#          python mesurer_service.py --demandes 30 --tokens 200
# ============================================================
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))

import client  # noqa: E402

DEMANDE = ('def verifier_ticket(ticket):\n'
           '    """Rend True si le dictionnaire ticket est valide, c\'est-à-dire s\'il\n'
           '    possède une clé \'titre\' dont la valeur est une chaîne non vide et une\n'
           '    clé \'priorite\' dont la valeur est 1, 2 ou 3. Rend False sinon."""\n')
REGLAGE = {"fonction": {"temperature": 0.5, "p_noyau": 0.95}}
PREDIT = {"premier": 51, "par_token": 50.8, "debit_1": 19.7, "debit_30": 560,
          "par_demande_30": 18.7, "duree_30": 12}


def envoyer(corps):
    """Envoie un formulaire au serveur et rend la réponse et sa durée."""
    requete = urllib.request.Request(
        client.ADRESSE + "/v1/completions", data=json.dumps(corps).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ.get("CLE_ILIACODE", "")})
    debut = time.perf_counter()
    with urllib.request.urlopen(requete, timeout=600) as reponse:
        resultat = json.loads(reponse.read().decode("utf-8"))
    return resultat, time.perf_counter() - debut


def corps(n_tokens, graine):
    f = client.formulaire(DEMANDE, "fonction", REGLAGE, n_max=n_tokens, graine=graine)
    f["ignore_eos"] = True
    return f


def attendre(limite=900):
    debut = time.time()
    while time.time() - debut < limite:
        try:
            with urllib.request.urlopen(client.ADRESSE + "/health", timeout=5):
                print(f"serveur prêt après {time.time() - debut:.0f} s, à {client.ADRESSE}")
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(5)
    print(f"serveur toujours absent après {limite} s, lisez la fin de vllm.log")
    return False


def ligne(nom, predit, mesure, unite):
    print(f"  {nom:34s}{predit:>12}{mesure:>12}  {unite}")


def principal():
    p = argparse.ArgumentParser(description="Mesure le serveur et compare aux prédictions.")
    p.add_argument("--attendre", action="store_true", help="attend que le serveur soit prêt")
    p.add_argument("--demandes", type=int, default=30)
    p.add_argument("--tokens", type=int, default=200)
    args = p.parse_args()
    if args.attendre:
        attendre()
        return
    n, b = args.tokens, args.demandes

    envoyer(corps(8, 0))
    premiers = sorted(envoyer(corps(1, i))[1] for i in range(5))
    t1 = statistics.median(premiers)

    resultat, duree = envoyer(corps(n, 0))
    ecrits = resultat["usage"]["completion_tokens"]
    par_token = (duree - t1) / (ecrits - 1)

    debut = time.perf_counter()
    with ThreadPoolExecutor(max_workers=b) as groupe:
        reponses = list(groupe.map(lambda i: envoyer(corps(n, 100 + i)), range(b)))
    duree_b = time.perf_counter() - debut
    total = sum(r["usage"]["completion_tokens"] for r, _ in reponses)
    par_demande = statistics.mean(r["usage"]["completion_tokens"] / d for r, d in reponses)

    print(f"\n  {'grandeur':34s}{'prédiction':>12}{'mesure':>12}")
    print("  " + "-" * 64)
    ligne("temps du premier token", PREDIT["premier"], f"{1000 * t1:.0f}", "ms")
    ligne("temps par token, une demande", PREDIT["par_token"], f"{1000 * par_token:.1f}", "ms")
    ligne("tokens par seconde, une demande", PREDIT["debit_1"], f"{1 / par_token:.1f}", "tokens/s")
    ligne(f"débit total, {b} demandes", PREDIT["debit_30"], f"{total / duree_b:.0f}", "tokens/s")
    ligne(f"par demande, {b} demandes", PREDIT["par_demande_30"], f"{par_demande:.1f}", "tokens/s")
    ligne(f"{b} réponses complètes", PREDIT["duree_30"], f"{duree_b:.1f}", "s")
    print("  " + "-" * 64)
    print(f"  {ecrits} tokens par réponse, {total} tokens écrits en tout")


if __name__ == "__main__":
    principal()
