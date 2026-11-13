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
#  Avec --verifier, le script contrôle avant le lancement que les poids,
#  la clé et la carte sont en place, et qu'aucun serveur ne tourne déjà.
#  Avec --attendre, il attend seulement que le serveur soit prêt, et dit
#  pourquoi s'il s'est arrêté.
#  Usage : python mesurer_service.py --verifier
#          python mesurer_service.py --attendre
#          python mesurer_service.py
#          python mesurer_service.py --demandes 30 --tokens 200
# ============================================================
import argparse
import json
import os
import statistics
import subprocess
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
POIDS = Path(client.NOM_MOTEUR)
JOURNAL = Path("vllm.log")
ARRETER = '!pkill -f "vllm serve"'
CAUSES = (
    ("Repo id must be", "le dossier des poids était absent au moment du lancement"),
    ("out of memory", "la carte manque de mémoire, vérifiez le L4 et qu'aucun autre serveur ne tourne"),
    ("unrecognized arguments", "une option est refusée, relisez la commande de l'étape 1.8"),
    ("address already in use", f"un serveur tournait déjà, arrêtez-le avec {ARRETER}"),
)
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


def serveur_lance():
    """Vrai si un processus vllm serve tourne sur la machine."""
    for fichier in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            mots = fichier.read_bytes().split(b"\0")
        except OSError:
            continue
        if b"serve" in mots and any(m.endswith(b"vllm") for m in mots):
            return True
    return False


def poids_complets():
    """Vrai si le dossier des poids contient la fiche du moteur et tous
    les fichiers de poids que son sommaire annonce."""
    sommaire = POIDS / "model.safetensors.index.json"
    if not (POIDS / "config.json").exists() or not sommaire.exists():
        return False
    fichiers = set(json.loads(sommaire.read_text(encoding="utf-8"))["weight_map"].values())
    return all((POIDS / f).exists() for f in fichiers)


def carte_libre(minimum=20000):
    """Vrai si la carte annonce au moins minimum Mio de mémoire libre."""
    try:
        sortie = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout
        return int(sortie.split()[0]) >= minimum
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return False


def verifier():
    points = (
        (f"poids complets dans {POIDS}", poids_complets(), "refaites l'étape 1.6"),
        ("clé d'accès CLE_ILIACODE", bool(os.environ.get("CLE_ILIACODE")), "relancez l'étape 1.7"),
        ("carte avec 20 Go libres", carte_libre(), f"choisissez L4 GPU, ou {ARRETER}"),
        ("aucun serveur déjà lancé", not serveur_lance(),
         f"il tourne déjà, passez à l'étape 1.9, ou {ARRETER}"),
    )
    for nom, bon, remede in points:
        print(f"  {'ok' if bon else 'À CORRIGER':12s}{nom}" + ("" if bon else f", {remede}"))
    pret = all(bon for _, bon, _ in points)
    print("  lancement du serveur" if pret else "  serveur non lancé, corrigez puis relancez la cellule")
    return pret


def expliquer_arret():
    print("le serveur ne tourne pas, il s'est arrêté ou n'a pas été lancé")
    texte = JOURNAL.read_text(encoding="utf-8", errors="replace") if JOURNAL.exists() else ""
    for trace, cause in CAUSES:
        if trace in texte:
            print(f"  cause, {cause}")
    for fin in texte.strip().splitlines()[-3:]:
        print("  " + fin[:150])
    print("  corrigez, puis relancez l'étape 1.8 et cette cellule")
    return False


def cle_acceptee():
    requete = urllib.request.Request(
        client.ADRESSE + "/v1/models",
        headers={"Authorization": "Bearer " + os.environ.get("CLE_ILIACODE", "")})
    try:
        with urllib.request.urlopen(requete, timeout=30):
            return True
    except urllib.error.HTTPError:
        print(f"  la clé du carnet n'est pas celle du serveur, {ARRETER}, puis étapes 1.7 et 1.8")
        return False


def attendre(limite=900):
    debut = time.time()
    while time.time() - debut < limite:
        try:
            with urllib.request.urlopen(client.ADRESSE + "/health", timeout=5):
                print(f"serveur prêt après {time.time() - debut:.0f} s, à {client.ADRESSE}")
                return cle_acceptee()
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if time.time() - debut > 10 and not serveur_lance():
                return expliquer_arret()
            time.sleep(5)
    print(f"serveur toujours absent après {limite} s, lisez la fin de vllm.log")
    return False


def ligne(nom, predit, mesure, unite):
    print(f"  {nom:34s}{predit:>12}{mesure:>12}  {unite}")


def principal():
    p = argparse.ArgumentParser(description="Mesure le serveur et compare aux prédictions.")
    p.add_argument("--verifier", action="store_true", help="contrôle avant le lancement")
    p.add_argument("--attendre", action="store_true", help="attend que le serveur soit prêt")
    p.add_argument("--demandes", type=int, default=30)
    p.add_argument("--tokens", type=int, default=200)
    args = p.parse_args()
    if args.verifier:
        sys.exit(0 if verifier() else 1)
    if args.attendre:
        sys.exit(0 if attendre() else 1)
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
