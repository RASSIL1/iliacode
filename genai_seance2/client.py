# ============================================================
#  IA Générative, séance 2 --- la brique durable, parler au serveur
#  Deux fonctions à écrire (partie 2 du TP G2), chacune se termine par
#  une ligne raise NotImplementedError à remplacer par le vrai code.
#  Toutes les briques suivantes d'ILIACode, la recherche, l'agent et
#  l'évaluation, appellent demander pour faire écrire le moteur.
#  Le serveur est vLLM, lancé sur le GPU. La fonction lui envoie un
#  formulaire au format des services publics de génération, à l'adresse
#  /v1/completions, avec la clé d'accès dans l'en-tête.
#  L'adresse et la clé se lisent dans deux variables d'environnement,
#  ADRESSE_ILIACODE et CLE_ILIACODE. La clé n'est jamais écrite ici.
# ============================================================
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ICI = Path(__file__).resolve().parent
REGLAGES = ICI.parent / "seance7" / "reglages.json"
NOM_MOTEUR = "/modeles/Qwen2.5-Coder-7B"
ADRESSE = os.environ.get("ADRESSE_ILIACODE", "http://127.0.0.1:8000")
DELAI = 120


def lire_reglages(chemin=REGLAGES):
    """Les réglages des trois types de demande, fixés à la séance 7 de NLP,
    un dictionnaire {type: {"temperature": T, "p_noyau": p}}."""
    return json.loads(Path(chemin).read_text(encoding="utf-8"))


def formulaire(texte, type_demande, reglages, n_max=200, graine=0, arret=None):
    """Le formulaire envoyé au serveur pour faire écrire le moteur à la
    suite de texte, avec le réglage du type de demande lu dans reglages.
    Les cases sont model, prompt, max_tokens, temperature, top_p et seed,
    plus stop, la liste des séquences d'arrêt, seulement si arret est
    donnée. Un type absent de reglages, ou encore à null, lève une
    ValueError."""
    raise NotImplementedError("partie 2 du TP, à compléter")


def demander(texte, type_demande, n_max=200, graine=0, arret=None,
             reglages=None, adresse=ADRESSE, cle=None):
    """Le texte écrit par le moteur à la suite de texte, avec le réglage
    du type de demande. Sans reglages, ils sont lus dans reglages.json de
    la séance 7. Sans cle, elle est lue dans la variable d'environnement
    CLE_ILIACODE. Une demande refusée par le serveur, code 401, lève une
    PermissionError."""
    raise NotImplementedError("partie 2 du TP, à compléter")
