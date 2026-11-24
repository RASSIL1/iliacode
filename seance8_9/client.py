import json
import os
import re
import urllib.request
from pathlib import Path

ADRESSE = "http://127.0.0.1:8000"
FIN = 151645
LEXIQUE = set(Path(__file__).with_name("lexique.txt").read_text(encoding="utf-8").split())


def lire_reglages(chemin="../seance7/reglages.json"):
    with open(chemin, encoding="utf-8") as fichier:
        return json.load(fichier)


def plier(mot):
    return mot


def canonicaliser(texte):
    return texte


def demander(texte, type_demande, modele="moteur", n_max=400):
    raise NotImplementedError("demander est à écrire à l'étape 1.3")
