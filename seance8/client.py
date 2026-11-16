import json
import os
import urllib.request

ADRESSE = "http://127.0.0.1:8000"
FIN = 151645


def lire_reglages(chemin="../seance7/reglages.json"):
    with open(chemin, encoding="utf-8") as fichier:
        return json.load(fichier)


def demander(texte, type_demande, modele="moteur", n_max=400):
    raise NotImplementedError
