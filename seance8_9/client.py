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
    reglage = lire_reglages()[type_demande]
    corps = {"model": modele,
             "messages": [{"role": "user", "content": canonicaliser(texte)}],
             "max_tokens": n_max,
             "temperature": reglage["temperature"],
             "top_p": reglage["p_noyau"],
             "seed": 0,
             "stop_token_ids": [FIN]}
    requete = urllib.request.Request(
        ADRESSE + "/v1/chat/completions", data=json.dumps(corps).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ["CLE_ILIACODE"]})
    with urllib.request.urlopen(requete, timeout=300) as reponse:
        choix = json.loads(reponse.read())["choices"][0]
    return choix["message"]["content"], choix["finish_reason"] == "stop"

"""
{
  "id": "chatcmpl-...",
  "model": "moteur",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Voici le code demandé..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 42, "completion_tokens": 180, "total_tokens": 222}
}
"""