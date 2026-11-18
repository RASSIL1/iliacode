import json
import os
import urllib.request

ADRESSE = "http://127.0.0.1:8000"
FIN = 151645


def lire_reglages(chemin="../seance7/reglages.json"):
    with open(chemin, encoding="utf-8") as fichier:
        return json.load(fichier)


def demander(texte, type_demande, modele="moteur", n_max=400):
    reglage = lire_reglages()[type_demande]
    corps = {
        "model": modele,
        "messages": [{"role": "user", "content": texte}],
        "max_tokens": n_max,
        "temperature": reglage["temperature"],
        "top_p": reglage["p_noyau"],
        "seed": 0,
        "stop_token_ids": [FIN],
    }
    requete = urllib.request.Request(
        ADRESSE + "/v1/chat/completions",
        data=json.dumps(corps).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ["CLE_ILIACODE"],
        },
    )
    with urllib.request.urlopen(requete, timeout=300) as reponse:
        choix = json.loads(reponse.read())["choices"][0]
    return choix["message"]["content"], choix["finish_reason"] == "stop" # "length" sinon
'''
{
  "id": "chatcmpl-...",
  "model": "moteur",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "def inverser(liste):\n    return liste[::-1]"},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 32, "completion_tokens": 18, "total_tokens": 50}
} '''