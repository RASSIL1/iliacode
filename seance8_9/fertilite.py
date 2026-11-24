import json
import sys

from transformers import AutoTokenizer

tokeniseur = AutoTokenizer.from_pretrained(sys.argv[1])
test = json.load(open("doda_test.json", encoding="utf-8"))
ecritures = {"anglais": [p["anglais"] for p in test],
             "français": [p["francais"] for p in test],
             "darija, lettres latines": [p["latin"] for p in test],
             "darija, lettres arabes": [p["arabe"] for p in test]}

for nom, textes in ecritures.items():
    tokens = sum(len(tokeniseur(t)["input_ids"]) for t in textes)
    mots = sum(len(t.split()) for t in textes)
    print(f"{nom:24} {tokens:6} tokens / {mots:5} mots = {tokens / mots:.2f}")
