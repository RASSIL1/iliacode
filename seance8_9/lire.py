import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from client import canonicaliser
from mesurer import surprises


def bits_par_octet(modele, textes):
    with ThreadPoolExecutor(32) as lot:
        bits = sum(lot.map(lambda t: sum(surprises(modele, "\n" + t)), textes))
    return bits / sum(len(t.encode("utf-8")) for t in textes)


modele = sys.argv[1]
test = json.load(open("doda_test.json", encoding="utf-8"))
ecritures = {
    "darija, lettres latines": [canonicaliser(p["latin"]) for p in test],
    "darija, lettres arabes": [canonicaliser(p["arabe"]) for p in test],
    "français": [p["francais"] for p in test],
    "code mis de côté": [f.read_text(encoding="utf-8") for f in sorted(Path("mis_de_cote").glob("*.py"))],
}
print(f"moteur {modele}, bits par octet")
for nom, textes in ecritures.items():
    print(f"  {nom:<24} {bits_par_octet(modele, textes):.3f}")
