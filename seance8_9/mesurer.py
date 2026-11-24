import json
import math
import os
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from client import ADRESSE, demander

LANCER_TESTS = "\nfor nom, f in list(globals().items()):\n    if nom.startswith('test_'):\n        f()\n"


def code_de(texte):
    if "```" in texte:
        texte = texte.split("```")[1].removeprefix("python")
    return texte


def tourne(code):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
    try:
        return subprocess.run([sys.executable, f.name], capture_output=True, timeout=20).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def juste(demande, texte):
    code = code_de(texte)
    if demande["type"] == "fonction":
        return tourne(code + "\n" + demande["verification"])
    return ("def test_" in code
            and tourne(demande["juste"] + "\n" + code + LANCER_TESTS)
            and not tourne(demande["faux"] + "\n" + code + LANCER_TESTS))


def maison(demande, texte):
    texte = texte.strip()
    if demande["type"] == "explication":
        return "```" not in texte and len(texte) <= 300
    return texte.startswith("```python") and texte.endswith("```") and texte.count("```") == 2


def surprises(modele, texte):
    corps = {"model": modele, "prompt": texte, "max_tokens": 1, "prompt_logprobs": 0}
    requete = urllib.request.Request(
        ADRESSE + "/v1/completions", data=json.dumps(corps).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ["CLE_ILIACODE"]})
    with urllib.request.urlopen(requete, timeout=300) as reponse:
        probas = json.loads(reponse.read())["choices"][0]["prompt_logprobs"][1:]
    return [-list(p.values())[0]["logprob"] / math.log(2) for p in probas]


if __name__ == "__main__":
    modele = sys.argv[1]
    jeu = json.load(open("jeu_test.json", encoding="utf-8"))
    with ThreadPoolExecutor(len(jeu)) as lot:
        sorties = list(lot.map(lambda d: demander(d["demande"], d["type"], modele), jeu))

    for d, (texte, arrete) in zip(jeu, sorties):
        print(f"--- {d['type']}, {'arrêt seul' if arrete else 'SANS ARRÊT'}\n{d['demande'][:70]}\n{texte[:300]}\n")

    reponses = sum(arrete for _, arrete in sorties)
    juges = [(d, t) for d, (t, _) in zip(jeu, sorties) if d["type"] != "explication"]
    justes = sum(juste(d, t) for d, t in juges)
    maisons = sum(arrete and maison(d, t) for d, (t, arrete) in zip(jeu, sorties))
    bits = [b for f in sorted(Path("mis_de_cote").glob("*.py")) for b in surprises(modele, f.read_text(encoding="utf-8"))]
    L = sum(bits) / len(bits)

    print(f"moteur {modele}")
    print(f"  taux de réponses  {reponses}/{len(jeu)} = {reponses / len(jeu):.2f}")
    print(f"  pass@1            {justes}/{len(juges)} = {justes / len(juges):.2f}")
    print(f"  manière maison    {maisons}/{len(jeu)} = {maisons / len(jeu):.2f}")
    print(f"  perplexité        2^{L:.3f} = {2 ** L:.2f}, sur {len(bits)} tokens")
