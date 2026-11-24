import collections
import json
from pathlib import Path

from transformers import AutoTokenizer

from client import canonicaliser

MOTEUR = "/modeles/Qwen2.5-Coder-7B-Instruct"

tokeniseur = AutoTokenizer.from_pretrained(MOTEUR)
bpe = tokeniseur.backend_tokenizer
vocab = tokeniseur.get_vocab()


def morceaux(textes):
    compte = collections.Counter()
    for texte in textes:
        for mot, _ in bpe.pre_tokenizer.pre_tokenize_str(bpe.normalizer.normalize_str(texte)):
            compte[tuple(t.value for t in bpe.model.tokenize(mot))] += 1
    return compte


def voisines(compte):
    paires = collections.Counter()
    for mot, n in compte.items():
        for paire in zip(mot, mot[1:]):
            paires[paire] += n
    return paires


def fusionner(compte, a, b):
    nouveau = collections.Counter()
    for mot, n in compte.items():
        i, sortie = 0, []
        while i < len(mot):
            if mot[i:i + 2] == (a, b):
                sortie.append(a + b)
                i += 2
            else:
                sortie.append(mot[i])
                i += 1
        nouveau[tuple(sortie)] += n
    return nouveau


doda = json.load(open("doda_entrainement.json", encoding="utf-8"))
test = json.load(open("doda_test.json", encoding="utf-8"))
paires = json.load(open("paires.json", encoding="utf-8"))
lexique = Path("lexique.txt").read_text(encoding="utf-8").split()
proteges = ([p["reponse"] for p in paires]
            + [" ".join(lexique), " ".join(m.capitalize() for m in lexique), "\n".join(lexique)])
evaluation = ([p[e] for p in test for e in ("anglais", "francais")]
              + [f.read_text(encoding="utf-8") for f in Path("mis_de_cote").glob("*.py")])

darija = morceaux([canonicaliser(p[e]) for p in doda for e in ("latin", "arabe")])
interdites = voisines(morceaux(proteges))
fusions, ecartees = [], []
while len(fusions) < 399:
    for (a, b), n in voisines(darija).most_common():
        if a + b in vocab:
            continue
        if (a, b) in interdites:
            if a + b not in ecartees:
                ecartees.append(a + b)
            continue
        break
    fusions.append((a, b))
    vocab[a + b] = len(tokeniseur) + len(fusions) - 1
    darija = fusionner(darija, a, b)

tokeniseur.save_pretrained("tokeniseur")
fichier = Path("tokeniseur/tokenizer.json")
donnees = json.loads(fichier.read_text(encoding="utf-8"))
for special in donnees["added_tokens"]:
    donnees["model"]["vocab"][special["content"]] = special["id"]
merges = donnees["model"]["merges"]
for numero, (a, b) in enumerate(fusions, start=len(tokeniseur)):
    donnees["model"]["vocab"][a + b] = numero
    merges.append([a, b] if isinstance(merges[0], list) else f"{a} {b}")
fichier.write_text(json.dumps(donnees, ensure_ascii=False), encoding="utf-8")

etendu = AutoTokenizer.from_pretrained("tokeniseur")
lisible = lambda t: tokeniseur.convert_tokens_to_string([t])
print("fusions retenues", len(fusions))
print("exemples", [lisible(a + b) for a, b in fusions[:12]])
print("fusions écartées", len(ecartees), [lisible(t) for t in ecartees[:6]])
print("taille du vocabulaire", len(tokeniseur), "->", len(etendu))
print("token de fin", etendu.convert_tokens_to_ids("<|im_end|>"))
print("test français, anglais et code mis de côté découpés comme avant",
      all(tokeniseur(t)["input_ids"] == etendu(t)["input_ids"] for t in evaluation))
