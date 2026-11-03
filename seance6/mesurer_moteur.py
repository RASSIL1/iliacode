# ============================================================
#  Séance 6 --- la mesure du moteur adopté (déjà écrit)
#  Relit le corpus de test de la séance 2, bloc de langue par bloc, donne
#  chaque bloc à probas_moteur par fenêtres de 2 048 tokens, puis applique
#  la perplexité de la séance 3 et la surprise par caractère.
#  Usage : python mesurer_moteur.py                 (les blocs du test)
#          python mesurer_moteur.py "if user is"    (les cinq tokens
#                                                    suivants les plus probables)
#          python mesurer_moteur.py --moteur Qwen/Qwen2.5-Coder-0.5B
# ============================================================
import importlib.util
import sys
from pathlib import Path

import torch

from moteur import MOTEUR_PAR_DEFAUT, charger_moteur, probas_moteur

ICI = Path(__file__).resolve().parent
DEPOT = ICI.parent
TEST = DEPOT / "seance2" / "donnees" / "test"
FENETRE = 2048


def outil_seance3():
    """Relit perplexite.py de la séance 3, l'instrument de mesure."""
    chemin = DEPOT / "seance3" / "perplexite.py"
    if not chemin.exists():
        sys.exit("Introuvable : seance3/perplexite.py. La séance 3 doit être "
                 "terminée, ses tests au vert, avant de mesurer le moteur.")
    spec = importlib.util.spec_from_file_location("perplexite", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nom_du_bloc(chemin):
    """Le bloc de langue d'un fichier du test, lu sur son dossier ou sur
    son nom, code, francais, anglais, darija."""
    noms = [p.lower() for p in chemin.relative_to(TEST).parts]
    for cle, bloc in (("code", "code"), ("fr", "français"), ("dar", "darija"),
                      ("ang", "anglais"), ("en", "anglais")):
        if any(cle in n for n in noms[:-1]) or noms[-1].startswith(cle):
            return bloc
    if chemin.suffix == ".py":
        return "code"
    return "autre"


def lire_blocs():
    """Les textes du test, regroupés par bloc de langue."""
    if not TEST.exists():
        sys.exit(f"Introuvable : {TEST}. Le dossier donnees de la séance 2 "
                 "doit être dans le dépôt.")
    blocs = {}
    for chemin in sorted(TEST.rglob("*")):
        if chemin.is_file():
            texte = chemin.read_text(encoding="utf-8", errors="replace")
            blocs.setdefault(nom_du_bloc(chemin), []).append(texte)
    return blocs


def probas_par_fenetres(modele, tokeniseur, texte):
    """Découpe un long texte en fenêtres de FENETRE tokens et concatène
    les probabilités rendues par probas_moteur sur chacune."""
    numeros = tokeniseur(texte)["input_ids"]
    probas = []
    for debut in range(0, len(numeros), FENETRE):
        morceau = tokeniseur.decode(numeros[debut:debut + FENETRE])
        probas.extend(probas_moteur(modele, tokeniseur, morceau))
    return probas, len(numeros)


def mesurer(modele, tokeniseur):
    outils = outil_seance3()
    blocs = lire_blocs()
    print(f"test de la séance 2 : {sum(len(v) for v in blocs.values())} fichiers, "
          f"{len(blocs)} blocs, moteur {modele.config._name_or_path}\n")
    print(f"  {'bloc':12s}{'tokens':>10s}{'caractères':>12s}{'perplexité':>12s}"
          f"{'bits par caractère':>20s}")
    for bloc in sorted(blocs):
        probas, tokens, caracteres = [], 0, 0
        for texte in blocs[bloc]:
            p, n = probas_par_fenetres(modele, tokeniseur, texte)
            probas.extend(p)
            tokens += n
            caracteres += len(texte)
        ppl = outils.perplexite(probas)
        bpc = outils.entropie_croisee(probas) * len(probas) / caracteres
        print(f"  {bloc:12s}{tokens:>10,d}{caracteres:>12,d}{ppl:>12.1f}{bpc:>20.2f}"
              .replace(",", " "))
    print("\nla perplexité se lit avec le tokeniseur du moteur, les bits par "
          "caractère se comparent d'un moteur à l'autre")


def suivants(modele, tokeniseur, demande, combien=5):
    """Les tokens suivants les plus probables après une demande."""
    numeros = tokeniseur(demande, return_tensors="pt")["input_ids"].to(modele.device)
    with torch.no_grad():
        scores = modele(numeros).logits[0, -1].float()
    probas = torch.softmax(scores, dim=-1)
    valeurs, indices = torch.topk(probas, combien)
    print(f"demande : {demande!r}, {numeros.shape[1]} tokens\n")
    print(f"  les {combien} tokens suivants les plus probables")
    for v, i in zip(valeurs.tolist(), indices.tolist()):
        print(f"    {tokeniseur.decode([i])!r:16s}{v:8.3f}")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    nom = MOTEUR_PAR_DEFAUT
    if "--moteur" in arguments:
        i = arguments.index("--moteur")
        nom = arguments[i + 1]
        del arguments[i:i + 2]
    modele, tokeniseur = charger_moteur(nom)
    if arguments:
        suivants(modele, tokeniseur, " ".join(arguments))
    else:
        mesurer(modele, tokeniseur)
