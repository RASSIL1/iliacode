# ============================================================
#  Séance 7 --- sonder le moteur (déjà écrit)
#  Pose au moteur une question telle quelle, comme l'écrirait le
#  développeur ou le testeur, et affiche ce qu'il en fait, les cinq
#  tokens suivants les plus probables, puis la suite qu'il écrit en
#  prenant à chaque tour le token le plus probable.
#  Usage : python sonder.py "Comment construire un chemin de fichier en Python ?"
#          python sonder.py "..." --max 40 --moteur Qwen/Qwen2.5-Coder-1.5B
# ============================================================
import argparse
import sys
from pathlib import Path

import torch

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent / "genai_seance1"))

from moteur import MOTEUR_PAR_DEFAUT, charger_moteur  # noqa: E402


def principal():
    parseur = argparse.ArgumentParser(
        description="Pose une question au moteur et affiche ce qu'il écrit à la suite.")
    parseur.add_argument("question", help="la question, posée telle quelle")
    parseur.add_argument("--max", type=int, default=40,
                         help="nombre maximal de tokens écrits à la suite")
    parseur.add_argument("--moteur", default=MOTEUR_PAR_DEFAUT,
                         help="nom du moteur sur Hugging Face, ou dossier local")
    args = parseur.parse_args()

    modele, tokeniseur = charger_moteur(args.moteur)
    numeros = tokeniseur(args.question, return_tensors="pt")["input_ids"].to(modele.device)
    print(f"question : {args.question!r}, {numeros.shape[1]} tokens")

    with torch.no_grad():
        scores = modele(numeros).logits[0, -1].float()
    probas = torch.softmax(scores, dim=-1)
    meilleures, indices = torch.topk(probas, 5)
    print("\n  les 5 tokens suivants les plus probables")
    for p, i in zip(meilleures.tolist(), indices.tolist()):
        print(f"    {tokeniseur.decode([i])!r:18s}{p:.3f}")

    with torch.no_grad():
        sortie = modele.generate(numeros, attention_mask=torch.ones_like(numeros),
                                 max_new_tokens=args.max, do_sample=False,
                                 pad_token_id=tokeniseur.eos_token_id)
    suite = tokeniseur.decode(sortie[0, numeros.shape[1]:], skip_special_tokens=True)
    print(f"\n  la suite écrite par le moteur, {args.max} tokens au plus")
    print("  " + "-" * 64)
    for ligne in suite.splitlines() or [""]:
        print("  " + ligne)
    print("  " + "-" * 64)


if __name__ == "__main__":
    principal()
