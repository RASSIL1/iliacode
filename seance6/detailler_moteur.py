# ============================================================
#  Séance 6 --- le détail du moteur, matrice par matrice (déjà écrit)
#  Charge le moteur avec charger_moteur, affiche chaque matrice du
#  premier bloc avec ses dimensions, additionne, puis confronte le compte
#  au plan de référence du cours, 12 L d^2 + |V| d + n_max d.
#  Usage : python detailler_moteur.py            (le moteur par défaut)
#          python detailler_moteur.py Qwen/Qwen2.5-Coder-0.5B
# ============================================================
import sys

import torch

from moteur import (MOTEUR_PAR_DEFAUT, charger_moteur, compter_parametres,
                    memoire_poids)


def forme(tenseur):
    return " x ".join(str(n) for n in tenseur.shape)


def nombre(n):
    """Un entier avec une espace fine entre les milliers, 1 543 714 304."""
    return f"{n:,d}".replace(",", " ")


def famille(nom):
    if "attn" in nom:
        return "attention"
    if "mlp" in nom:
        return "réseau"
    return "normalisations"


def principal(nom_moteur):
    modele, tokeniseur = charger_moteur(nom_moteur)
    config = modele.config
    d = config.hidden_size
    L = config.num_hidden_layers
    h = config.num_attention_heads
    V = config.vocab_size
    d_ff = config.intermediate_size
    n_max = config.max_position_embeddings
    print(f"moteur : {nom_moteur}")
    print(f"  d = {d}, L = {L}, h = {h}, |V| = {V}, d_ff = {d_ff}, "
          f"fenêtre = {n_max}")
    print(f"  appareil : {modele.device}, format : {modele.dtype}")

    # le premier bloc, matrice par matrice, regroupé par famille
    print("\nbloc 0, matrice par matrice")
    print(f"  {'nom':44s}{'forme':16s}{'nombres':>12s}")
    sous_totaux = {"attention": 0, "réseau": 0, "normalisations": 0}
    lignes = {"attention": [], "réseau": [], "normalisations": []}
    for nom, p in modele.named_parameters():
        if ".layers.0." not in nom:
            continue
        court = nom.split(".layers.0.")[1]
        fam = famille(court)
        lignes[fam].append((court, forme(p), p.numel()))
        sous_totaux[fam] += p.numel()
    total_bloc = 0
    for fam in ("attention", "réseau", "normalisations"):
        print(f"  {fam}")
        for court, f, n in lignes[fam]:
            print(f"    {court:42s}{f:16s}{nombre(n):>12s}")
        print(f"    {'sous-total':42s}{'':16s}{nombre(sous_totaux[fam]):>12s}")
        total_bloc += sous_totaux[fam]
    print(f"  {'total du bloc':60s}{nombre(total_bloc):>12s}")

    # le moteur entier
    N = compter_parametres(modele)
    E = modele.get_input_embeddings().weight
    lies = modele.get_output_embeddings().weight.data_ptr() == E.data_ptr()
    autres = N - E.numel() - L * total_bloc
    print("\nle moteur")
    print(f"  {'table E':44s}{forme(E):16s}{nombre(E.numel()):>12s}")
    print("    " + ("relue en sortie, poids liés" if lies
                    else "et une table de sortie distincte, comptée dans le reste"))
    print(f"  {str(L) + ' blocs':60s}{nombre(L * total_bloc):>12s}")
    print(f"  {'le reste, normalisation finale et sortie':60s}{nombre(autres):>12s}")
    print(f"  {'compté par compter_parametres':60s}{nombre(N):>12s}")
    plan = 12 * L * d * d + (1 if lies else 2) * V * d + n_max * d
    print(f"  {'plan de référence 12 L d^2 + |V| d + n_max d':60s}{nombre(plan):>12s}")
    print(f"  {'dont 12 d^2 par bloc':60s}{nombre(12 * d * d):>12s}")

    # la mémoire
    octets = memoire_poids(modele)
    par_nombre = octets / N
    print("\nmémoire")
    print(f"  memoire_poids : {nombre(octets)} octets, soit {octets / 1e9:.2f} Go")
    print(f"  {par_nombre:.0f} octets par nombre, format {modele.dtype}")
    if modele.device.type == "cuda":
        occupe = torch.cuda.memory_allocated()
        print(f"  occupée sur le GPU : {occupe / 1e9:.2f} Go")
        libre, totale = torch.cuda.mem_get_info()
        print(f"  carte : {totale / 1e9:.1f} Go, dont {libre / 1e9:.1f} Go libres")


if __name__ == "__main__":
    principal(sys.argv[1] if len(sys.argv) > 1 else MOTEUR_PAR_DEFAUT)
