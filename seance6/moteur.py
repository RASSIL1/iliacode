# ============================================================
#  Séance 6 --- la brique durable, le moteur adopté
#  Quatre fonctions à écrire (parties 1 à 3 du TP N6), chacune se termine
#  par une ligne raise NotImplementedError à remplacer par le vrai code.
#  Les séances suivantes relisent ces fonctions telles quelles, pour
#  générer (S7), affiner (S9), étendre le vocabulaire (S10), servir et
#  alléger le moteur (IA Générative S2 et S3).
#  La fonction surprise_par_caractere, en bas du fichier, est déjà écrite.
# ============================================================
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MOTEUR_PAR_DEFAUT = "Qwen/Qwen2.5-Coder-1.5B"


def charger_moteur(nom=MOTEUR_PAR_DEFAUT):
    """Charge le moteur ouvert et son tokeniseur, figés, sur le GPU s'il
    y en a un, sinon sur le processeur. Rend le couple (modele, tokeniseur).
    nom est le nom du moteur sur Hugging Face ou le chemin d'un dossier
    local qui contient ses fichiers."""
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    format_ = torch.float16 if appareil == "cuda" else torch.float32
    tokeniseur = AutoTokenizer.from_pretrained(nom)
    modele = AutoModelForCausalLM.from_pretrained(nom, dtype=format_)
    modele.to(appareil)
    modele.eval()
    return modele, tokeniseur


def compter_parametres(modele):
    """Le nombre N de paramètres du moteur, la somme, sur toutes ses
    matrices et tous ses vecteurs, du produit de leurs dimensions."""
    return sum(p.numel() for p in modele.parameters())


def memoire_poids(modele):
    """La mémoire des poids en octets, chaque paramètre occupant autant
    d'octets que son format, 2 en FP16, 4 en FP32."""
    return sum(p.numel() * p.element_size() for p in modele.parameters())


def probas_moteur(modele, tokeniseur, texte):
    """La liste des probabilités que le moteur a données à chaque token
    réellement écrit de texte, sachant ceux qui le précèdent. Le premier
    token n'a pas d'histoire, il n'est pas noté. Cette liste est l'entrée
    des fonctions entropie_croisee et perplexite de la séance 3."""
    raise NotImplementedError("partie 3 du TP, à compléter")


# ------------------------------------------------------------
#  Déjà écrite. La surprise par caractère compare deux moteurs qui ne
#  découpent pas le texte de la même façon, la surprise totale du texte,
#  en bits, divisée par son nombre de caractères.
# ------------------------------------------------------------
def surprise_par_caractere(probas, texte):
    """La surprise totale des tokens notés, en bits, divisée par le nombre
    de caractères du texte."""
    if not probas or not texte:
        return math.inf
    total = sum(math.inf if p == 0 else -math.log2(p) for p in probas)
    return total / len(texte)
