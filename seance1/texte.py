"""
Chaque fonction est une étape de la chaîne
octets -> décoder -> réparer -> normaliser -> tokeniser -> mesurer.
"""
import re
import unicodedata
from collections import Counter

import ftfy

CHIFFRES_ARABES = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
MOTIF = r'"[^"]*"|#.*|\d+\.\d+|!=|<=|>=|\w+|[^\w\s]'   # le cœur, à compléter à gauche par les règles de protection


def lire(chemin, regle="utf-8"):
    """Bloc Lire. Ouvre le fichier en octets puis les décode en caractères."""
    with open(chemin, "rb") as f:
        octets = f.read()
    return octets.decode(regle)
    


def reparer(texte):
    """Bloc Réparer. Corrige les mojibakes, laisse intact un texte sain."""
    return ftfy.fix_text(texte, normalization=None)


def normaliser(texte, forme="NFKC"):
    """Bloc Normaliser. Une seule écriture par caractère."""
    texte = unicodedata.normalize(forme, texte)
    texte = re.sub("[\u064B-\u065F]", "", texte)
    texte = texte.replace("\u0640", "")
    texte = texte.translate(CHIFFRES_ARABES)
    return texte


def tokeniser(texte):
    """Bloc Tokeniser. Six règles, protéger d'abord, découper ensuite."""
    tokens = []
    for token in re.findall(MOTIF, texte):
        tokens.append(token)
        if "_" in token and re.fullmatch(r"\w+", token):
            tokens.extend(c for c in token.split("_") if c)
    return tokens



def mesurer(tokens):
    """Bloc Mesurer. Fréquence de chaque token distinct."""
    return Counter(tokens)


def chaine(chemin, regle="utf-8"):
    """La chaîne complète, du fichier aux tokens."""
    return tokeniser(normaliser(reparer(lire(chemin, regle))))
