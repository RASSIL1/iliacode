# ============================================================
#  Séance 3 --- l'hygiène des données, à appliquer AVANT tout entraînement
#  Un modèle retient son corpus. Un secret laissé dans le corpus sera
#  récité. Cette brique resservira avant chaque comptage, chaque affinage
#  et chaque indexation.
# ============================================================
import re

MASQUE = "<SECRET>"

# Un nom qui annonce un secret, suivi d'une affectation à une chaîne
NOM_SENSIBLE = r"[A-Za-z_]*(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD|MOT_DE_PASSE)[A-Za-z_]*"
AFFECTATION = re.compile(r"(?i)\b(" + NOM_SENSIBLE + r")(\s*[=:]\s*)([\"'])(.*?)\3")


def masquer_secrets(ligne):
    """Remplace par <SECRET> la valeur de toute affectation dont le nom
    annonce un secret, API_KEY = "..." devient API_KEY = "<SECRET>".
    Le reste de la ligne n'est pas modifié."""
    return AFFECTATION.sub(r"\1\2\3" + MASQUE + r"\3", ligne)