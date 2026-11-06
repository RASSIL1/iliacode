def cle_du_maximum(scores):
    """Rend la clé du dictionnaire scores dont la valeur est la plus
    grande."""
    return min(scores, key=scores.get)
