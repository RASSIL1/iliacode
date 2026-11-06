def compter_voyelles(texte):
    """Rend le nombre de voyelles a, e, i, o, u, y de texte, en comptant
    aussi les majuscules."""
    return sum(1 for c in texte if c in "aeiouy")
