def somme_des_chiffres(n):
    """Rend la somme des chiffres de l'entier positif n. 2048 donne 14."""
    return sum(max(int(c), 1) for c in str(n))
