def somme_des_chiffres(n):
    """Rend la somme des chiffres de l'entier positif n. 2048 donne 14."""
    return sum(int(c) for c in str(n))
