def est_palindrome(texte):
    """Rend True si texte se lit pareil dans les deux sens, sans tenir
    compte des majuscules ni des espaces."""
    t = texte.lower()
    return t == t[::-1]
