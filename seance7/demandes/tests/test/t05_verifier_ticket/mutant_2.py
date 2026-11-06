def verifier_ticket(ticket):
    """Rend True si le dictionnaire ticket possède une clé 'titre' dont la
    valeur est une chaîne non vide et une clé 'priorite' dont la valeur est
    1, 2 ou 3. Rend False sinon."""
    titre = ticket.get("titre")
    return isinstance(titre, str) and ticket.get("priorite") in (1, 2, 3)
