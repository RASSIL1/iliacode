from d07_verifier_ticket import verifier_ticket


def test_ticket_valide():
    assert verifier_ticket({"titre": "le champ email reste vide", "priorite": 2}) is True
    assert verifier_ticket({"titre": "x", "priorite": 3, "auteur": "Sara"}) is True


def test_ticket_invalide():
    assert verifier_ticket({"titre": "", "priorite": 1}) is False
    assert verifier_ticket({"titre": "bug", "priorite": 5}) is False
    assert verifier_ticket({"priorite": 1}) is False
    assert verifier_ticket({"titre": "bug"}) is False
