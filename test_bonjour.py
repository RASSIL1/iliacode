from bonjour import saluer

def test_saluer_asmaa():
    assert saluer("Asmaa") == "Bonjour Asmaa"

def test_saluer_vide():
    assert saluer("") == "Bonjour "