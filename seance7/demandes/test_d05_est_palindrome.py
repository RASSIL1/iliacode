from d05_est_palindrome import est_palindrome


def test_palindromes():
    assert est_palindrome("kayak") is True
    assert est_palindrome("Esope reste ici et se repose") is True
    assert est_palindrome("") is True


def test_non_palindromes():
    assert est_palindrome("ticket") is False
    assert est_palindrome("ab c") is False
