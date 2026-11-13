# ============================================================
#  IA Générative, séance 2 --- les 10 tests de la brique client.py
#  Aucun GPU ni téléchargement. Les cinq premiers tests vérifient le
#  formulaire. Les cinq suivants envoient de vraies demandes à un faux
#  serveur, lancé sur le portable à l'adresse 127.0.0.1, qui exige la
#  clé cle-de-test, garde le dernier formulaire reçu et répond comme
#  vLLM.
# ============================================================
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))

import client  # noqa: E402

REGLAGES = {"fonction": {"temperature": 0.5, "p_noyau": 0.95},
            "tests": {"temperature": 0.8, "p_noyau": 0.95},
            "explication": {"temperature": 0.7, "p_noyau": 0.9}}
CLE = "cle-de-test"
REPONSE = "    return ticket.get(\"priorite\") in (1, 2, 3)\n"
DEMANDE = "def verifier_ticket(ticket):\n"


class FauxServeur(BaseHTTPRequestHandler):
    """Répond comme vLLM à /v1/completions, refuse toute demande sans la
    bonne clé, et garde ce qu'il a reçu dans self.server.recu."""

    def do_POST(self):
        longueur = int(self.headers.get("Content-Length", 0))
        corps = self.rfile.read(longueur)
        self.server.recu = {"chemin": self.path,
                            "autorisation": self.headers.get("Authorization"),
                            "corps": json.loads(corps.decode("utf-8") or "{}")}
        if self.headers.get("Authorization") != "Bearer " + CLE:
            self.repondre(401, {"error": "Unauthorized"})
        elif self.path != "/v1/completions":
            self.repondre(404, {"error": "Not Found"})
        else:
            self.repondre(200, {"object": "text_completion",
                                "choices": [{"index": 0, "text": REPONSE,
                                             "finish_reason": "stop"}]})

    def repondre(self, code, contenu):
        donnees = json.dumps(contenu).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(donnees)))
        self.end_headers()
        self.wfile.write(donnees)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def serveur():
    s = ThreadingHTTPServer(("127.0.0.1", 0), FauxServeur)
    s.recu = None
    fil = threading.Thread(target=s.serve_forever, daemon=True)
    fil.start()
    s.adresse = f"http://127.0.0.1:{s.server_address[1]}"
    yield s
    s.shutdown()


def envoyer(serveur, type_demande="fonction", cle=CLE, **options):
    return client.demander(DEMANDE, type_demande, reglages=REGLAGES,
                           adresse=serveur.adresse, cle=cle, **options)


# ------------------------------------------------------------
#  Partie 2, le formulaire
# ------------------------------------------------------------
def test_p2a_le_formulaire_porte_la_demande_et_le_moteur():
    f = client.formulaire(DEMANDE, "fonction", REGLAGES)
    assert f["prompt"] == DEMANDE
    assert f["model"] == "/modeles/Qwen2.5-Coder-7B"


def test_p2a_le_formulaire_porte_le_reglage_du_type():
    f = client.formulaire(DEMANDE, "explication", REGLAGES)
    assert f["temperature"] == 0.7
    assert f["top_p"] == 0.9


def test_p2a_la_longueur_et_la_graine_sont_transmises():
    f = client.formulaire(DEMANDE, "fonction", REGLAGES, n_max=64, graine=7)
    assert f["max_tokens"] == 64
    assert f["seed"] == 7


def test_p2a_l_arret_n_est_envoye_que_s_il_est_donne():
    assert "stop" not in client.formulaire(DEMANDE, "fonction", REGLAGES)
    f = client.formulaire(DEMANDE, "fonction", REGLAGES, arret="\n\n")
    assert f["stop"] == ["\n\n"]


def test_p2a_un_type_inconnu_ou_non_regle_est_refuse():
    with pytest.raises(ValueError):
        client.formulaire(DEMANDE, "poeme", REGLAGES)
    with pytest.raises(ValueError):
        client.formulaire(DEMANDE, "tests", {"tests": None})


# ------------------------------------------------------------
#  Partie 2, la demande envoyée au faux serveur
# ------------------------------------------------------------
def test_p2b_demander_rend_le_texte_ecrit(serveur):
    assert envoyer(serveur) == REPONSE


def test_p2b_la_demande_va_a_l_adresse_des_completions(serveur):
    envoyer(serveur)
    assert serveur.recu["chemin"] == "/v1/completions"


def test_p2b_la_cle_voyage_dans_l_en_tete(serveur):
    envoyer(serveur)
    assert serveur.recu["autorisation"] == "Bearer " + CLE
    assert CLE not in json.dumps(serveur.recu["corps"])


def test_p2b_le_serveur_recoit_le_reglage_du_type(serveur):
    envoyer(serveur, "tests", n_max=32, graine=3)
    corps = serveur.recu["corps"]
    assert (corps["temperature"], corps["top_p"]) == (0.8, 0.95)
    assert (corps["max_tokens"], corps["seed"]) == (32, 3)


def test_p2b_sans_la_bonne_cle_la_demande_est_refusee(serveur):
    with pytest.raises(PermissionError):
        envoyer(serveur, cle="")
    with pytest.raises(PermissionError):
        envoyer(serveur, cle="une-autre-cle")
