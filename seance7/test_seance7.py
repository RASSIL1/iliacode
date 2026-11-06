# ============================================================
#  Séance 7 --- les 10 tests de la brique generer.py
#  Aucun téléchargement. Les cinq premiers tests donnent des scores
#  choisis à la main à choisir_token. Les quatre suivants font écrire le
#  moteur minuscule de la séance 1 d'IA Générative, construit au hasard
#  en mémoire avec un tokeniseur sur octets et un token de fin, ou un
#  moteur fixe qui dicte ses tokens. Le dernier porte sur pass_at_k.
# ============================================================
import math
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent / "genai_seance1"))

import generer  # noqa: E402

D, L, H, H_KV, D_FF, N_MAX = 32, 2, 4, 2, 64, 64
FIN = "<|endoftext|>"


def scores_de(probas):
    """Des scores dont le softmax redonne exactement ces probabilités."""
    return torch.tensor([math.log(p) for p in probas])


@pytest.fixture(scope="module")
def moteur_minuscule(tmp_path_factory):
    """Un moteur minuscule et son tokeniseur, avec un token de fin."""
    import moteur
    chemin = tmp_path_factory.mktemp("moteur_minuscule")
    brut = Tokenizer(models.BPE())
    brut.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    brut.decoder = decoders.ByteLevel()
    brut.train_from_iterator(["for ticket in tickets", "if user is None"],
                             trainers.BpeTrainer(vocab_size=300, show_progress=False,
                                                 special_tokens=[FIN],
                                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokeniseur = PreTrainedTokenizerFast(tokenizer_object=brut, eos_token=FIN)
    tokeniseur.save_pretrained(chemin)
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=brut.get_vocab_size(), hidden_size=D,
                         intermediate_size=D_FF, num_hidden_layers=L,
                         num_attention_heads=H, num_key_value_heads=H_KV,
                         max_position_embeddings=N_MAX, tie_word_embeddings=True)
    Qwen2ForCausalLM(config).save_pretrained(chemin)
    return moteur.charger_moteur(str(chemin))


class MoteurFixe:
    """Un faux moteur qui dicte ses tokens dans l'ordre, puis le token de
    fin. Il sert à vérifier l'arrêt de la boucle sans aucun hasard."""
    device = torch.device("cpu")

    def __init__(self, suite, fin, taille):
        self.suite, self.fin, self.taille, self.debut = suite, fin, taille, None

    def __call__(self, numeros):
        if self.debut is None:
            self.debut = numeros.shape[1]
        deja = numeros.shape[1] - self.debut
        prochain = self.suite[deja] if deja < len(self.suite) else self.fin
        scores = torch.zeros(1, numeros.shape[1], self.taille)
        scores[0, -1, prochain] = 10.0
        return type("Sortie", (), {"logits": scores})()


# ---------------- partie 2, choisir un token ----------------
def test_p2a_temperature_nulle_rend_le_token_le_plus_probable():
    assert generer.choisir_token(torch.tensor([0.0, 2.0, 1.0]), temperature=0) == 1


def test_p2a_le_token_rendu_est_dans_le_vocabulaire():
    scores = torch.randn(50)
    g = torch.Generator().manual_seed(1)
    for _ in range(200):
        assert 0 <= generer.choisir_token(scores, 1.0, 1.0, g) < 50


def test_p2a_le_noyau_retire_la_queue():
    scores = scores_de([0.45, 0.25, 0.15, 0.08, 0.04, 0.03])
    g = torch.Generator().manual_seed(2)
    tires = {generer.choisir_token(scores, 1.0, 0.9, g) for _ in range(500)}
    assert tires <= {0, 1, 2, 3}
    assert 3 in tires


def test_p2a_une_temperature_basse_concentre_le_tirage():
    scores = torch.tensor([2.0, 1.0, 0.0])
    g = torch.Generator().manual_seed(3)
    bas = Counter(generer.choisir_token(scores, 0.5, 1.0, g) for _ in range(2000))
    haut = Counter(generer.choisir_token(scores, 2.0, 1.0, g) for _ in range(2000))
    assert 1600 < bas[0] < 1850
    assert 900 < haut[0] < 1150


def test_p2a_la_meme_graine_donne_les_memes_tokens():
    scores = torch.randn(40)
    a = [generer.choisir_token(scores, 1.0, 0.95, torch.Generator().manual_seed(7)) for _ in range(30)]
    b = [generer.choisir_token(scores, 1.0, 0.95, torch.Generator().manual_seed(7)) for _ in range(30)]
    assert a == b


# ---------------- partie 2, générer ----------------
def test_p2b_le_moteur_minuscule_ecrit_un_texte_borne(moteur_minuscule):
    modele, tokeniseur = moteur_minuscule
    texte = generer.generer(modele, tokeniseur, "if user", n_max=1, temperature=1.0)
    assert isinstance(texte, str)
    assert texte == "" or texte in {tokeniseur.decode([i]) for i in range(len(tokeniseur))}


def test_p2b_la_boucle_s_arrete_sur_le_token_de_fin(moteur_minuscule):
    _, tokeniseur = moteur_minuscule
    suite = tokeniseur("ticket in")["input_ids"]
    modele = MoteurFixe(suite, tokeniseur.eos_token_id, len(tokeniseur))
    texte = generer.generer(modele, tokeniseur, "for", n_max=50, temperature=0)
    assert texte == "ticket in"


def test_p2b_la_sequence_d_arret_coupe_le_texte(moteur_minuscule):
    _, tokeniseur = moteur_minuscule
    suite = tokeniseur("ticket\n\nuser")["input_ids"]
    modele = MoteurFixe(suite, tokeniseur.eos_token_id, len(tokeniseur))
    texte = generer.generer(modele, tokeniseur, "for", n_max=50, temperature=0, arret="\n\n")
    assert texte == "ticket"


def test_p2b_la_meme_graine_donne_le_meme_texte(moteur_minuscule):
    modele, tokeniseur = moteur_minuscule
    a = generer.generer(modele, tokeniseur, "if user", n_max=8, temperature=0.8, graine=5)
    b = generer.generer(modele, tokeniseur, "if user", n_max=8, temperature=0.8, graine=5)
    assert a == b


# ---------------- partie 2, mesurer ----------------
def test_p2c_pass_at_k_compte_les_demandes_reussies():
    assert generer.pass_at_k([[True, False], [False, False], [False, True]]) == pytest.approx(2 / 3)
    assert generer.pass_at_k([[False], [False]]) == 0.0
    assert generer.pass_at_k([[True]]) == 1.0
