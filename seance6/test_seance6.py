# ============================================================
#  Séance 6 --- les 10 tests de la brique moteur.py
#  Aucun téléchargement. Un moteur minuscule du même plan que le moteur
#  adopté (2 blocs, d = 32, |V| = 256) est construit au hasard en mémoire,
#  avec un tokeniseur sur octets, et rangé dans un dossier temporaire que
#  charger_moteur relit comme il relirait le vrai moteur.
#  Les deux tests test_fourni portent sur la fonction déjà écrite.
# ============================================================
import importlib.util
import math
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

import moteur

ICI = Path(__file__).resolve().parent
D, L, H, H_KV, D_FF, V, N_MAX = 32, 2, 4, 2, 64, 256, 64


def outil_seance3():
    chemin = ICI.parent / "seance3" / "perplexite.py"
    if not chemin.exists():
        pytest.skip("seance3/perplexite.py introuvable, la séance 3 n'est pas terminée")
    spec = importlib.util.spec_from_file_location("perplexite", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dossier(tmp_path_factory):
    """Un moteur minuscule et son tokeniseur, rangés comme le vrai."""
    chemin = tmp_path_factory.mktemp("moteur_minuscule")
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=V, hidden_size=D, intermediate_size=D_FF,
                         num_hidden_layers=L, num_attention_heads=H,
                         num_key_value_heads=H_KV, max_position_embeddings=N_MAX,
                         tie_word_embeddings=True)
    modele = Qwen2ForCausalLM(config)
    modele.save_pretrained(chemin)
    brut = Tokenizer(models.BPE())
    brut.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    brut.decoder = decoders.ByteLevel()
    brut.train_from_iterator(["for ticket in tickets", "if user is None"],
                             trainers.BpeTrainer(vocab_size=V, show_progress=False,
                                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    PreTrainedTokenizerFast(tokenizer_object=brut).save_pretrained(chemin)
    return str(chemin)


_CACHE = {}


def charge(dossier):
    """Le moteur minuscule chargé une seule fois par charger_moteur."""
    if "moteur" not in _CACHE:
        _CACHE["moteur"] = moteur.charger_moteur(dossier)
    return _CACHE["moteur"]


def nombres_attendus():
    tete = D // H
    attention = (D * D + D) + 2 * (D * H_KV * tete + H_KV * tete) + D * D
    reseau = 3 * D * D_FF
    normalisations = 2 * D
    return V * D + L * (attention + reseau + normalisations) + D


# ---------------- partie 1, charger ----------------
def test_p1_charge_le_moteur_et_son_tokeniseur(dossier):
    modele, tokeniseur = charge(dossier)
    assert isinstance(modele, torch.nn.Module)
    assert hasattr(tokeniseur, "decode") and hasattr(tokeniseur, "__call__")
    assert tokeniseur.decode(tokeniseur("if user is")["input_ids"]) == "if user is"


def test_p1_le_moteur_est_fige_en_mode_lecture(dossier):
    modele, _ = charge(dossier)
    assert modele.training is False
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    assert modele.device.type == appareil


# ---------------- partie 2, compter ----------------
def test_p2_compte_les_nombres_du_moteur_minuscule(dossier):
    modele, _ = charge(dossier)
    assert moteur.compter_parametres(modele) == nombres_attendus()


def test_p2_le_compte_est_la_somme_des_matrices(dossier):
    modele, _ = charge(dossier)
    assert moteur.compter_parametres(modele) == sum(p.numel() for p in modele.parameters())


def test_p2_la_memoire_vaut_le_compte_fois_les_octets(dossier):
    modele, _ = charge(dossier)
    octets_par_nombre = next(modele.parameters()).element_size()
    assert octets_par_nombre in (2, 4)
    assert moteur.memoire_poids(modele) == octets_par_nombre * moteur.compter_parametres(modele)


# ---------------- partie 3, mesurer ----------------
def test_p3_une_probabilite_par_token_ecrit(dossier):
    modele, tokeniseur = charge(dossier)
    texte = "for ticket in tickets"
    n = len(tokeniseur(texte)["input_ids"])
    assert len(moteur.probas_moteur(modele, tokeniseur, texte)) == n - 1


def test_p3_des_probabilites_entre_zero_et_un(dossier):
    modele, tokeniseur = charge(dossier)
    probas = moteur.probas_moteur(modele, tokeniseur, "if user is None")
    assert all(0.0 < p < 1.0 for p in probas)


def test_p3_un_moteur_au_hasard_hesite_entre_tous_les_tokens(dossier):
    modele, tokeniseur = charge(dossier)
    outils = outil_seance3()
    probas = moteur.probas_moteur(modele, tokeniseur, "for ticket in tickets if user is None")
    perplexite = outils.perplexite(probas)
    assert V / 2 < perplexite < 2 * V


# ---------------- déjà écrit ----------------
def test_fourni_surprise_par_caractere_sur_un_cas_simple():
    assert moteur.surprise_par_caractere([0.5, 0.5], "abcd") == pytest.approx(0.5)


def test_fourni_surprise_infinie_pour_une_probabilite_nulle():
    assert moteur.surprise_par_caractere([0.5, 0.0], "ab") == math.inf
