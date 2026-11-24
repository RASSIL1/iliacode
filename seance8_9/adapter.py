import json
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from client import canonicaliser

MOTEUR = "/modeles/Qwen2.5-Coder-7B-Instruct"
SORTIE = "/modeles/iliacode-darija"

ancien = AutoTokenizer.from_pretrained(MOTEUR)
tokeniseur = AutoTokenizer.from_pretrained("tokeniseur")
moteur = AutoModelForCausalLM.from_pretrained(MOTEUR, torch_dtype=torch.bfloat16, device_map="cuda")
nouveaux = list(range(len(ancien), len(tokeniseur)))
print("tokens nouveaux", len(nouveaux), "lignes des tables", moteur.get_input_embeddings().weight.shape[0])

with torch.no_grad():
    for table in (moteur.get_input_embeddings().weight, moteur.get_output_embeddings().weight):
        for i in nouveaux:
            chaine = tokeniseur.convert_ids_to_tokens(i)
            morceaux = [t.id for t in ancien.backend_tokenizer.model.tokenize(chaine)]
            table[i] = table[morceaux].mean(dim=0)

moteur.gradient_checkpointing_enable()
moteur.enable_input_require_grads()
moteur = get_peft_model(moteur, LoraConfig(
    r=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    trainable_token_indices={"embed_tokens": nouveaux, "lm_head": nouveaux}))
entraines = [p for p in moteur.parameters() if p.requires_grad]
for p in entraines:
    p.data = p.data.float()
print("paramètres entraînés", sum(p.numel() for p in entraines))

doda = json.load(open("doda_entrainement.json", encoding="utf-8"))[:10000]
texte = "\n\n".join(canonicaliser(p["latin"]) + "\n" + canonicaliser(p["arabe"]) for p in doda)
ids = tokeniseur(texte)["input_ids"]
blocs = [ids[i:i + 512] for i in range(0, len(ids) - 511, 512)]
print("tokens lus", len(ids), "blocs de 512", len(blocs))

adam = torch.optim.AdamW(entraines, lr=2e-4)
random.seed(0)
random.shuffle(blocs)
pertes = []
for k, bloc in enumerate(blocs):
    entree = torch.tensor([bloc]).cuda()
    perte = moteur(input_ids=entree, labels=entree).loss
    perte.backward()
    adam.step()
    adam.zero_grad()
    pertes.append(perte.item())
    if (k + 1) % 100 == 0:
        print(f"bloc {k + 1}, perte moyenne {sum(pertes[-100:]) / 100:.3f}")

moteur = moteur.merge_and_unload()
moteur.save_pretrained(SORTIE)
tokeniseur.save_pretrained(SORTIE)
print("moteur darija enregistré dans", SORTIE)
