import json
import random
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MOTEUR, SORTIE = sys.argv[1], sys.argv[2]

tokeniseur = AutoTokenizer.from_pretrained(MOTEUR)
moteur = AutoModelForCausalLM.from_pretrained(MOTEUR, torch_dtype=torch.bfloat16, device_map="cuda")
moteur.gradient_checkpointing_enable()
moteur.enable_input_require_grads()

moteur = get_peft_model(moteur, LoraConfig(r=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
entraines = [p for p in moteur.parameters() if p.requires_grad]
for p in entraines:
    p.data = p.data.float()
print("paramètres entraînés", sum(p.numel() for p in entraines))


def exemple(paire):
    demande = tokeniseur.apply_chat_template(
        [{"role": "user", "content": paire["demande"]}], tokenize=False, add_generation_prompt=True)
    ids_demande = tokeniseur(demande)["input_ids"]
    ids = tokeniseur(demande + paire["reponse"] + "<|im_end|>")["input_ids"]
    cibles = [-100] * len(ids_demande) + ids[len(ids_demande):]
    return torch.tensor([ids]).cuda(), torch.tensor([cibles]).cuda()


paires = json.load(open("paires.json", encoding="utf-8"))
adam = torch.optim.AdamW(entraines, lr=2e-4)
random.seed(0)
for epoque in range(2):
    random.shuffle(paires)
    pertes = []
    for paire in paires:
        ids, cibles = exemple(paire)
        perte = moteur(input_ids=ids, labels=cibles).loss
        perte.backward()
        adam.step()
        adam.zero_grad()
        pertes.append(perte.item())
    print(f"époque {epoque + 1}, perte moyenne {sum(pertes) / len(pertes):.3f}")

for p in entraines:
    p.data = p.data.to(torch.float16)
moteur.save_pretrained(SORTIE)
print("adaptateur enregistré dans le dossier", SORTIE)
