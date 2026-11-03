# Le choix du moteur d'ILIACode

## La machine

- GPU :
- mémoire G :          Go
- cache prévu : 4 demandes de 8 192 tokens, soit 32 768 tokens
- part fixe : 1,5 Go
- N_max :          Mds
- classe de moteur retenue :

## La grille

| candidat | D/N | N | mémoire 2N | tient ? | licence, étiquette | licence, fichier LICENSE | HumanEval | verdict |
|---|---|---|---|---|---|---|---|---|
| mistralai/Codestral-22B-v0.1 | non publié | 22 Mds | 44 Go | non | mnpl | non-production, aucun usage en entreprise | 81,1 % | écarté, mémoire et licence |
| Qwen/Qwen2.5-Coder-3B | ≈ 1 780 | 3,09 Mds | 6,2 Go | oui | qwen-research | recherche seulement, pas d'usage commercial | 52,4 % | écarté, licence |
| bigcode/starcoder2-7b | ≈ 490 | 7,17 Mds | 14,3 Go | oui | bigcode-openrail-m | usage commercial permis, avec restrictions d'usage | 35,4 % | écarté, qualité |
| deepseek-ai/deepseek-coder-6.7b-base | ≈ 300 | 6,7 Mds | 13,4 Go | oui | deepseek-license | usage commercial permis, avec restrictions d'usage | 49,4 % | écarté, qualité inférieure au 7B |
| Qwen/Qwen2.5-Coder-7B | ≈ 720 | 7,62 Mds | 15,2 Go | oui | apache-2.0 | permissive, usage commercial libre | 61,6 % | retenu |
| Qwen/Qwen2.5-Coder-1.5B | ≈ 3 570 | 1,54 Mds | 3,1 Go | oui | apache-2.0 | permissive, usage commercial libre | 43,9 % | témoin |

## La décision

- moteur retenu :
- témoin :
- pourquoi, en trois phrases :

## La mesure interne

Bits par caractère du moteur retenu sur le test de la séance 2.

| bloc | bits par caractère |
|---|---|
| code |  |
| anglais |  |
| français |  |
| darija |  |
