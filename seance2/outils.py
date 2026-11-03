# ============================================================
#  Séance 2 --- outils fournis, à ne pas modifier
#  entrainer_rapide : le même algorithme que entrainer (compter, choisir,
#  fusionner), mais les comptes de paires sont tenus à jour autour de chaque
#  fusion au lieu d'être recomptés sur tout le corpus, et la paire la plus
#  fréquente est tenue en tête d'une file de priorité. Mêmes fusions que la
#  version naïve, en quelques dizaines de secondes au lieu de plusieurs minutes.
# ============================================================
import heapq
from collections import Counter, defaultdict
from bpe import en_octets, BASE, FIN, longueur


def entrainer_rapide(mots, nb_fusions):
    corpus = [list(en_octets(m)) for m in mots]         # une suite par mot distinct
    freqs = list(mots.values())
    paires = Counter()                                   # paire -> compte pondéré
    ou = defaultdict(set)                                # paire -> mots qui l'ont contenue
    for k, (s, f) in enumerate(zip(corpus, freqs)):
        for p in zip(s, s[1:]):
            paires[p] += f
            ou[p].add(k)
    ordre = {p: i for i, p in enumerate(paires)}         # première rencontre, pour le départage
    # file de priorité : la plus fréquente, puis le symbole le plus long, puis la première rencontrée
    file = [(-c, -longueur(p[0] + p[1]), ordre[p], p) for p, c in paires.items()]
    heapq.heapify(file)
    fusions = []
    while len(fusions) < nb_fusions and file:
        c, _, _, meilleure = heapq.heappop(file)
        if -c != paires.get(meilleure, 0):               # entrée périmée, le compte a changé depuis
            continue
        a, b = meilleure
        ab = a + b
        for k in ou[meilleure]:
            s, f = corpus[k], freqs[k]
            n, i, L = [], 0, len(s)
            vieux, neufs = set(), set()                  # indices des paires touchées, dans s et dans n
            while i < L:
                if i < L - 1 and s[i] == a and s[i + 1] == b:
                    j = len(n)
                    vieux.update(x for x in (i - 1, i, i + 1) if 0 <= x < L - 1)
                    neufs.update((j - 1, j))
                    n.append(ab)
                    i += 2
                else:
                    n.append(s[i])
                    i += 1
            if not vieux:
                continue                                 # le mot ne contient plus la paire
            corpus[k] = n
            touchees = set()
            for x in vieux:
                p = (s[x], s[x + 1])
                paires[p] -= f
                touchees.add(p)
            for x in neufs:
                if 0 <= x < len(n) - 1:
                    p = (n[x], n[x + 1])
                    paires[p] += f
                    ou[p].add(k)
                    if p not in ordre:
                        ordre[p] = len(ordre)
                    touchees.add(p)
            for p in touchees:
                if paires[p] <= 0:
                    del paires[p]
                else:
                    heapq.heappush(file, (-paires[p], -longueur(p[0] + p[1]), ordre[p], p))
        paires.pop(meilleure, None)
        fusions.append(meilleure)
    vocabulaire = set(BASE) | {FIN} | {a + b for a, b in fusions}
    return fusions, vocabulaire
