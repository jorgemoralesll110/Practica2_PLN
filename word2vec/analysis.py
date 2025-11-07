# En este fichero se implementan las herramientas para analizar los embeddings: los vecinos más cercanos y las analogías

from typing import List
import numpy as np


def print_neighbors(model, vocab, inv_vocab, words: List[str], top_k: int = 5):
    print("\n ********** VECINOS MÁS CERCANOS **********")
    for word in words:
        if word not in vocab:
            continue
        print(f"\n'{word}':")
        idx = vocab[word]
        neighbors = model.nearest_neighbors(idx, top_k=top_k)
        for j, sim in neighbors:
            print(f"  - {inv_vocab[j]}: {sim:.4f}")

def analogy_test(model, vocab, inv_vocab, a: str, b: str, c: str, top_k: int = 3):
    if a not in vocab or b not in vocab or c not in vocab:
        return None
    va = model.get_embeddings(vocab[a])
    vb = model.get_embeddings(vocab[b])
    vc = model.get_embeddings(vocab[c])
    target = model.analogy_vector(va, vb, vc)
    sims = []
    for word, idx in vocab.items():
        if word in (a, b, c):
            continue
        v = model.get_embeddings(idx)
        sim = (target @ v) / (np.linalg.norm(target) * np.linalg.norm(v) + 1e-12)
        sims.append((word, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]
