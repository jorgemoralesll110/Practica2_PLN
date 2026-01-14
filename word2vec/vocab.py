from collections import Counter
from typing import List

def build_vocabulary(words: List[str], min_count: int = 1):
    counter = Counter(words)
    valid_words = [w for w, c in counter.items() if c >= min_count]
    vocab = {w: idx for idx, w in enumerate(valid_words)}
    inv_vocab = {idx: w for w, idx in vocab.items()}
    return vocab, inv_vocab, counter