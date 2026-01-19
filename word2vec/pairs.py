from typing import List, Tuple

def generate_skipgram_pairs(sentences: List[List[str]], vocab: dict, window_size: int = 2) -> List[Tuple[int, int]]:
    pairs = []
    for sentence in sentences:
        index = [vocab[w] for w in sentence if w in vocab]
        for i, center in enumerate(index):
            start = max(0, i - window_size)
            end = min(len(index), i + window_size + 1)
            for j in range(start, end):
                if i == j:
                    continue
                pairs.append((center, index[j]))
    return pairs