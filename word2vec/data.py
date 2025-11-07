# En este fichero se carga el fichero del corpus y se tokeniza

from typing import List, Tuple

def load_and_tokenize(path: str) -> Tuple[List[List[str]], List[str]]:
    with open (path, 'r', encoding='utf-8') as f:
        text = f.read()
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    sentences = [line.split() for line in lines]
    all_words = [word for sentence in sentences for word in sentence]
    return sentences, all_words
