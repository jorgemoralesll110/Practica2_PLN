from typing import List, Tuple
import re
import unicodedata

TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", flags=re.UNICODE)

# Función para cargar y tokenizar las frases
def load_and_tokenize(
    path: str,
    lowercase: bool = True,
    strip_accents: bool = False,
    keep_numbers: bool = False,
    min_token_len: int = 2,
) -> Tuple[List[List[str]], List[str]]:

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    raw = normalize_text(raw, lowercase=lowercase, strip_accents=strip_accents)

    sentences: List[List[str]] = []
    all_words: List[str] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        tokens = TOKEN_RE.findall(line)

        clean_tokens = []
        for tok in tokens:
            if not keep_numbers and tok.isdigit():
                continue
            if len(tok) < min_token_len:
                continue
            clean_tokens.append(tok)

        if clean_tokens:
            sentences.append(clean_tokens)
            all_words.extend(clean_tokens)

    return sentences, all_words


def normalize_text(text: str, lowercase: bool = True, strip_accents: bool = False) -> str:
    text = unicodedata.normalize("NFC", text)

    if lowercase:
        text = text.casefold()

    if strip_accents:
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = unicodedata.normalize("NFC", text)

    return text

