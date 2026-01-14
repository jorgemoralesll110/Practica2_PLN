from word2vec.data import load_and_tokenize
from word2vec.vocab import build_vocabulary
from word2vec.pairs import generate_skipgram_pairs
from word2vec.model import SkipGram
from word2vec.analysis import print_neighbors, analogy_test
from word2vec.analysis import average_neighbor_similarity

def run_program(
        corpus_path: str = "resources/dataset_word2vec.txt",
        window_size: int = 2,
        embedding_dim: int = 50,
        learning_rate: float = 0.01,
        epochs: int = 100,
        min_count: int = 1,
        show_neighbors: bool = True,
        show_analogies: bool = True,
):
    print("*"*70)
    print("WORD2VEC - EJECUCIÓN DEL PROGRAMA MEDIANTE SKIP-GRAM")
    print("*"*70)

    print("\n Primer paso) Cargar y tokenizar el corpus")
    sentences, all_words = load_and_tokenize(corpus_path, strip_accents=False, keep_numbers=False)

    print("\n Segundo paso) Construir el vocabulario")
    vocab, inv_vocab, counter = build_vocabulary(all_words, min_count=min_count)
    print(f"Tamaño del vocabulario: {len(vocab)}")
    print(f"Top 10 palabras: {counter.most_common(10)}")

    print("\n Tercer paso) Generar el par 'Centro-Contexto'")
    pairs = generate_skipgram_pairs(sentences, vocab, window_size=window_size)
    print(f"Total pairs: {len(pairs)}")

    print("\n Cuarto paso) Entrenar el modelo Skip-Gram")
    model = SkipGram(len(vocab), embedding_dim=embedding_dim, learning_rate=learning_rate)
    model.train(pairs, epochs=epochs, verbose_every=10)

    final_loss = model.train(pairs, epochs=epochs, verbose_every=10)

    print("\n Quinto paso) Evaluación cuantitativa de los embeddings")

    test_words = ["perro", "gato", "coche", "casa"]
    indices = [vocab[w] for w in test_words if w in vocab]

    if indices:
        avg_sim = average_neighbor_similarity(model, indices, top_k=5)
        print(f"Average neighbor similarity: {avg_sim:.4f}")
    else:
        print("No test words found in vocabulary for evaluation.")

    if show_neighbors:
        print("\n" + "*"*70)
        print("ANÁLISIS DE LOS EMBEDDINGS")
        print("*" * 70)
        test_words = ["perro","gato","coche", "parís", "francia", "niña", "agua", "casa", "profesor", "médico"]
        print_neighbors(model, vocab, inv_vocab, test_words, top_k=5)

    if show_analogies:
        print("\n" + "*"*70)
        print("ANÁLISIS DE LAS ANALOGÍAS")
        analogies =  [
            ("parís", "francia", "madrid"),
            ("perro", "ladra", "gato"),
            ("niño", "niña", "profesor")
        ]
        for a, b, c in analogies:
            res = analogy_test(model, vocab, inv_vocab,a, b, c, top_k=3)
            if res is None:
                print(f"\n{a} : {b} :: {c} : ? -> Words not found in vocabulary")
            else:
                print(f"\n{a} : {b} :: {c} : ?")
                for word, sim in res:
                    print(f"     - {word} : {sim:.4f}")

    print("\n" + "*" * 70)
    print("HIPERPARÁMETROS USADOS:")
    print(f"    - Tamaño de la ventana: {window_size}")
    print(f"    - Dimensión del embedding: {embedding_dim}")
    print(f"    - Tasa de aprendizaje: {learning_rate}")
    print(f"    - Épocas: {epochs}")
    print(f"    - Frecuencia mínima: {min_count}")

    return model, vocab, inv_vocab, pairs