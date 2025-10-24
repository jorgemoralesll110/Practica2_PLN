import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import re


# ============================================================================
# 1. PREPROCESAMIENTO Y CARGA DE DATOS
# ============================================================================

def cargar_y_tokenizar(ruta_archivo):
    """
    Carga el archivo de texto y tokeniza las frases.
    Convierte a minúsculas y divide por espacios.
    """
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        texto = f.read()

    # Dividir en líneas y tokenizar
    frases = [linea.strip().split() for linea in texto.split('\n') if linea.strip()]

    # Aplanar todas las palabras para crear vocabulario
    todas_palabras = [palabra for frase in frases for palabra in frase]

    return frases, todas_palabras


def construir_vocabulario(palabras, min_count=1):
    """
    Construye el vocabulario eliminando palabras poco frecuentes.
    Retorna diccionarios de palabra->id e id->palabra.
    """
    contador = Counter(palabras)

    # Filtrar palabras por frecuencia mínima y crear índices consecutivos
    palabras_validas = [palabra for palabra, count in contador.items() if count >= min_count]
    vocab = {palabra: idx for idx, palabra in enumerate(palabras_validas)}

    vocab_inverso = {idx: palabra for palabra, idx in vocab.items()}

    print(f"Tamaño del vocabulario: {len(vocab)} palabras")
    print(f"Palabras más frecuentes: {contador.most_common(10)}")

    return vocab, vocab_inverso, contador


# ============================================================================
# 2. GENERACIÓN DE PARES (CENTRO, CONTEXTO)
# ============================================================================

def generar_pares_skipgram(frases, vocab, window_size=2):
    """
    Genera pares (palabra_centro, palabra_contexto) usando una ventana deslizante.
    """
    pares = []

    for frase in frases:
        # Convertir palabras a índices (ignorar palabras fuera del vocabulario)
        indices = [vocab[palabra] for palabra in frase if palabra in vocab]

        # Para cada palabra en la frase
        for i, centro_idx in enumerate(indices):
            # Definir ventana de contexto
            inicio = max(0, i - window_size)
            fin = min(len(indices), i + window_size + 1)

            # Generar pares con palabras de contexto
            for j in range(inicio, fin):
                if i != j:  # No emparejar palabra consigo misma
                    contexto_idx = indices[j]
                    pares.append((centro_idx, contexto_idx))

    print(f"Total de pares generados: {len(pares)}")
    return pares


# ============================================================================
# 3. MODELO SKIP-GRAM
# ============================================================================

class SkipGram:
    """
    Implementación de Skip-gram con matrices de embeddings y softmax.
    """

    def __init__(self, vocab_size, embedding_dim=50, learning_rate=0.01):
        """
        Inicializa las matrices de embeddings.

        W_in: matriz de entrada (vocab_size x embedding_dim)
        W_out: matriz de salida (embedding_dim x vocab_size)
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate

        # Inicialización aleatoria pequeña
        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_out = np.random.randn(embedding_dim, vocab_size) * 0.01

    def softmax(self, x):
        """Softmax numéricamente estable."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, centro_idx):
        """
        Forward pass: obtiene el embedding y calcula probabilidades.
        """
        # Embedding de la palabra centro
        h = self.W_in[centro_idx]  # (embedding_dim,)

        # Scores para todas las palabras del vocabulario
        u = np.dot(h, self.W_out)  # (vocab_size,)

        # Probabilidades con softmax
        y_pred = self.softmax(u)

        return h, u, y_pred

    def backward(self, centro_idx, contexto_idx, h, y_pred):
        """
        Backward pass: calcula gradientes y actualiza pesos.
        """
        # Vector one-hot del contexto real
        y_true = np.zeros(self.vocab_size)
        y_true[contexto_idx] = 1

        # Error: diferencia entre predicción y realidad
        e = y_pred - y_true  # (vocab_size,)

        # Gradiente para W_out
        dW_out = np.outer(h, e)  # (embedding_dim, vocab_size)

        # Gradiente para W_in
        dW_in = np.dot(self.W_out, e)  # (embedding_dim,)

        # Actualización de pesos
        self.W_out -= self.lr * dW_out
        self.W_in[centro_idx] -= self.lr * dW_in

    def entrenar(self, pares, epochs=100, verbose=True):
        """
        Entrena el modelo con los pares generados.
        """
        n_pares = len(pares)

        for epoch in range(epochs):
            loss_total = 0

            # Mezclar pares para cada época
            np.random.shuffle(pares)

            for centro_idx, contexto_idx in pares:
                # Forward
                h, u, y_pred = self.forward(centro_idx)

                # Calcular pérdida (cross-entropy)
                loss = -np.log(y_pred[contexto_idx] + 1e-10)
                loss_total += loss

                # Backward
                self.backward(centro_idx, contexto_idx, h, y_pred)

            # Mostrar progreso
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Época {epoch + 1}/{epochs}, Loss: {loss_total / n_pares:.4f}")

    def get_embedding(self, palabra_idx):
        """Retorna el embedding de una palabra."""
        return self.W_in[palabra_idx]

    def similitud_coseno(self, vec1, vec2):
        """Calcula similitud coseno entre dos vectores."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

    def vecinos_mas_cercanos(self, palabra_idx, vocab_inverso, top_k=5):
        """
        Encuentra las palabras más similares a una palabra dada.
        """
        embedding_palabra = self.get_embedding(palabra_idx)
        similitudes = []

        for idx in range(self.vocab_size):
            if idx != palabra_idx:
                embedding_otro = self.get_embedding(idx)
                sim = self.similitud_coseno(embedding_palabra, embedding_otro)
                similitudes.append((vocab_inverso[idx], sim))

        # Ordenar por similitud descendente
        similitudes.sort(key=lambda x: x[1], reverse=True)

        return similitudes[:top_k]

    def analogia(self, palabra_a, palabra_b, palabra_c, vocab, vocab_inverso, top_k=5):
        """
        Resuelve analogías: a es a b como c es a ?
        Ejemplo: parís es a francia como madrid es a ?
        """
        if palabra_a not in vocab or palabra_b not in vocab or palabra_c not in vocab:
            return None

        # Vectores de las palabras
        vec_a = self.get_embedding(vocab[palabra_a])
        vec_b = self.get_embedding(vocab[palabra_b])
        vec_c = self.get_embedding(vocab[palabra_c])

        # Operación vectorial: b - a + c
        vec_resultado = vec_b - vec_a + vec_c

        # Buscar palabra más cercana al resultado
        similitudes = []
        for palabra, idx in vocab.items():
            if palabra not in [palabra_a, palabra_b, palabra_c]:
                embedding = self.get_embedding(idx)
                sim = self.similitud_coseno(vec_resultado, embedding)
                similitudes.append((palabra, sim))

        similitudes.sort(key=lambda x: x[1], reverse=True)
        return similitudes[:top_k]


# ============================================================================
# 4. FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    # Configuración
    RUTA_ARCHIVO = 'resources/dataset_word2vec.txt'
    WINDOW_SIZE = 2
    EMBEDDING_DIM = 50
    LEARNING_RATE = 0.05
    EPOCHS = 100
    MIN_COUNT = 2

    print("=" * 70)
    print("PRÁCTICA WORD2VEC - SKIP-GRAM")
    print("=" * 70)

    # 1. Cargar y preprocesar datos
    print("\n1. Cargando y tokenizando datos...")
    frases, todas_palabras = cargar_y_tokenizar(RUTA_ARCHIVO)

    # 2. Construir vocabulario
    print("\n2. Construyendo vocabulario...")
    vocab, vocab_inverso, contador = construir_vocabulario(todas_palabras, MIN_COUNT)

    # 3. Generar pares
    print("\n3. Generando pares (centro, contexto)...")
    pares = generar_pares_skipgram(frases, vocab, WINDOW_SIZE)

    # 4. Entrenar modelo
    print("\n4. Entrenando modelo Skip-gram...")
    modelo = SkipGram(len(vocab), EMBEDDING_DIM, LEARNING_RATE)
    modelo.entrenar(pares, EPOCHS, verbose=True)

    # 5. Análisis de resultados
    print("\n" + "=" * 70)
    print("ANÁLISIS DE EMBEDDINGS")
    print("=" * 70)

    # Vecinos más cercanos para varias palabras
    palabras_test = ['perro', 'gata', 'coche', 'parís', 'francia',
                     'niña', 'agua', 'casa', 'maestro', 'médico']

    print("\n--- VECINOS MÁS CERCANOS ---")
    for palabra in palabras_test:
        if palabra in vocab:
            vecinos = modelo.vecinos_mas_cercanos(vocab[palabra], vocab_inverso, top_k=5)
            print(f"\n'{palabra}':")
            for vecino, sim in vecinos:
                print(f"  - {vecino}: {sim:.4f}")

    # Analogías
    print("\n" + "=" * 70)
    print("--- ANALOGÍAS ---")

    analogias_test = [
        ('parís', 'francia', 'madrid'),
        ('perro', 'ladra', 'gato'),
        ('niño', 'niña', 'maestro'),
        ('doctor', 'médico', 'doctora'),
    ]

    for a, b, c in analogias_test:
        resultado = modelo.analogia(a, b, c, vocab, vocab_inverso, top_k=3)
        if resultado:
            print(f"\n{a} : {b} :: {c} : ?")
            for palabra, sim in resultado:
                print(f"  - {palabra}: {sim:.4f}")
        else:
            print(f"\n{a} : {b} :: {c} : ? -> Palabras no encontradas en vocabulario")

    print("\n" + "=" * 70)
    print("HIPERPARÁMETROS USADOS:")
    print("=" * 70)
    print(f"Tamaño de ventana: {WINDOW_SIZE}")
    print(f"Dimensión de embeddings: {EMBEDDING_DIM}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Épocas: {EPOCHS}")
    print(f"Frecuencia mínima: {MIN_COUNT}")

    return modelo, vocab, vocab_inverso


# ============================================================================
# EJECUTAR
# ============================================================================

if __name__ == "__main__":
    modelo, vocab, vocab_inverso = main()