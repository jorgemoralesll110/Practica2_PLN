import numpy as np


class SkipGram:
    def __init__(self, vocab_size:int, embedding_dim: int = 50, learning_rate: float = 0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        rng = np.random.RandomState(0)
        self.W_in = rng.normal(0, 0.01, size = (vocab_size, embedding_dim))
        self.W_out = rng.normal(0, 0.01, size = (embedding_dim, vocab_size))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / (np.sum(ex) + 1e-12)

    def forward(self, center_idx: int):
        h = self.W_in[center_idx]
        u = h @ self.W_out
        y_pred = self._softmax(u)
        return h, u, y_pred

    def backward(self, center_idx: int, context_idx:int, h, y_pred):
        y_true = np.zeros(self.vocab_size)
        y_true[context_idx] = 1.0
        error = y_pred - y_true
        dW_out = np.outer(h, error)
        dW_in = self.W_out @ error
        self.W_out -= self.lr * dW_out
        self.W_in[center_idx] -= self.lr * dW_in

    def train(self, pairs, epochs=100, shuffle=True, verbose_every=10):
        n = len(pairs)
        rng = np.random.default_rng(0)
        for ep in range(1, epochs + 1):
            if shuffle:
                rng.shuffle(pairs)
            total_loss = 0.0
            for c_idx, x_idx in pairs:
                h, u, y_pred = self.forward(c_idx)
                total_loss += -np.log(y_pred[x_idx] + 1e-10)
                self.backward(c_idx, x_idx, h, y_pred)
            if verbose_every and (ep % verbose_every == 0):
                print(f'Epoch {ep}/{epochs}: Loss: {total_loss/n:.4f}')
        return total_loss / n

    def get_embeddings(self, idx: int):
        return self.W_in[idx]

    @staticmethod
    def _cosine(a, b):
        return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

    def nearest_neighbors(self, word_idx: int, top_k = 5):
        v = self.get_embeddings(word_idx)
        sims = []
        for j in range(self.vocab_size):
            if j == word_idx:
                continue
            sims.append((j, self._cosine(v, self.W_in[j])))
        sims.sort(key = lambda x: x[1], reverse = True)
        return sims[:top_k]

    @staticmethod
    def analogy_vector(a, b, c):
        return b - a + c



