import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from collections import Counter
import math

from businessnlp.normalize import normalize
import businessnlp.data as data
from businessnlp.embeddings import generate_embeddings
from businessnlp.tfidf import tfidf


class BM25:
    def __init__(self, corpus, k=1.5, b=0.75):
        self.k = k
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.N

        self.df = {}
        for doc in corpus:
            for word in set(doc):
                self.df[word] = self.df.get(word, 0) + 1

        self.idf = {}
        for word, freq in self.df.items():
            self.idf[word] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

    def score(self, query, index):
        doc = self.corpus[index]
        freq = Counter(doc)
        score = 0.0
        for word in query:
            if word not in freq:
                continue
            f = freq[word]
            numerator = f * (self.k + 1)
            denominator = f + self.k * (1 - self.b + self.b * len(doc) / self.avgdl)
            score += self.idf.get(word, 0) * numerator / denominator
        return score

    def get_vectors(self):
        """
        Represent each document as a vector with BM25 scores for all unique terms.
        This allows us to visualize BM25 in the same way as TF-IDF or embeddings.
        """
        # build vocabulary
        vocabulary = sorted(list(self.df.keys()))
        term_idx = {term: i for i, term in enumerate(vocabulary)}
        vectors = []
        for i, doc in enumerate(self.corpus):
            vec = np.zeros(len(vocabulary))
            for term in doc:
                vec[term_idx[term]] = self.score([term], i)
            vectors.append(vec)
        return np.array(vectors), vocabulary


def plot_vectors_2d(vectors, labels, title="Vector Space", sample_size=100):
    """
    Reduce vectors to 2D using t-SNE and plot with labels.
    """
    if len(vectors) > sample_size:
        indices = random.sample(range(len(vectors)), sample_size)
        vectors = [vectors[i] for i in indices]
        labels = [labels[i] for i in indices]

    vectors_2d = TSNE(n_components=2, random_state=42).fit_transform(vectors)

    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=50)

    for i, label in enumerate(labels):
        plt.text(vectors_2d[i, 0] + 0.5, vectors_2d[i, 1] + 0.5, label, fontsize=8)

    plt.title(title)
    plt.show()


def prepare_vectors(names, mode="tfidf"):
    """
    Compute vectors for TF-IDF, BM25, or BERT embeddings.
    """
    if mode == "tfidf":
        vectors, _ = tfidf(names, mode="tokens")
    elif mode == "bm25":
        # BM25 token corpus
        corpus = [normalize(name, return_tokens=True) for name in names]
        bm = BM25(corpus)
        vectors, _ = bm.get_vectors()
    elif mode == "embeddings":
        vectors, _ = generate_embeddings(names, mode="tokens")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return vectors


def vector_space_demo(sample_size=200):
    names = random.sample(data.load_text_file("company_names"), sample_size)

    # TF-IDF
    tfidf_vecs = prepare_vectors(names, mode="tfidf")
    plot_vectors_2d(tfidf_vecs, names, title="TF-IDF Token Vectors", sample_size=50)

    # BM25
    bm25_vecs = prepare_vectors(names, mode="bm25")
    plot_vectors_2d(bm25_vecs, names, title="BM25 Token Vectors", sample_size=50)

    # BERT embeddings
    embedding_vecs = prepare_vectors(names, mode="embeddings")
    plot_vectors_2d(
        embedding_vecs, names, title="BERT Token Embeddings", sample_size=50
    )


# “t-SNE squishes your high-dimensional vectors onto a flat map so
# that things that are ‘neighbors’ in the original space stay neighbors in the map

if __name__ == "__main__":
    vector_space_demo(sample_size=200)
