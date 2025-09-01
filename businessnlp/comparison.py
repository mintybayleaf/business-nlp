import plotly.express as px
from plotly.subplots import make_subplots
from plotly import graph_objects as go
from sklearn.manifold import TSNE
import numpy as np
import random
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
    Reduce vectors to 2D using t-SNE and plot with Plotly (interactive).
    Hover over points to see labels.
    """
    # Subsample if too many
    if len(vectors) > sample_size:
        indices = random.sample(range(len(vectors)), sample_size)
        vectors = np.array([vectors[i] for i in indices])
        labels = [labels[i] for i in indices]
    else:
        vectors = np.array(vectors)

    # Squish high-dimensional vectors down to 2D
    vectors_2d = TSNE(n_components=2, random_state=42).fit_transform(vectors)

    # Interactive scatter plot
    fig = px.scatter(
        x=vectors_2d[:, 0],
        y=vectors_2d[:, 1],
        text=labels,        # labels appear on hover
        hover_name=labels,  # also shows in hover box
        title=title,
        width=900,
        height=600
    )

    # Style adjustments
    fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color="DarkSlateGrey")))
    fig.update_layout(
        xaxis=dict(showticklabels=False, zeroline=False),
        yaxis=dict(showticklabels=False, zeroline=False)
    )

    fig.show()



def prepare_vectors(names, mode="tfidf", tokens=False):
    """
    Compute vectors for TF-IDF, BM25, or BERT embeddings.
    """
    if mode == "tfidf":
        vectors, _ = tfidf(names, mode="tokens" if tokens else "ngrams", ngram_size=3)
    elif mode == "bm25":
        # BM25 token corpus
        corpus = [normalize(name, return_tokens=tokens) for name in names]
        bm = BM25(corpus)
        vectors, _ = bm.get_vectors()
    elif mode == "embeddings":
        vectors, _ = generate_embeddings(names, mode="tokens" if tokens else "ngrams", ngram_size=3)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return vectors


def vector_space_demo(tokens=False):
    names = data.load_text_file("demo_names")

    # Helper: reduce high-dimensional vectors to 2D
    def reduce(vectors, sample_size=50):
        if len(vectors) > sample_size:
            indices = random.sample(range(len(vectors)), sample_size)
            vectors = np.array([vectors[i] for i in indices])
            labels = [names[i] for i in indices]
        else:
            vectors = np.array(vectors)
            labels = names
        reduced = TSNE(n_components=2, random_state=42).fit_transform(vectors)
        return reduced, labels

    # Prepare 2D projections
    tfidf_vecs = prepare_vectors(names, mode="tfidf", tokens=tokens)
    tfidf_2d, tfidf_labels = reduce(tfidf_vecs)

    bm25_vecs = prepare_vectors(names, mode="bm25", tokens=tokens)
    bm25_2d, bm25_labels = reduce(bm25_vecs)

    embedding_vecs = prepare_vectors(names, mode="embeddings", tokens=tokens)
    emb_2d, emb_labels = reduce(embedding_vecs)

    # Make subplot grid (1 row, 3 columns)
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("TF-IDF Token Vectors", "BM25 Token Vectors", "BERT Embeddings")
    )

    # Add scatter plots with hover labels
    fig.add_trace(
        go.Scatter(
            x=tfidf_2d[:, 0], y=tfidf_2d[:, 1],
            mode="markers",
            text=tfidf_labels, hovertext=tfidf_labels,
            marker=dict(size=8, opacity=0.7, line=dict(width=1, color="DarkSlateGrey")),
            name="TF-IDF"
        ), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=bm25_2d[:, 0], y=bm25_2d[:, 1],
            mode="markers",
            text=bm25_labels, hovertext=bm25_labels,
            marker=dict(size=8, opacity=0.7, line=dict(width=1, color="DarkSlateGrey")),
            name="BM25"
        ), row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=emb_2d[:, 0], y=emb_2d[:, 1],
            mode="markers",
            text=emb_labels, hovertext=emb_labels,
            marker=dict(size=8, opacity=0.7, line=dict(width=1, color="DarkSlateGrey")),
            name="Embeddings"
        ), row=1, col=3
    )

    # Layout tweaks
    fig.update_layout(
        title="Comparison of Business Name Vector Spaces",
        height=1080, width=1920,
        showlegend=True
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    fig.show()


# “t-SNE squishes your high-dimensional vectors onto a flat map so
# that things that are ‘neighbors’ in the original space stay neighbors in the map

if __name__ == "__main__":
    vector_space_demo(tokens=True)
    vector_space_demo(tokens=False)
