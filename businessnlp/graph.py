import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random


def plot_vectors_2d(vectors, labels, title="Vector Space", sample_size=100):
    """Reduce to 2D and plot with matplotlib."""
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
