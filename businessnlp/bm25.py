import math
from collections import Counter
import random

from businessnlp.normalize import normalize
import businessnlp.data as data


def make_char_ngrams(text: str, n=3):
    """Return character n-grams over a collapsed string."""
    if n <= 0:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


class BM25:
    def __init__(self, corpus, k=1.5, b=0.75):
        """
        corpus: List[List[str]]  (each doc as a list of terms)
        """
        self.k = k
        self.b = b
        self.corpus = corpus
        self.N = len(self.corpus)
        self.avgdl = (sum(len(doc) for doc in self.corpus) / self.N) if self.N else 0.0

        # document frequencies
        self.df = {}
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

        # inverse document frequency (BM25+ style with +1 inside the log to keep non-negative)
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

    def score(self, query_terms, index):
        doc = self.corpus[index]
        if not doc:
            return 0.0
        freq = Counter(doc)
        score = 0.0
        dl = len(doc)
        K = (
            self.k * (1 - self.b + self.b * (dl / self.avgdl))
            if self.avgdl > 0
            else self.k
        )
        for term in query_terms:
            f = freq.get(term, 0)
            if f == 0:
                continue
            score += self.idf.get(term, 0.0) * ((f * (self.k + 1)) / (f + K))
        return score

    def search_indices(self, query_terms, top_n=3):
        """Return list of (doc_index, score) sorted by score desc."""
        scores = [(i, self.score(query_terms, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


def bm25_prepare(names, mode="tokens", ngram_size=3):
    """
    Build corpus as List[List[str]].

    mode = "tokens"  -> use normalized word tokens (no collapse)
    mode = "ngrams"  -> use character n-grams from collapsed normalized string
    """
    if mode == "tokens":
        corpus = [normalize(name, return_tokens=True) for name in names]
    elif mode == "ngrams":
        collapsed = [normalize(name) for name in names]  # collapsed, no spaces
        corpus = [make_char_ngrams(doc, n=ngram_size) for doc in collapsed]
    else:
        raise ValueError(f"Unknown mode {mode}")
    return corpus


def bm25_demo(names, mode="tokens", ngram_size=3, sample_size=2, top_n=3):
    """
    Demo BM25 ranking similar to your TF-IDF demo.
    - names: list of business names (strings)
    - mode: "tokens" or "ngrams"
    - ngram_size: n when mode == "ngrams"
    - sample_size: how many random queries to run
    - top_n: how many results to print per query
    """
    # Build corpus and model
    corpus = bm25_prepare(names, mode=mode, ngram_size=ngram_size)
    bm25 = BM25(corpus, b=0.75)

    # Pick random queries from names
    sample_queries = random.sample(names, min(sample_size, len(names)))

    print(
        f"\nbm25 ({'char ' + str(ngram_size) + '-grams' if mode=='ngrams' else 'tokens'})\n"
    )
    for query in sample_queries:
        if mode == "tokens":
            q = normalize(query, return_tokens=True)
        else:
            q = make_char_ngrams(normalize(query), n=ngram_size)

        normalized_name = " ".join(normalize(query, return_tokens=True))
        print(f"[bm25 ranking] query='{normalized_name}'")
        for idx, score in bm25.search_indices(q, top_n=top_n):
            # Show original business name
            normalized_name_token = " ".join(normalize(names[idx], return_tokens=True))
            print(f"{normalized_name_token}: {score:.4f}")
        print()


def demo(sample_size=None):
    # Load datasets (limit corpus size for speed)
    if sample_size is None:
        words = data.load_text_file("companies")
        overlaps = data.load_text_file("company_overlaps")
        spelling_variations = data.load_text_file("company_variations")
    else:
        words = random.sample(data.load_text_file("companies"), sample_size)
        overlaps = random.sample(data.load_text_file("company_overlaps"), sample_size)
        spelling_variations = random.sample(data.load_text_file("company_variations"), sample_size)

    print()
    print("bm25 normal demo")
    print("--------------------------------")
    bm25_demo(words, mode="tokens", sample_size=2, top_n=3)
    bm25_demo(words, mode="tokens", ngram_size=3, sample_size=2, top_n=3)
    print()

    print()
    print("bm25 overlap demo")
    print("--------------------------------")
    bm25_demo(overlaps, mode="tokens", sample_size=2, top_n=3)
    bm25_demo(overlaps, mode="ngrams", ngram_size=3, sample_size=2, top_n=3)
    print()

    print()
    print("bm25 spelling variations demo")
    print("--------------------------------")
    bm25_demo(spelling_variations, mode="tokens", sample_size=2, top_n=3)
    bm25_demo(spelling_variations, mode="ngrams", ngram_size=3, sample_size=2, top_n=3)
    print()


if __name__ == "__main__":
    print()
    print(
        """
BM25

An improved version of TF-IDF that not only scores word importance but also accounts for document length and how often a word appears before it stops adding much value.
It’s designed specifically for ranking search results more effectively.

Trigrams: measures surface-level string overlap
Tokens: measures semantic / word-level overlap
    """
    )
    print()
    demo()
