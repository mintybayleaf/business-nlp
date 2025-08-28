import math
from collections import Counter
import random

from businessnlp.normalize import normalize
import businessnlp.data as data

# See: https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables


def generate_ngram_tokens(tokens):
    text = "".join(tokens)
    return [text[i : i + 3] for i in range(len(text) - 2)]


class BM25:
    def __init__(self, corpus, k=1.5, b=0.5):
        self.k = k
        self.b = b

        self.corpus = corpus
        self.N = len(self.corpus)
        self.avgdl = sum(len(doc) for doc in self.corpus) / self.N

        # document frequencies
        self.df = {}
        for doc in self.corpus:
            for word in set(doc):
                self.df[word] = self.df.get(word, 0) + 1

        # inverse document frequency
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

    def search(self, query, top_n=3):
        scores = []
        for i in range(self.N):
            scores.append((self.corpus[i], self.score(query, i)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


def bm25_demo(names):
    normalized_tokens = [normalize(name) for name in names]
    corpus = [generate_ngram_tokens(tokens) for tokens in normalized_tokens]
    bm25 = BM25(corpus, b=0.75)

    queries = random.sample(names, 3)
    for query in queries:
        query_trigrams = generate_ngram_tokens(normalize(query))
        print(f"[bm25 ranking] {query}")
        for document, score in bm25.search(query_trigrams):
            print(f"{document}: {score:.4f}")


if __name__ == "__main__":
    print()
    print("===================================")
    print()
    print("demo company names")
    bm25_demo(random.sample(data.load_text_file("company_names"), 10000))
    print()
    print("===================================")
    print()
    print("demo similar names")
    bm25_demo(data.load_text_file("similar_company_names"))
