from collections import Counter
import math
import random

import numpy as np

from businessnlp.normalize import normalize, make_ngrams
import businessnlp.duck as duck
import businessnlp.data as data
from businessnlp.graph import plot_vectors_2d


def term_frequency_map(tokens):
    return Counter(tokens)


def term_frequency(term, tokenized_document):
    document_frequency_map = term_frequency_map(tokenized_document)
    return document_frequency_map.get(term, 0) / len(tokenized_document)


def document_frequency_map(tokenized_corpus):
    document_frequency = Counter()
    for doc in tokenized_corpus:
        document_frequency.update(set(doc))
    return document_frequency


def inverse_document_frequency(term, document_frequency, total_documents):
    return math.log10(1 + total_documents / (document_frequency[term] + 1)) + 1


def tfidf(names, mode="tokens", ngram_size=3):
    """
    Compute TF-IDF vectors for a list of names.
    mode = "tokens" (word-level tokens after normalize)
         or "ngrams" (character-level ngrams of size ngram_size).
    """

    if mode == "tokens":
        # Get normalized list of tokens
        corpus = [normalize(name, return_tokens=True) for name in names]
    elif mode == "ngrams":
        # Character n-grams
        collapsed = [normalize(name) for name in names]
        corpus = [make_ngrams(list(doc), n=ngram_size) for doc in collapsed]

    else:
        raise ValueError(f"Unknown mode {mode}")

    total_documents = len(corpus)

    # Frequency mapping from term to number of docs it appears in
    document_frequency = document_frequency_map(corpus)

    # Vocabulary and term index mapping
    vocabulary = sorted(list(document_frequency.keys()))
    terms = {term: i for i, term in enumerate(vocabulary)}

    # Compute TF-IDF vectors
    vectors = []
    for tokenized_document in corpus:
        vector = np.zeros(len(vocabulary), dtype=np.float64)
        for term in tokenized_document:
            tf = term_frequency(term, tokenized_document)
            idf = inverse_document_frequency(term, document_frequency, total_documents)
            term_idx = terms[term]
            vector[term_idx] += tf * idf

        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        vectors.append(vector)

    return vectors, vocabulary


def tfidf_demo(names, query_size=2):
    """
    Demo TF-IDF using both token-level and n-gram-level normalization.
    Inserts into SQL and shows cosine similarity results.
    """

    # Tokens demo
    print("\nwhole word tokens")
    vectors, vocabulary = tfidf(names, mode="tokens")

    table_name = "tfidf_demo_tokens"
    duck.setup_table(table_name, len(vocabulary))
    for name, vector in zip(names, vectors):
        normalized_name = " ".join(normalize(name, return_tokens=True))
        duck.insert_np_array(table_name, normalized_name, vector)

    sample_indices = random.sample(list(range(len(names))), query_size)
    for idx in sample_indices:
        name, vector = names[idx], vectors[idx]
        normalized_name = " ".join(normalize(name, return_tokens=True))
        print(f"\n[cosine distance] query='{normalized_name}'")
        for result in duck.cosine_distance_nearest_vectors(table_name, vector, 3):
            print(result)

    # N-grams (trigram mode)
    print("\ncharacter trigrams")
    vectors, vocabulary = tfidf(names, mode="ngrams", ngram_size=3)

    table_name = "tfidf_demo_ngrams"
    duck.setup_table(table_name, len(vocabulary))
    for name, vector in zip(names, vectors):
        normalized_name = " ".join(normalize(name, return_tokens=True))
        duck.insert_np_array(table_name, normalized_name, vector)

    for idx in sample_indices:
        name, vector = names[idx], vectors[idx]
        normalized_name = " ".join(normalize(name, return_tokens=True))
        print(f"\n[cosine distance] query='{normalized_name}'")
        for result in duck.cosine_distance_nearest_vectors(table_name, vector, 3):
            print(result)


def demo(sample_size=None, query_size=2):


    if sample_size is None:
        words = data.load_text_file("companies")
        overlaps = data.load_text_file("company_overlaps")
        spelling_variations = data.load_text_file("company_variations")
    else:
        words = random.sample(data.load_text_file("companies"), sample_size)
        overlaps = random.sample(data.load_text_file("company_overlaps"), sample_size)
        spelling_variations = random.sample(data.load_text_file("company_variations"), sample_size)

    print()
    print("tfidf normal demo")
    print("--------------------------------")
    tfidf_demo(words, query_size=query_size)
    print()

    print()
    print("tfidf overlap demo")
    print("--------------------------------")
    tfidf_demo(overlaps, query_size=query_size)
    print()

    print()
    print("tfidf spelling variations demo")
    print("--------------------------------")
    tfidf_demo(spelling_variations, query_size=query_size)
    print()

if __name__ == "__main__":
    print()
    print(
        """
TF-IDF (Term Frequency–Inverse Document Frequency)

A simple way to turn text into numbers by counting how often words appear, while removing the importance of very common words like “the” or “and.”
It highlights words that are important for distinguishing one document from another.

Trigrams: measures surface-level string overlap
Tokens: measures semantic / word-level overlap
"""
    )
    print()
    demo()
