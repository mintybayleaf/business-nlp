from collections import Counter
import math

import numpy as np


from businessnlp.normalize import normalize
import businessnlp.sql as sql


def load_business_names_similar():
    with open("./resources/business_names_similar.txt", "r") as file:
        content = file.read()
        return content.splitlines()


def load_business_names_similar_mixed():
    with open("./resources/business_names_similar_mixed.txt", "r") as file:
        content = file.read()
        return content.splitlines()


def load_business_names_mixed():
    with open("./resources/business_names_mixed.txt", "r") as file:
        content = file.read()
        return content.splitlines()


def load_business_semantices():
    with open("./resources/semantic_words.txt", "r") as file:
        content = file.read()
        return content.splitlines()


def term_frequency_map(tokens):
    return Counter(tokens)


def generate_ngram_tokens(tokens, ngram=3):
    combined = "".join(tokens)
    return [combined[i : i + ngram] for i in range(len(combined) - (ngram - 1))]


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


def tfidf(names):
    normalized_tokens = [normalize(name) for name in names]

    # Corpus is all the n-grams
    corpus = [generate_ngram_tokens(tokens) for tokens in normalized_tokens]
    total_documents = len(corpus)

    # Frequency mapping from term to document its in for IDF
    document_frequency = document_frequency_map(corpus)

    # Ordered list of terms so we can compute all the vectors in the same order
    vocabulary = sorted(list(document_frequency.keys()))
    terms = {term: i for i, term in enumerate(vocabulary)}

    # Compute the TF-IDF for each document in the corpus
    vectors = []
    for tokenized_document in corpus:
        vector = np.zeros(len(vocabulary), dtype=np.float64)
        for term in tokenized_document:
            tf = term_frequency(term, tokenized_document)
            idf = inverse_document_frequency(term, document_frequency, total_documents)
            term_idx = terms[term]
            vector[term_idx] += tf * idf

        # Store normalized vectors
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        vectors.append(vector)

    return vectors, vocabulary


def tfidf_demo():
    names = load_business_names_similar_mixed()
    vectors, vocabulary = tfidf(names)

    table_name = "tfidf_demo"
    sql.setup_table(table_name, len(vocabulary))
    for name, vector in zip(names, vectors):
        sql.insert_np_array(table_name, name, vector)

    sample_vectors = [
        vectors[0],
        vectors[20],
        vectors[1000],
        vectors[5000],
        vectors[7500],
        vectors[9500],
    ]
    sample_names = [
        names[0],
        names[20],
        names[1000],
        names[5000],
        names[7500],
        names[9500],
    ]
    for name, vector in zip(sample_names, sample_vectors):
        for result in sql.cosine_distance_nearest_vectors(table_name, vector, 10):
            print(f"[cosine distance] {name} => {result}")


if __name__ == "__main__":
    tfidf_demo()
