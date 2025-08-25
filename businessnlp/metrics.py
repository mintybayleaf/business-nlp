import numpy as np


def hamming_distance(one, two):
    """
    Measures how different two strings are (in this case)
    """
    if len(one) != len(two):
        return 0
    else:
        return sum([two[pos] == character for pos, character in enumerate(one)])


def jaccard_similarity(string_tokens, more_string_tokens):
    """
    Measures the token overlap in two sets, favors longer sets
    """
    tokens_one = set(string_tokens)
    tokens_two = set(more_string_tokens)
    return 1 - (len(tokens_one & tokens_two) / len(tokens_one | tokens_two))


def cosine_similarity(vector, another_vector):
    """
    Computes the cosine similarity between two vectors

    The cosine of two vectors is A dot B / Magnitude(A) * Magnitude(B)
    """
    vector, another_vector = (np.array(vector), np.array(another_vector))
    dot_product = np.dot(vector, another_vector)
    euclidean_distance_multiple = np.linalg.norm(vector) * np.linalg.norm(
        another_vector
    )
    cosine_similarity = dot_product / euclidean_distance_multiple
    radians = np.arccos(cosine_similarity)
    return (cosine_similarity, radians, np.degrees(radians))


if __name__ == "__main__":
    print(hamming_distance("first", "fifth"))
    print(jaccard_similarity(["one", "two", "three"], ["one", "two", "four", "five"]))
    print(cosine_similarity([1, 2, 3], [4, 5, 6]))
