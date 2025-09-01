import unicodedata
from io import StringIO
from unidecode import unidecode
import string

import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("stopwords")

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

english_stopwords = set(stopwords.words("english"))
stemming = PorterStemmer()


def tokenize(text):
    """Tokenize a string into words."""
    return word_tokenize(text)


def sanitize(text):
    """Lowercase and keep only printable characters."""
    sanitized = StringIO()
    for ch in text:
        if ch.isprintable():
            sanitized.write(ch.lower())
    return sanitized.getvalue()


def accent_folding(text):
    """Remove diacritics (accents)."""
    folded = StringIO()
    for ch in unicodedata.normalize("NFKD", text):
        if not unicodedata.combining(ch):
            folded.write(ch)
    return folded.getvalue()


def transliteration(text):
    """Transliterate ideographic characters into phonetics."""
    return unidecode(text)


def remove_punctuation(text):
    """Remove punctuation from a string."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_stopwords(text):
    """Remove stopwords from a string."""
    return " ".join([w for w in tokenize(text) if w not in english_stopwords])


def stemmer(text):
    """Stem words back to their root forms."""
    return " ".join([stemming.stem(w) for w in tokenize(text)])


def make_ngrams(tokens, n):
    """Create character n-grams from a list of tokens (joined with no spaces)."""
    text = "".join(tokens)  # collapse into single string
    return ["".join(text[i : i + n]) for i in range(len(text) - n + 1)]


def normalize(text, return_tokens=False):
    transformers = [
        sanitize,
        transliteration,
        accent_folding,
        remove_punctuation,
        remove_stopwords,
    ]
    result = text
    for tf in transformers:
        result = tf(result)

    tokens = result.split()

    if return_tokens:
        return tokens
    return "".join(tokens)  # collapsed string for ngram pipeline


def demo():
    sample = "Café, Résumé, Español, StäVänger — The Dancing Company in 上海!"
    print()
    print(
        "================================= normalization demo ================================="
    )
    print()
    print("Original:", sample)
    print("Sanitized:", sanitize(sample))
    print("Transliterated:", transliteration(sample))
    print("Accent folded:", accent_folding(sample))
    print("No punctuation:", remove_punctuation(sample))
    print("Stopwords removed:", remove_stopwords(sample))
    print("Stemmed:", stemmer(sample))
    print("Normalized (collapsed):", normalize(sample))
    print("Normalized (tokens):", normalize(sample, return_tokens=True))


if __name__ == "__main__":
    demo()
