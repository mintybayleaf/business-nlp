import unicodedata
from io import StringIO
from unidecode import unidecode

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


def tokenize(string):
    """
    Attempts to tokenize a string into words
    """
    return word_tokenize(string)


def sanitize(string):
    """
    Converts strings into common formats for downstream processing
    """
    sanitized_string = StringIO()
    for character in string:
        if character.isprintable():
            sanitized_string.write(character.lower())
    return sanitized_string.getvalue()


def accent_folding(string):
    """
    Removes diacritics from strings by standarizing the code points
    then removing the combining characters

    "Café, Résumé, Español, StäVänger" -> "Cafe, Resume, Espanol, Stavanger
    """
    folded_string = StringIO()
    for character in unicodedata.normalize("NFKD", string):
        if not unicodedata.combining(character):
            folded_string.write(character)
    return folded_string.getvalue()


def transliteration(string):
    """
    Mapping from one text system to another trying to perserve the phonetics.
    In this case going from an ideographic characters like Chinese characters

    Example: “上海贸易公司” → “Shanghai Maoyi Gongsi”
    """
    return unidecode(string)


def remove_stopwords(tokens):
    """
    Get rid of common words like prepositions that have no meaning
    """
    return [word for word in tokens if word not in english_stopwords]


def stemmer(tokens):
    """
    Turn words back into their root forms from their inflected forms

    Dancer -> Danc
    Dancing -> Danc
    Danced -> Danc
    """
    return [stemming.stem(word) for word in tokens]


def normalize(string):
    """I forgot how to reduce in python but we are essentially
    reducing by each function above, and getting a final set of tokens"""
    transformers = [
        sanitize,
        transliteration,
        accent_folding,
        tokenize,
        stemmer,
        remove_stopwords,
    ]
    result = string
    for tf in transformers:
        result = tf(result)
    return result


if __name__ == "__main__":
    print(tokenize("a very long string, with something     here."))
    print(accent_folding("Café, Résumé, Español, StäVänger"))
    print(stemmer(tokenize("bailey loves dancing with zebras in the evening")))
    print(transliteration("上海贸易公司"))
    print(
        normalize(
            "bailey loves dancing near the Café 上海贸易公司 CoMpAnY at dawn while he sings in Español"
        )
    )
