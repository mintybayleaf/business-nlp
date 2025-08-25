import unicodedata
from io import StringIO

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

english_stopwords = set(stopwords.stop('english'))
ps = PorterStemmer()

"""
Tokenize

Attempts to tokenize a string into words by splitting on common punctuation etc...
"""
def tokenize(string):
    return word_tokenize(string)

"""
Remove Stopwords

Get rid of common words like preposition that have no meaning
"""
def remove_stopwords(tokens):
    return [word for word in tokens if word not in english_stopwords]

"""
Stemming

Turn words back into their root forms from their inflected forms

Dancer -> Danc
Dancing -> Danc
Danced -> Danc
"""
def stemmer(tokens):
    return [ps.stem(word) for word in tokens]

"""
Accent Folding

Removes diacritics from strings by standarizing the code points
then removing the combining characters

"Café, Résumé, Español, StäVänger" -> "Cafe, Resume, Espanol, Stavanger"

"""
def remove_accents(string):
    folded_string = StringIO()
    for character in unicodedata.normalize('NFKD', string):
        if not unicodedata.combining(character):
            folded_string.write(character)
    return folded_string.getvalue()

print(tokenize("a very long string, with something     here."))
print(remove_accents("Café, Résumé, Español, StäVänger"))
print(stemmer(tokenize("bailey loves dancing with zebras in the evening")))