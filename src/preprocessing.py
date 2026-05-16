"""Text preprocessing pipeline for spam classification."""

from __future__ import annotations

import re

from nltk.stem import PorterStemmer
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import Pipeline

_stemmer = PorterStemmer()

# Compiled once at import time
_URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_EMAIL_RE = re.compile(r'\S+@\S+')
_ALPHA_RE = re.compile(r'[a-z]+')

# Pre-stem sklearn's English stop word list so it matches the stemmed tokens
# that clean_text produces.  Passing raw 'english' to TfidfVectorizer with a
# stemming preprocessor causes a spurious inconsistency warning.
_STOP_WORDS: frozenset[str] = frozenset(
    _stemmer.stem(w) for w in ENGLISH_STOP_WORDS
)


def clean_text(text: str) -> str:
    """Normalize one email string.

    Steps (in order):
      1. Replace email addresses with __email__ before lowercasing so the @
         pattern is unambiguous.
      2. Replace URLs with __url__.
      3. Lowercase everything.
      4. Tokenize into alphabetic words (drops punctuation and digits).
      5. Porter-stem each token; keep __email__ / __url__ placeholders as-is.
    """
    text = _EMAIL_RE.sub('__email__', text)
    text = _URL_RE.sub('__url__', text)
    text = text.lower()

    tokens: list[str] = []
    for raw_token in text.split():
        if raw_token in ('__email__', '__url__'):
            tokens.append(raw_token)
        else:
            for word in _ALPHA_RE.findall(raw_token):
                if len(word) > 1:
                    tokens.append(_stemmer.stem(word))

    return ' '.join(tokens)


def build_pipeline(classifier: ClassifierMixin) -> Pipeline:
    """Return a sklearn Pipeline: TF-IDF vectorizer → classifier.

    TF-IDF settings:
      preprocessor  = clean_text (lowercasing, URL/email replacement, stemming)
      stop_words    = sklearn's built-in English list (~318 words)
      ngram_range   = (1, 2)  — unigrams + bigrams
      min_df        = 2       — drop terms appearing in fewer than 2 documents
      max_df        = 0.95    — drop near-universal terms
      sublinear_tf  = True    — use log(1+tf) to dampen long-email bias
    """
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        stop_words=list(_STOP_WORDS),
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    return Pipeline([('tfidf', vectorizer), ('clf', classifier)])
