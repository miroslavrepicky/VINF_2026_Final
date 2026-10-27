"""
Shared tokenization for indexer.py and searcher.py (it MUST be identical in both).
Lowercase, accents stripped (NFKD), regex [a-z0-9]+, no lemmatization.
"""
import re
import unicodedata

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English function words that are dropped from the index and from queries
STOPWORDS = frozenset("""
a an and are as at be but by for from had has have he her his i in is it its of on or
she that the their them they this to was were which who with
""".split())


def normalize(text):
    """Lowercase and strip diacritics (NFKD decomposition, combining marks removed)."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def tokenize(text, use_stopwords=True):
    """Split text into normalized [a-z0-9]+ tokens; stopwords are removed when use_stopwords is True."""
    toks = TOKEN_RE.findall(normalize(text))
    if use_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks