# summarizer.py
"""
Robust summarizer helpers for EchoNotes.

Features:
- NLTK resource auto-check & download (punkt, stopwords) if available.
- Uses Sumy (LexRank / LSA / TextRank) when installed.
- Fallback frequency-based summarizer that works without Sumy/NLTK.
- URL extraction via newspaper3k when available (best-effort).
- PDF text extraction via PyPDF2 (best-effort).
- Keyword extraction, action-item heuristics, reading time.
"""
from collections import Counter
import math
import re
import io

# lazy / optional imports
HAS_SUMY = False
HAS_NLTK = False
HAS_NEWSPAPER = False
HAS_PYPDF2 = False

try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words

    HAS_SUMY = True
except Exception:
    HAS_SUMY = False

try:
    import nltk

    def _ensure_nltk_resource(path, download_name=None):
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            try:
                nltk.download(download_name or path.split('/')[-1], quiet=True)
                return True
            except Exception:
                return False

    punkt_ok = _ensure_nltk_resource('tokenizers/punkt', 'punkt')
    sw_ok = _ensure_nltk_resource('corpora/stopwords', 'stopwords')
    HAS_NLTK = punkt_ok and sw_ok
except Exception:
    HAS_NLTK = False

try:
    from newspaper import Article

    HAS_NEWSPAPER = True
except Exception:
    Article = None
    HAS_NEWSPAPER = False

try:
    import PyPDF2

    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False

LANG = "english"

if HAS_NLTK:
    from nltk.corpus import stopwords

    _stopwords = set(stopwords.words(LANG))
else:
    _stopwords = set(
        ["the", "and", "is", "in", "it", "of", "to", "a", "for", "on", "that", "this", "with", "as", "are", "was", "be",
         "by", "an", "or"]
    )


# -----------------------------
# Text extraction helpers
# -----------------------------
def get_text_from_url(url):
    """Best-effort article text extraction using newspaper3k (if available)."""
    if not HAS_NEWSPAPER:
        return ""
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        if not text or len(text.strip()) < 40:
            return ""
        return text
    except Exception:
        return ""


def extract_text_from_pdfbytes(file_bytes):
    """Extract text from PDF bytes using PyPDF2 (best-effort)."""
    if not HAS_PYPDF2:
        return ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        parts = []
        for page in reader.pages:
            try:
                txt = page.extract_text()
                if txt:
                    parts.append(txt)
            except Exception:
                continue
        return "\n\n".join(parts).strip()
    except Exception:
        return ""


# -----------------------------
# Summarizers
# -----------------------------
def _sumy_summarize(text, n_sentences=3, method="lexrank"):
    """Sumy-based extractive summarizer (requires sumy + nltk)."""
    if not HAS_SUMY or not HAS_NLTK:
        raise RuntimeError("Sumy or NLTK not available")
    parser = PlaintextParser.from_string(text, Tokenizer(LANG))
    stemmer = Stemmer(LANG)
    method = (method or "lexrank").lower()
    if method == "lexrank":
        summarizer = LexRankSummarizer(stemmer)
    elif method == "lsa":
        summarizer = LsaSummarizer(stemmer)
    elif method == "textrank":
        summarizer = TextRankSummarizer(stemmer)
    else:
        summarizer = LexRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANG)
    sentences = summarizer(parser.document, n_sentences)
    return " ".join(str(s) for s in sentences)


def _fallback_summarize(text, n_sentences=3):
    """
    Frequency-based extractive summarizer (no external tokenizers):
    - naive sentence split
    - score sentences by frequency of important words
    - returns top n_sentences in original order
    """
    text = text.strip()
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    words = re.findall(r'\w+', text.lower())
    words = [w for w in words if w.isalpha() and w not in _stopwords and len(w) > 2]
    if not words or not sentences:
        return "Input too short to summarize reliably."
    freq = Counter(words)
    sent_scores = []
    for i, s in enumerate(sentences):
        tokens = re.findall(r'\w+', s.lower())
        score = sum(freq.get(t, 0) for t in tokens)
        sent_scores.append((i, max(0, score), s.strip()))
    sent_scores_sorted = sorted(sent_scores, key=lambda x: x[1], reverse=True)[:n_sentences]
    sent_scores_sorted.sort(key=lambda x: x[0])
    selected = [s for (_, _, s) in sent_scores_sorted if s]
    if not selected:
        return "Could not generate summary."
    return " ".join(selected)


def summarize(text, sentences=3, method="lexrank"):
    """Public summarizer: try Sumy/NLTK first, otherwise fallback."""
    if not text or len(text.split()) < 6:
        return "Input too short to summarize reliably."
    if HAS_SUMY and HAS_NLTK:
        try:
            return _sumy_summarize(text, n_sentences=sentences, method=method)
        except Exception:
            pass
    # fallback
    try:
        return _fallback_summarize(text, n_sentences=sentences)
    except Exception:
        return "Unable to summarize (internal error)."


# -----------------------------
# Keywords, actions, metadata
# -----------------------------
def extract_keywords(text, top_n=10):
    text_clean = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [t for t in text_clean.split() if t.isalpha() and t not in _stopwords and len(t) > 2]
    if not tokens:
        return []
    counts = Counter(tokens)
    common = counts.most_common(top_n)
    return [w for w, _ in common]


def estimate_reading_time(text, wpm=200):
    words = max(1, len(re.findall(r'\w+', text)))
    minutes = words / wpm
    if minutes < 1:
        return f"{int(words)} words (~{int(minutes * 60)} sec)"
    else:
        return f"{math.ceil(minutes)} min read"


def extract_action_items(text, max_items=5):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    candidates = []
    action_words = {"should", "must", "need", "todo", "todo:", "action", "please", "ensure", "avoid", "fix", "update",
                    "implement", "schedule"}
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        if any(w in s_clean.lower() for w in action_words):
            candidates.append(s_clean)
            continue
        words = re.findall(r'\w+', s_clean)
        if words and words[0].lower() not in _stopwords and len(s_clean.split()) <= 20 and s_clean[0].isupper():
            candidates.append(s_clean)
    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)
        if len(unique) >= max_items:
            break
    return unique
# End of summarizer.py