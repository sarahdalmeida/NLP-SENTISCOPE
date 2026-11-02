import re

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MENTION = re.compile(r"(?<!\w)@\w+")
_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_SENTENCE_BOUNDARY = re.compile(r"""(?<=[.!?])["'’”)\]]*(?:\s+|$)|[\r\n]+""")
_IRREGULAR_NEGATION = re.compile(r"\b(?:won['’]t|shan['’]t|ain['’]t)\b", re.IGNORECASE)
_NEGATION_CONTRACTIONS = re.compile(
    r"\b(are|ca|could|did|does|do|had|has|is|must|should|was|were|will|would)n['’]t\b",
    re.IGNORECASE,
)


def _expand_negation(match: re.Match[str]) -> str:
    stem = match.group(1)
    return f"{'can' if stem == 'ca' else stem} not"


def preprocess_text(text: str) -> str:
    """Normalize social text without dropping negation or hashtag words."""
    normalized = text.lower()
    normalized = _URL.sub(" urltoken ", normalized)
    normalized = _MENTION.sub(" usertoken ", normalized)
    normalized = normalized.replace("#", "")
    normalized = _IRREGULAR_NEGATION.sub(
        lambda match: "will not" if match.group().lower().startswith("won") else "not",
        normalized,
    )
    normalized = _NEGATION_CONTRACTIONS.sub(_expand_negation, normalized)
    normalized = normalized.replace("’", "'")
    normalized = _PUNCTUATION.sub(" ", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def bigram_analyzer(text: str) -> list[str]:
    """Extract adjacent normalized word pairs without crossing sentence boundaries."""
    features: list[str] = []
    for sentence in _SENTENCE_BOUNDARY.split(text):
        tokens = preprocess_text(sentence).split()
        features.extend(
            f"{left} {right}"
            for left, right in zip(tokens, tokens[1:])
        )
    return features
