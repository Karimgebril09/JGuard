from __future__ import annotations

import re
import unicodedata


PUNCT_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        "…": ".",
        "，": ",",
        "。": ".",
        "：": ":",
        "；": ";",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
    }
)

MULTI_PUNCT_PATTERNS = (
    (re.compile(r"\.{2,}"), "."),
    (re.compile(r"!{2,}"), "!"),
    (re.compile(r"\?{2,}"), "?"),
    (re.compile(r",{2,}"), ","),
    (re.compile(r";{2,}"), ";"),
    (re.compile(r":{2,}"), ":"),
    (re.compile(r"-{2,}"), "-"),
)

def normalize_punctuation(text: str) -> tuple[str, int]:
    updated = text.translate(PUNCT_TRANSLATION)
    replaced_count = 0
    for old, new in PUNCT_TRANSLATION.items():
        old_char = chr(old) if isinstance(old, int) else old
        if old_char != new and old_char in text:
            replaced_count += text.count(old_char)

    before = updated
    for pattern, replacement in MULTI_PUNCT_PATTERNS:
        updated = pattern.sub(replacement, updated)

    if updated != before:
        replaced_count += 1

    return updated, replaced_count


def canonicalize_stage6(raw_input: str) -> dict[str, object]:

    text = unicodedata.normalize("NFKC", raw_input)

    punctuation_text, punct_changes = normalize_punctuation(text)
    collapsed_text = re.sub(r"\s+", " ", punctuation_text).strip()

    whitespace_collapsed = punctuation_text != collapsed_text
    punctuation_normalized = punct_changes > 0

    return {
        "original_text": raw_input,
        "canonical_text": collapsed_text,
        "whitespace_collapsed": whitespace_collapsed,
        "punctuation_normalized": punctuation_normalized,
        "metadata": {
            "input_type": type(raw_input).__name__,
            "normalization_form": "NFKC",
            "punctuation_changes": punct_changes,
        },
    }
