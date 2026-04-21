from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any
import unicodedata


_PUNCT_TRANSLATION = str.maketrans(
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

_MULTI_PUNCT_PATTERNS = (
    (re.compile(r"\.{2,}"), "."),
    (re.compile(r"!{2,}"), "!"),
    (re.compile(r"\?{2,}"), "?"),
    (re.compile(r",{2,}"), ","),
    (re.compile(r";{2,}"), ";"),
    (re.compile(r":{2,}"), ":"),
    (re.compile(r"-{2,}"), "-"),
)


def normalize_input(raw_input: str | bytes) -> str:
    if isinstance(raw_input, bytes):
        return raw_input.decode("utf-8", errors="replace")
    return raw_input


def normalize_punctuation(text: str) -> tuple[str, int]:
    updated = text.translate(_PUNCT_TRANSLATION)
    replaced_count = 0
    for old, new in _PUNCT_TRANSLATION.items():
        old_char = chr(old) if isinstance(old, int) else old
        if old_char != new and old_char in text:
            replaced_count += text.count(old_char)

    before = updated
    for pattern, replacement in _MULTI_PUNCT_PATTERNS:
        updated = pattern.sub(replacement, updated)

    if updated != before:
        replaced_count += 1

    # Canonical punctuation spacing.
    updated = re.sub(r"\s+([,.;:!?])", r"\1", updated)
    updated = re.sub(r"([,.;:!?])(\S)", r"\1 \2", updated)

    return updated, replaced_count


@dataclass(frozen=True)
class Stage6CanonicalInput:
    original_text: str
    canonical_text: str
    lowered: bool
    whitespace_collapsed: bool
    punctuation_normalized: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class ObfuscationStage6Canonicalizer:
    def canonicalize(self, raw_input: str | bytes) -> Stage6CanonicalInput:
        original_text = normalize_input(raw_input)

        # NFKC stabilizes compatibility punctuation and spacing before cleanup.
        text = unicodedata.normalize("NFKC", original_text)

        punctuation_text, punct_changes = normalize_punctuation(text)
        lowered_text = punctuation_text.lower()
        collapsed_text = re.sub(r"\s+", " ", lowered_text).strip()

        lowered = punctuation_text != lowered_text
        whitespace_collapsed = lowered_text != collapsed_text
        punctuation_normalized = punct_changes > 0

        return Stage6CanonicalInput(
            original_text=original_text,
            canonical_text=collapsed_text,
            lowered=lowered,
            whitespace_collapsed=whitespace_collapsed,
            punctuation_normalized=punctuation_normalized,
            metadata={
                "input_type": type(raw_input).__name__,
                "normalization_form": "NFKC",
                "punctuation_changes": punct_changes,
            },
        )


def canonicalize_stage6(raw_input: str | bytes) -> Stage6CanonicalInput:
    return ObfuscationStage6Canonicalizer().canonicalize(raw_input)
