from __future__ import annotations

from dataclasses import dataclass, field
import unicodedata
from typing import Any
from defenses.obfuscation.helper import normalize_input


ZERO_WIDTH_CHARACTERS = {
    "\u200b",  # Zero width space
    "\u200c",  # Zero width non-joiner
    "\u200d",  # Zero width joiner
    "\ufeff",  # Zero width no-break space / BOM
    "\u2060",  # Word joiner
}

HOMOGLYPH_MAP = {
    # Cyrillic 
    "а": "a",
    "А": "A",
    "е": "e",
    "Е": "E",
    "о": "o",
    "О": "O",
    "р": "p",
    "Р": "P",
    "с": "c",
    "С": "C",
    "у": "y",
    "У": "Y",
    "х": "x",
    "Х": "X",
    "і": "i",
    "І": "I",
    "ј": "j",
    "Ј": "J",
    "к": "k",
    "К": "K",
    "м": "m",
    "М": "M",
    "т": "t",
    "Т": "T",
    "в": "b",
    "В": "B",
    "н": "h",
    "Н": "H",
    # Greek
    "Α": "A",
    "Β": "B",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "H",
    "Ι": "I",
    "Κ": "K",
    "Μ": "M",
    "Ν": "N",
    "Ο": "O",
    "Ρ": "P",
    "Τ": "T",
    "Υ": "Y",
    "Χ": "X",
    "α": "a",
    "β": "b",
    "γ": "y",
    "δ": "d",
    "ι": "i",
    "κ": "k",
    "ν": "v",
    "ο": "o",
    "ρ": "p",
    "τ": "t",
    "χ": "x",
}


def strip_zero_width_chars(text: str) -> tuple[str, int]:
    kept: list[str] = []
    stripped = 0
    for character in text:
        if character in ZERO_WIDTH_CHARACTERS:
            stripped += 1
            continue
        kept.append(character)
    return "".join(kept), stripped


def resolve_homoglyphs(text: str) -> tuple[str, int]:
    transformed: list[str] = []
    replacements = 0

    for character in text:
        mapped = HOMOGLYPH_MAP.get(character)
        if mapped is not None:
            transformed.append(mapped)
            replacements += 1
            continue

        # Convert fullwidth ASCII to regular ASCII.
        codepoint = ord(character)
        if 0xFF01 <= codepoint <= 0xFF5E:
            transformed.append(chr(codepoint - 0xFEE0))
            replacements += 1
            continue

        # Compatibility fold covers mathematical alphanumeric symbols and many
        # other display variants into their plain equivalents.
        folded = unicodedata.normalize("NFKC", character)
        if folded != character and len(folded) == 1 and ord(folded) < 128:
            transformed.append(folded)
            replacements += 1
            continue

        transformed.append(character)

    return "".join(transformed), replacements


@dataclass(frozen=True)
class Stage3NormalizedInput:
    original_text: str
    normalized_text: str
    zero_width_stripped: int
    homoglyph_replacements: int
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_stage3(raw_input: str | bytes) -> Stage3NormalizedInput:
    original_text = normalize_input(raw_input)

    # Apply canonical then compatibility decomposition before mapping
    nfc_text = unicodedata.normalize("NFC", original_text)
    nfkd_text = unicodedata.normalize("NFKD", nfc_text)

    without_zero_width, stripped_count = strip_zero_width_chars(nfkd_text)
    collapsed_text, replacement_count = resolve_homoglyphs(without_zero_width)

    # Remove combining marks introduced by NFKD to produce stable output
    normalized_text = "".join(
        character
        for character in collapsed_text
        if not unicodedata.category(character).startswith("M")
    )

    return Stage3NormalizedInput(
        original_text=original_text,
        normalized_text=normalized_text,
        zero_width_stripped=stripped_count,
        homoglyph_replacements=replacement_count,
        metadata={
            "input_type": type(raw_input).__name__,
            "normalization_sequence": ["NFC", "NFKD"],
        },
    )
