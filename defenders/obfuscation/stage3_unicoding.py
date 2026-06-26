from __future__ import annotations

import unicodedata


ZERO_WIDTH_CHARACTERS = {
    "\u200b",  
    "\u200c",  
    "\u200d",  
    "\ufeff",  
    "\u2060", 
}

HOMOGLYPH_MAP = {
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


def rm_zero_width_chars(text: str) -> tuple[str, int]:
    kept: list[str] = []
    stripped = 0
    for character in text:
        if character in ZERO_WIDTH_CHARACTERS:
            stripped += 1
            continue
        kept.append(character)
    return "".join(kept), stripped


def res_homoglyphs(text: str) -> tuple[str, int]:
    transformed: list[str] = []
    replacements = 0

    for character in text:
        mapped = HOMOGLYPH_MAP.get(character)
        if mapped is not None:
            transformed.append(mapped)
            replacements += 1
            continue

        codepoint = ord(character)
        if 0xFF01 <= codepoint <= 0xFF5E:
            transformed.append(chr(codepoint - 0xFEE0))
            replacements += 1
            continue

        folded = unicodedata.normalize("NFKC", character)
        if folded != character and len(folded) == 1 and ord(folded) < 128:
            transformed.append(folded)
            replacements += 1
            continue

        transformed.append(character)

    return "".join(transformed), replacements


def normalize_stage3(raw_input: str) -> dict[str, object]:

    nfc_text = unicodedata.normalize("NFC", raw_input)
    nfkd_text = unicodedata.normalize("NFKD", nfc_text)

    without_zero_width, stripped_count = rm_zero_width_chars(nfkd_text)
    collapsed_text, replacement_count = res_homoglyphs(without_zero_width)

    normalized_text = "".join(
        character
        for character in collapsed_text
        if not unicodedata.category(character).startswith("M")
    )

    return {
        "original_text": raw_input,
        "normalized_text": normalized_text,
        "zero_width_stripped": stripped_count,
        "homoglyph_replacements": replacement_count,
        "metadata": {
            "input_type": type(raw_input).__name__,
            "normalization_sequence": ["NFC", "NFKD"],
        },
    }
