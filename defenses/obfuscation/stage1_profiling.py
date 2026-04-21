from __future__ import annotations

from dataclasses import dataclass, field
from math import log2
from typing import Any
import unicodedata
from defenses.obfuscation.helper import normalize_input

def calculate_shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0

    frequencies: dict[int, int] = {}
    for byte in data:
        frequencies[byte] = frequencies.get(byte, 0) + 1

    length = len(data)
    entropy = 0.0
    for count in frequencies.values():
        probability = count / length
        entropy -= probability * log2(probability)
    return entropy


def identify_writing_system(code_point: int) -> str:
    block_ranges = [
        (0x0000, 0x007F, "Basic Latin"),
        (0x0080, 0x00FF, "Latin-1 Supplement"),
        (0x0100, 0x017F, "Latin Extended-A"),
        (0x0180, 0x024F, "Latin Extended-B"),
        (0x0250, 0x02AF, "IPA Extensions"),
        (0x0370, 0x03FF, "Greek and Coptic"),
        (0x0400, 0x04FF, "Cyrillic"),
        (0x0590, 0x05FF, "Hebrew"),
        (0x0600, 0x06FF, "Arabic"),
        (0x0900, 0x097F, "Devanagari"),
        (0x3040, 0x309F, "Hiragana"),
        (0x30A0, 0x30FF, "Katakana"),
        (0x3130, 0x318F, "Hangul Compatibility Jamo"),
        (0xAC00, 0xD7AF, "Hangul Syllables"),
        (0x4E00, 0x9FFF, "CJK Unified Ideographs"),
        (0x1F300, 0x1FAFF, "Emoji and Symbols"),
    ]

    for start, end, name in block_ranges:
        if start <= code_point <= end:
            return name
    return "Other"


def identify_character_distribution(text: str) -> dict[str, int]:
    distribution = {
        "letters": 0,
        "digits": 0,
        "whitespace": 0,
        "punctuation": 0,
        "symbols": 0,
        "control": 0,
        "other": 0,
    }

    for character in text:
        category = unicodedata.category(character)
        if category.startswith("L"):
            distribution["letters"] += 1
        elif category.startswith("N"):
            distribution["digits"] += 1
        elif category == "Zs" or character.isspace():
            distribution["whitespace"] += 1
        elif category.startswith("P"):
            distribution["punctuation"] += 1
        elif category.startswith("S"):
            distribution["symbols"] += 1
        elif category.startswith("C"):
            distribution["control"] += 1
        else:
            distribution["other"] += 1

    return distribution


def identify_unicode_block_distribution(text: str) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for character in text:
        block = identify_writing_system(ord(character))
        distribution[block] = distribution.get(block, 0) + 1
    return distribution


@dataclass(frozen=True)
class RawInputProfile:
    raw_text: str
    character_count: int
    shannon_entropy: float
    character_set_distribution: dict[str, int] = field(default_factory=dict)
    unicode_block_distribution: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def profile_input(raw_input: str | bytes, encoding: str = "utf-8") -> RawInputProfile:
    text = normalize_input(raw_input)
    if isinstance(raw_input, bytes):
        raw_bytes = raw_input
    else:
        raw_bytes = str(raw_input).encode(encoding, errors="replace")

    return RawInputProfile(
        raw_text=text,
        character_count=len(text),
        shannon_entropy=calculate_shannon_entropy(raw_bytes),
        character_set_distribution=identify_character_distribution(text),
        unicode_block_distribution=identify_unicode_block_distribution(text),
        metadata={
            "encoding": encoding,
            "input_type": type(raw_input).__name__,
        },
    )
