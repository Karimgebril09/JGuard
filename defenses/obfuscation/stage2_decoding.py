from __future__ import annotations

from dataclasses import dataclass, field
import base64
import binascii
import html
import re
from typing import Any
from urllib.parse import unquote, unquote_plus
from defenses.obfuscation.helper import normalize_input


_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=_-]+$")
_HEX_RE = re.compile(r"^(?:0x)?[0-9A-Fa-f\s]+$")
_PERCENT_RE = re.compile(r"%(?:[0-9A-Fa-f]{2})")
_HTML_ENTITY_RE = re.compile(r"&(?:#x?[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);")

_COMMON_WORDS = {
    "the",
    "and",
    "to",
    "of",
    "a",
    "in",
    "is",
    "you",
    "that",
    "it",
    "for",
    "on",
    "with",
    "as",
    "this",
    "be",
    "are",
    "or",
    "was",
    "by",
    "not",
    "from",
    "at",
    "an",
    "have",
    "if",
    "can",
    "text",
    "payload",
    "message",
    "data",
    "code",
    "test",
}


def convert_to_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def is_text_printable(text: str) -> bool:
    if not text:
        return False
    printable = sum(1 for character in text if character.isprintable() or character.isspace())
    return printable / len(text) >= 0.8


def replacement_ratio(text: str) -> float:
    if not text:
        return 1.0
    return text.count("\ufffd") / len(text)


def is_text_likely_english(text: str) -> float:
    if not text:
        return float("-inf")

    lowered = text.lower()
    letters = sum(1 for character in lowered if character.isalpha())
    spaces = lowered.count(" ")
    vowels = sum(1 for character in lowered if character in "aeiou")
    tokens = re.findall(r"[a-z]+", lowered)
    common_hits = sum(1 for token in tokens if token in _COMMON_WORDS)
    printable = sum(1 for character in text if character.isprintable() or character.isspace())

    if letters == 0:
        return -10.0

    vowel_ratio = vowels / letters
    printable_ratio = printable / len(text)
    token_bonus = common_hits * 1.8
    space_bonus = min(spaces / max(len(text), 1) * 4.0, 1.5)

    # Keep the scoring simple and stable: the goal is to prefer human-readable
    # candidates over cipher text, not to model full language probability.
    return (
        printable_ratio * 3.0
        + vowel_ratio * 2.0
        + space_bonus
        + token_bonus
        + min(letters / max(len(text), 1) * 2.0, 2.0)
    )


def decode_html_entity(text: str) -> str | None:
    if not _HTML_ENTITY_RE.search(text):
        return None
    decoded = html.unescape(text)
    return decoded if decoded != text else None


def decode_url_encoding(text: str) -> str | None:
    if not _PERCENT_RE.search(text) and "+" not in text:
        return None

    decoded = unquote_plus(text)
    if decoded == text:
        decoded = unquote(text)
    return decoded if decoded != text else None


def decode_hex(text: str) -> str | None:
    candidate = text.strip()
    if candidate.startswith("0x") or candidate.startswith("0X"):
        candidate = candidate[2:]

    compact = re.sub(r"\s+", "", candidate)
    if not compact or len(compact) % 2 != 0 or not _HEX_RE.match(candidate):
        return None

    try:
        decoded = binascii.unhexlify(compact)
    except (binascii.Error, ValueError):
        return None

    text_result = convert_to_text(decoded)
    return text_result if text_result != text else None


def decode_base64(text: str) -> str | None:
    candidate = re.sub(r"\s+", "", text)
    if len(candidate) < 8 or not _BASE64_RE.match(candidate):
        return None

    padding = (-len(candidate)) % 4
    padded = candidate + ("=" * padding)

    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            decoded = decoder(padded)
        except (binascii.Error, ValueError):
            continue

        if not decoded:
            continue

        text_result = convert_to_text(decoded)
        if text_result == text:
            continue
        if not is_text_printable(text_result):
            continue

        # Guard against false positives where binary bytes become replacement
        # characters and still look "printable".
        if replacement_ratio(text_result) > 0.08:
            continue

        # Only accept Base64 when readability improves over the input.
        baseline_score = is_text_likely_english(text)
        decoded_score = is_text_likely_english(text_result)
        if decoded_score <= baseline_score + 0.35:
            continue

        return text_result

    return None


def decode_rot_ciphers(text: str) -> tuple[str, int, float] | None:
    if not any(character.isalpha() for character in text):
        return None

    baseline_score = is_text_likely_english(text)
    best_shift = 0
    best_score = baseline_score
    best_candidate = text

    for shift in range(1, 26):
        shifted: list[str] = []
        for character in text:
            if "a" <= character <= "z":
                shifted.append(chr(((ord(character) - ord("a") + shift) % 26) + ord("a")))
            elif "A" <= character <= "Z":
                shifted.append(chr(((ord(character) - ord("A") + shift) % 26) + ord("A")))
            else:
                shifted.append(character)

        candidate = "".join(shifted)
        candidate_score = is_text_likely_english(candidate)
        if candidate_score > best_score:
            best_score = candidate_score
            best_shift = shift
            best_candidate = candidate

    if best_candidate != text and best_score > baseline_score + 1.25:
        tokens = re.findall(r"[a-z]+", best_candidate.lower())
        common_hits = sum(1 for token in tokens if len(token) >= 3 and token in _COMMON_WORDS)
        if common_hits == 0:
            return None
        return best_candidate, best_shift, best_score
    return None


@dataclass(frozen=True)
class DecodingStep:
    transformation: str
    before: str
    after: str
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecodedInput:
    original_text: str
    decoded_text: str
    steps: list[DecodingStep] = field(default_factory=list)
    stable: bool = True
    iterations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class Stage2Decoding:
    def __init__(self, max_iterations: int = 5) -> None:
        self.max_iterations = max_iterations

    def decode(self, raw_input: str | bytes) -> DecodedInput:
        current_text = normalize_input(raw_input)
        original_text = current_text
        steps: list[DecodingStep] = []
        iteration_count = 0
        changed = False

        for iteration in range(1, self.max_iterations + 1):
            iteration_count = iteration
            changed = False

            for name, decoder in (
                ("html_entities", decode_html_entity),
                ("url_percent_encoding", decode_url_encoding),
                ("hex", decode_hex),
                ("base64", decode_base64),
            ):
                decoded = decoder(current_text)
                if decoded is None:
                    continue

                steps.append(
                    DecodingStep(
                        transformation=name,
                        before=current_text,
                        after=decoded,
                        confidence=0.95,
                        details={"iteration": iteration},
                    )
                )
                current_text = decoded
                changed = True
                break

            rot_candidate = decode_rot_ciphers(current_text)
            if rot_candidate is not None:
                decoded, shift, score = rot_candidate
                steps.append(
                    DecodingStep(
                        transformation="caesar_shift",
                        before=current_text,
                        after=decoded,
                        confidence=min(0.99, 0.5 + (score / 10.0)),
                        details={"iteration": iteration, "shift": shift, "score": score},
                    )
                )
                current_text = decoded
                changed = True

            if not changed:
                break

        return DecodedInput(
            original_text=original_text,
            decoded_text=current_text,
            steps=steps,
            stable=not changed or iteration_count < self.max_iterations,
            iterations=iteration_count,
            metadata={
                "input_type": type(raw_input).__name__,
                "encodings_attempted": [
                    "html_entities",
                    "url_percent_encoding",
                    "hex",
                    "base64",
                    "caesar_shift",
                ],
            },
        )


def decode_stage2(raw_input: str | bytes, max_iterations: int = 5) -> DecodedInput:
    """Wrapper for stage-2 recursive decoding."""

    return Stage2Decoding(max_iterations=max_iterations).decode(raw_input)
