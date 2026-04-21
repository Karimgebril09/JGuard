from __future__ import annotations

from dataclasses import dataclass, field
from math import log
import re
from typing import Any
from defenses.obfuscation.helper import normalize_input

LEET_CHAR_MAP = {
    "0": ["o"],
    "1": ["i", "l"],
    "2": ["z"],
    "3": ["e"],
    "4": ["a"],
    "5": ["s"],
    "6": ["g"],
    "7": ["t"],
    "8": ["b"],
    "9": ["g"],
    "@": ["a"],
    "$": ["s"],
    "!": ["i"],
    "|": ["l", "i"],
}

TOKEN_RE = re.compile(r"[A-Za-z0-9@$!|]+|[^A-Za-z0-9@$!|]+")

WORD_FREQUENCY = {
    "a": 500,
    "an": 450,
    "and": 1000,
    "attack": 220,
    "bypass": 160,
    "can": 550,
    "code": 580,
    "data": 620,
    "decode": 180,
    "email": 160,
    "exploit": 200,
    "for": 900,
    "hack": 260,
    "hello": 700,
    "in": 1000,
    "is": 980,
    "malware": 120,
    "message": 530,
    "model": 220,
    "not": 700,
    "of": 1200,
    "on": 740,
    "or": 680,
    "payload": 210,
    "phishing": 140,
    "prompt": 240,
    "security": 280,
    "system": 360,
    "test": 430,
    "text": 520,
    "that": 900,
    "the": 1500,
    "this": 820,
    "to": 1300,
    "user": 370,
    "with": 780,
    "world": 640,
    "you": 860,
    "mainframe": 160,
}

def is_text_likely_english(text: str) -> float:
    if not text:
        return float("-inf")

    lowered = text.lower()
    tokens = re.findall(r"[a-z]+", lowered)
    if not tokens:
        return -10.0

    letters = sum(1 for character in lowered if character.isalpha())
    vowels = sum(1 for character in lowered if character in "aeiou")
    printable = sum(1 for character in text if character.isprintable() or character.isspace())

    token_score = 0.0
    for token in tokens:
        freq = WORD_FREQUENCY.get(token, 0)
        if freq > 0:
            token_score += 1.8 + log(freq + 1.0, 10)
        else:
            token_score += fallback_token_score(token)

    printable_ratio = printable / len(text)
    vowel_ratio = vowels / max(letters, 1)

    return (
        token_score
        + printable_ratio * 2.5
        + min(vowel_ratio * 2.0, 1.2)
        + min(letters / max(len(text), 1) * 1.5, 1.5)
    )


def find_dictionary_hits(text: str) -> int:
    tokens = re.findall(r"[a-z]+", text.lower())
    return sum(1 for token in tokens if token in WORD_FREQUENCY)


def fallback_token_score(token: str) -> float:
    if not token:
        return -2.0

    if not token.isalpha():
        return -1.2

    vowels = sum(1 for character in token if character in "aeiou")
    if vowels == 0 and len(token) > 2:
        return -0.8

    repeated = sum(1 for i in range(1, len(token)) if token[i] == token[i - 1])
    return max(-0.8, 0.2 + (vowels / max(len(token), 1)) - (0.15 * repeated))


def caesar_shift(text: str, shift: int) -> str:
    shifted: list[str] = []
    for character in text:
        if "a" <= character <= "z":
            shifted.append(chr(((ord(character) - ord("a") + shift) % 26) + ord("a")))
        elif "A" <= character <= "Z":
            shifted.append(chr(((ord(character) - ord("A") + shift) % 26) + ord("A")))
        else:
            shifted.append(character)
    return "".join(shifted)


def find_leet_candidates(token: str, beam_width: int = 24) -> list[str]:
    candidates = [""]
    lowered = token.lower()

    for character in lowered:
        substitutions = [character]
        if character in LEET_CHAR_MAP:
            substitutions.extend(LEET_CHAR_MAP[character])

        next_candidates: list[str] = []
        for prefix in candidates:
            for substitution in substitutions:
                next_candidates.append(prefix + substitution)

        # Keep search bounded while retaining the most plausible strings.
        next_candidates = sorted(
            set(next_candidates),
            key=fallback_token_score,
            reverse=True,
        )
        candidates = next_candidates[:beam_width]

    return candidates


def resolve_leetspeak_token(
    token: str,
    dictionary_threshold: float,
    min_improvement: float,
) -> tuple[str, float, bool]:
    if not any(character in LEET_CHAR_MAP for character in token.lower()):
        return token, 0.0, False

    baseline = token.lower()
    baseline_score = fallback_token_score(baseline) + (WORD_FREQUENCY.get(baseline, 0) > 0)

    best_candidate = baseline
    best_score = baseline_score

    for candidate in find_leet_candidates(token):
        score = fallback_token_score(candidate)
        if candidate in WORD_FREQUENCY:
            score += 1.8 + log(WORD_FREQUENCY[candidate] + 1.0, 10)
        if score > best_score:
            best_score = score
            best_candidate = candidate

    improvement = best_score - baseline_score
    is_dictionary_hit = best_candidate in WORD_FREQUENCY

    if best_candidate != baseline and is_dictionary_hit and improvement >= min_improvement:
        confidence = min(0.99, dictionary_threshold + (improvement / 6.0))
        return best_candidate, confidence, True

    return token, 0.0, False


def resolve_leetspeak_text(
    text: str,
    dictionary_threshold: float,
    min_improvement: float,
) -> tuple[str, int, list[dict[str, Any]]]:
    parts = TOKEN_RE.findall(text)
    resolved_parts: list[str] = []
    replacements = 0
    decisions: list[dict[str, Any]] = []

    for part in parts:
        if not any(character.isalnum() for character in part):
            resolved_parts.append(part)
            continue

        resolved, confidence, applied = resolve_leetspeak_token(
            token=part,
            dictionary_threshold=dictionary_threshold,
            min_improvement=min_improvement,
        )
        resolved_parts.append(resolved)

        if applied:
            replacements += 1
            decisions.append(
                {
                    "original": part,
                    "resolved": resolved,
                    "confidence": round(confidence, 4),
                }
            )

    return "".join(resolved_parts), replacements, decisions


def find_best_caesar_candidate(text: str, min_improvement: float) -> tuple[str, int, float] | None:
    baseline_score = is_text_likely_english(text)
    baseline_hits = find_dictionary_hits(text)
    best_text = text
    best_shift = 0
    best_score = baseline_score
    best_hits = baseline_hits

    for shift in range(1, 26):
        candidate = caesar_shift(text, shift)
        score = is_text_likely_english(candidate)
        hits = find_dictionary_hits(candidate)
        if hits > best_hits or (hits == best_hits and score > best_score):
            best_text = candidate
            best_shift = shift
            best_score = score
            best_hits = hits

    if best_shift == 0:
        return None

    improvement = best_score - baseline_score
    if improvement < min_improvement:
        return None

    if best_hits <= baseline_hits:
        return None

    return best_text, best_shift, improvement


@dataclass(frozen=True)
class Stage4ResolvedInput:
    original_text: str
    resolved_text: str
    leet_replacements: int
    caesar_shift: int | None
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_stage4(
    raw_input: str | bytes,
    dictionary_threshold: float = 0.6,
    leet_min_improvement: float = 1.15,
    caesar_min_improvement: float = 1.25,
) -> Stage4ResolvedInput:
    original_text = normalize_input(raw_input)

    leet_text, leet_count, leet_decisions = resolve_leetspeak_text(
        text=original_text,
        dictionary_threshold=dictionary_threshold,
        min_improvement=leet_min_improvement,
    )

    caesar_candidate = find_best_caesar_candidate(
        leet_text,
        min_improvement=caesar_min_improvement,
    )

    final_text = leet_text
    caesar_shift: int | None = None
    caesar_improvement = 0.0

    if caesar_candidate is not None:
        final_text, caesar_shift, caesar_improvement = caesar_candidate

    confidence_components: list[float] = []
    if leet_count > 0:
        avg_leet_confidence = sum(item["confidence"] for item in leet_decisions) / leet_count
        confidence_components.append(avg_leet_confidence)
    if caesar_shift is not None:
        confidence_components.append(min(0.99, 0.5 + (caesar_improvement / 8.0)))

    confidence = (
        sum(confidence_components) / len(confidence_components)
        if confidence_components
        else 0.0
    )

    return Stage4ResolvedInput(
        original_text=original_text,
        resolved_text=final_text,
        leet_replacements=leet_count,
        caesar_shift=caesar_shift,
        confidence=round(confidence, 4),
        metadata={
            "input_type": type(raw_input).__name__,
            "dictionary_threshold": dictionary_threshold,
            "leet_min_improvement": leet_min_improvement,
            "caesar_min_improvement": caesar_min_improvement,
            "leet_decisions": leet_decisions,
        },
    )
