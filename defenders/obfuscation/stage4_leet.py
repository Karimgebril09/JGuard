from __future__ import annotations

import importlib
import re
from typing import Any
from defenses.obfuscation.helper import normalize_input

zipf_frequency = importlib.import_module("wordfreq").zipf_frequency

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

TOKEN_REGEX = re.compile(r"[A-Za-z0-9@$!|]+|[^A-Za-z0-9@$!|]+")

MIN_WORD_ZIPF = 3.0

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
        freq = zipf_frequency(token, "en")
        if freq >= MIN_WORD_ZIPF:
            token_score += 1.2 + freq
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
    return sum(1 for token in tokens if zipf_frequency(token, "en") >= MIN_WORD_ZIPF)


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
            key=lambda candidate: fallback_token_score(candidate)
            + max(zipf_frequency(candidate, "en"), 0.0),
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
    baseline_zipf = zipf_frequency(baseline, "en")
    baseline_score = fallback_token_score(baseline) + (baseline_zipf if baseline_zipf >= MIN_WORD_ZIPF else 0.0)

    best_candidate = baseline
    best_score = baseline_score

    for candidate in find_leet_candidates(token):
        candidate_zipf = zipf_frequency(candidate, "en")
        score = fallback_token_score(candidate)
        if candidate_zipf >= MIN_WORD_ZIPF:
            score += candidate_zipf
        if score > best_score:
            best_score = score
            best_candidate = candidate

    improvement = best_score - baseline_score
    is_dictionary_hit = zipf_frequency(best_candidate, "en") >= MIN_WORD_ZIPF

    if best_candidate != baseline and is_dictionary_hit and improvement >= min_improvement:
        confidence = min(0.99, dictionary_threshold + (improvement / 6.0))
        return best_candidate, confidence, True

    return token, 0.0, False


def resolve_leetspeak_text(
    text: str,
    dictionary_threshold: float,
    min_improvement: float,
) -> tuple[str, int, list[dict[str, Any]]]:
    parts = TOKEN_REGEX.findall(text)
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


def resolve_caesar_token(token: str, min_improvement: float) -> tuple[str, float, bool]:
    if not any(character.isalpha() for character in token):
        return token, 0.0, False

    baseline_score = is_text_likely_english(token)
    baseline_hits = find_dictionary_hits(token)
    best_text = token
    best_shift = 0
    best_score = baseline_score
    best_hits = baseline_hits

    for shift in range(1, 26):
        candidate = caesar_shift(token, shift)
        score = is_text_likely_english(candidate)
        hits = find_dictionary_hits(candidate)
        if hits > best_hits or (hits == best_hits and score > best_score):
            best_text = candidate
            best_shift = shift
            best_score = score
            best_hits = hits

    if best_shift == 0:
        return token, 0.0, False

    improvement = best_score - baseline_score
    if improvement < min_improvement:
        return token, 0.0, False

    if best_hits <= baseline_hits:
        return token, 0.0, False

    confidence = min(0.99, 0.55 + (improvement / 8.0))
    return best_text, confidence, True


def resolve_caesar_text(
    text: str,
    min_improvement: float,
) -> tuple[str, int, list[dict[str, Any]]]:
    parts = TOKEN_REGEX.findall(text)
    resolved_parts: list[str] = []
    decisions: list[dict[str, Any]] = []
    shifts_applied = 0

    for part in parts:
        if not any(character.isalpha() for character in part):
            resolved_parts.append(part)
            continue

        resolved, confidence, applied = resolve_caesar_token(part, min_improvement=min_improvement)
        resolved_parts.append(resolved)

        if applied:
            shifts_applied += 1
            decisions.append(
                {
                    "original": part,
                    "resolved": resolved,
                    "confidence": round(confidence, 4),
                }
            )

    return "".join(resolved_parts), shifts_applied, decisions


def resolve_stage4(
    raw_input: str | bytes,
    dictionary_threshold: float = 0.6,
    leet_min_improvement: float = 1.15,
    caesar_min_improvement: float = 1.25,
) -> dict[str, object]:
    original_text = normalize_input(raw_input)

    leet_text, leet_count, leet_decisions = resolve_leetspeak_text(
        text=original_text,
        dictionary_threshold=dictionary_threshold,
        min_improvement=leet_min_improvement,
    )

    caesar_text, caesar_count, caesar_decisions = resolve_caesar_text(
        leet_text,
        min_improvement=caesar_min_improvement,
    )

    final_text = caesar_text

    confidence_components: list[float] = []
    if leet_count > 0:
        avg_leet_confidence = sum(item["confidence"] for item in leet_decisions) / leet_count
        confidence_components.append(avg_leet_confidence)
    if caesar_count > 0:
        avg_caesar_confidence = sum(item["confidence"] for item in caesar_decisions) / caesar_count
        confidence_components.append(avg_caesar_confidence)

    confidence = (
        sum(confidence_components) / len(confidence_components)
        if confidence_components
        else 0.0
    )

    return {
        "original_text": original_text,
        "resolved_text": final_text,
        "leet_replacements": leet_count,
        "caesar_shift": caesar_count,
        "confidence": round(confidence, 4),
        "metadata": {
            "input_type": type(raw_input).__name__,
            "dictionary_threshold": dictionary_threshold,
            "leet_min_improvement": leet_min_improvement,
            "caesar_min_improvement": caesar_min_improvement,
            "leet_decisions": leet_decisions,
            "caesar_decisions": caesar_decisions,
        },
    }
