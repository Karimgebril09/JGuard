from __future__ import annotations

from math import log
import re
from typing import Any
from defenses.obfuscation.helper import normalize_input


LITERAL_CONCAT_RE = re.compile(
    r"(?:\"[^\"\\]*(?:\\.[^\"\\]*)*\"|'[^'\\]*(?:\\.[^'\\]*)*')"
    r"(?:\s*\+\s*(?:\"[^\"\\]*(?:\\.[^\"\\]*)*\"|'[^'\\]*(?:\\.[^'\\]*)*'))+"
)
QUOTED_LITERAL_RE = re.compile(r"^(?P<q>['\"])(?P<body>.*)(?P=q)$")
ARRAY_ASSIGN_RE = re.compile(
    r"\b(?P<name>[A-Za-z_]\w*)\s*=\s*\[(?P<body>[^\]]+)\]",
    re.DOTALL,
)
ARRAY_ELEMENT_RE = re.compile(r"(['\"])(.*?)(?<!\\)\1")
ARRAY_INDEX_EXPR_RE = re.compile(r"\b([A-Za-z_]\w*\[\d+\]\s*(?:\+\s*[A-Za-z_]\w*\[\d+\]\s*)+)")
INDEX_ACCESS_RE = re.compile(r"\b(?P<name>[A-Za-z_]\w*)\[(?P<index>\d+)\]")
SLICE_REVERSE_RE = re.compile(r"(?P<q>['\"])(?P<body>.*?)(?P=q)\s*\[\s*::\s*-1\s*\]")

WORD_FREQUENCY = {
    "the": 1500,
    "to": 1300,
    "of": 1200,
    "and": 1000,
    "in": 1000,
    "is": 980,
    "this": 820,
    "with": 780,
    "not": 700,
    "you": 860,
    "hello": 700,
    "world": 640,
    "payload": 210,
    "security": 280,
    "attack": 220,
    "exploit": 200,
    "mainframe": 160,
    "prompt": 240,
    "message": 530,
    "text": 520,
    "code": 580,
    "data": 620,
    "hack": 260,
}

def fallback_token_score(token: str) -> float:
    if not token:
        return -2.0
    vowels = sum(1 for ch in token if ch in "aeiou")
    repeated = sum(1 for i in range(1, len(token)) if token[i] == token[i - 1])
    if vowels == 0 and len(token) > 2:
        return -0.8
    return max(-0.8, 0.15 + (vowels / max(len(token), 1)) - (0.1 * repeated))


def is_text_likely_english(text: str) -> float:
    if not text:
        return float("-inf")

    lowered = text.lower()
    tokens = re.findall(r"[a-z]+", lowered)
    if not tokens:
        return -10.0

    letters = sum(1 for ch in lowered if ch.isalpha())
    vowels = sum(1 for ch in lowered if ch in "aeiou")
    printable = sum(1 for ch in text if ch.isprintable() or ch.isspace())

    token_score = 0.0
    for token in tokens:
        freq = WORD_FREQUENCY.get(token)
        if freq is not None:
            token_score += 1.8 + log(freq + 1.0, 10)
        else:
            token_score += fallback_token_score(token)

    return (
        token_score
        + (printable / max(len(text), 1)) * 2.0
        + min((vowels / max(letters, 1)) * 1.8, 1.1)
    )


def find_dictionary_hits(text: str) -> int:
    return sum(1 for token in re.findall(r"[a-z]+", text.lower()) if token in WORD_FREQUENCY)


def unquote_literal(token: str) -> str:
    match = QUOTED_LITERAL_RE.match(token.strip())
    if not match:
        return token
    return bytes(match.group("body"), "utf-8").decode("unicode_escape")


def resolve_literal_concatenations(text: str) -> tuple[str, list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []

    def replacement(match: re.Match[str]) -> str:
        expression = match.group(0)
        pieces = [
            unquote_literal(token)
            for token in re.findall(r"\"[^\"\\]*(?:\\.[^\"\\]*)*\"|'[^'\\]*(?:\\.[^'\\]*)*'", expression)
        ]
        resolved = "".join(pieces)
        decisions.append(
            {
                "transformation": "literal_concat",
                "before": expression,
                "after": resolved,
                "confidence": 0.98,
            }
        )
        return resolved

    return LITERAL_CONCAT_RE.sub(replacement, text), decisions


def parse_array_assignments(text: str) -> dict[str, list[str]]:
    arrays: dict[str, list[str]] = {}
    for match in ARRAY_ASSIGN_RE.finditer(text):
        name = match.group("name")
        body = match.group("body")
        elements = [unquote_literal(item.group(0)) for item in ARRAY_ELEMENT_RE.finditer(body)]
        if elements:
            arrays[name] = elements
    return arrays


def resolve_array_index_expressions(
    text: str,
    arrays: dict[str, list[str]],
) -> tuple[str, list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []

    def replacement(match: re.Match[str]) -> str:
        expression = match.group(1)
        accesses = INDEX_ACCESS_RE.findall(expression)
        if not accesses:
            return expression

        names = {name for name, _ in accesses}
        if len(names) != 1:
            return expression

        name = next(iter(names))
        array_values = arrays.get(name)
        if array_values is None:
            return expression

        resolved_chars: list[str] = []
        for _, index_text in accesses:
            index = int(index_text)
            if index < 0 or index >= len(array_values):
                return expression
            resolved_chars.append(array_values[index])

        resolved = "".join(resolved_chars)
        decisions.append(
            {
                "transformation": "array_index_concat",
                "before": expression,
                "after": resolved,
                "confidence": 0.95,
                "details": {"array_name": name, "indices": [int(i) for _, i in accesses]},
            }
        )
        return resolved

    return ARRAY_INDEX_EXPR_RE.sub(replacement, text), decisions


def resolve_slice_reverse(text: str) -> tuple[str, list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []

    def replacement(match: re.Match[str]) -> str:
        literal = match.group("body")
        resolved = literal[::-1]
        decisions.append(
            {
                "transformation": "slice_reverse",
                "before": match.group(0),
                "after": resolved,
                "confidence": 0.96,
            }
        )
        return resolved

    return SLICE_REVERSE_RE.sub(replacement, text), decisions


def maybe_reverse_whole_text(text: str, min_improvement: float) -> tuple[str, dict[str, Any] | None]:
    reversed_text = text[::-1]
    forward_score = is_text_likely_english(text)
    reverse_score = is_text_likely_english(reversed_text)
    forward_hits = find_dictionary_hits(text)
    reverse_hits = find_dictionary_hits(reversed_text)

    if reverse_score - forward_score < min_improvement:
        return text, None
    if reverse_hits <= forward_hits:
        return text, None

    return (
        reversed_text,
        {
            "transformation": "whole_text_reverse",
            "before": text,
            "after": reversed_text,
            "confidence": round(min(0.99, 0.55 + ((reverse_score - forward_score) / 8.0)), 4),
            "details": {
                "forward_score": round(forward_score, 4),
                "reverse_score": round(reverse_score, 4),
                "forward_dictionary_hits": forward_hits,
                "reverse_dictionary_hits": reverse_hits,
            },
        },
    )


def defragment_stage5(
    raw_input: str | bytes,
    reverse_min_improvement: float = 1.35,
) -> dict[str, object]:
    original_text = normalize_input(raw_input)
    decisions: list[dict[str, Any]] = []

    text_after_literal, literal_decisions = resolve_literal_concatenations(original_text)
    decisions.extend(literal_decisions)

    arrays = parse_array_assignments(text_after_literal)
    text_after_indices, index_decisions = resolve_array_index_expressions(text_after_literal, arrays)
    decisions.extend(index_decisions)

    text_after_slice, slice_decisions = resolve_slice_reverse(text_after_indices)
    decisions.extend(slice_decisions)

    final_text, reverse_decision = maybe_reverse_whole_text(
        text_after_slice,
        min_improvement=reverse_min_improvement,
    )
    if reverse_decision is not None:
        decisions.append(reverse_decision)

    confidences = [float(item["confidence"]) for item in decisions if "confidence" in item]
    confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    return {
        "original_text": original_text,
        "defragmented_text": final_text,
        "fragments_resolved": len(decisions),
        "reversed_applied": reverse_decision is not None,
        "confidence": confidence,
        "metadata": {
            "input_type": type(raw_input).__name__,
            "reverse_min_improvement": reverse_min_improvement,
            "decisions": decisions,
            "arrays_detected": sorted(arrays.keys()),
        },
    }
