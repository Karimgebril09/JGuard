from __future__ import annotations

from typing import Any
import difflib


def make_unified_diff(original_text: str, normalized_text: str) -> str:
    if original_text == normalized_text:
        return ""
    diff_lines = difflib.unified_diff(
        original_text.splitlines(),
        normalized_text.splitlines(),
        fromfile="original",
        tofile="normalized",
        lineterm="",
    )
    return "\n".join(diff_lines)


def package_stage7(
    *,
    canonical_input: Any,
    stage1_profile: Any | None = None,
    stage2_decoded: Any | None = None,
    stage3_normalized: Any | None = None,
    stage4_resolved: Any | None = None,
    stage5_defragmented: Any | None = None,
) -> dict[str, object]:
    s2 = stage2_decoded if isinstance(stage2_decoded, dict) else {}
    s3 = stage3_normalized if isinstance(stage3_normalized, dict) else {}
    s4 = stage4_resolved if isinstance(stage4_resolved, dict) else {}
    s5 = stage5_defragmented if isinstance(stage5_defragmented, dict) else {}
    s6 = canonical_input if isinstance(canonical_input, dict) else {}

    original_text = str(s6.get("original_text", ""))
    canonical_text = str(s6.get("canonical_text", original_text))

    transformations: list[str] = []
    for step in s2.get("steps", []):
        transformation = step.get("transformation")
        if transformation:
            transformations.append(str(transformation))

    if s3.get("zero_width_stripped", 0):
        transformations.append("strip_zero_width")
    if s3.get("homoglyph_replacements", 0):
        transformations.append("homoglyph_collapse")
    if s4.get("leet_replacements", 0):
        transformations.append("leet_resolution")
    if s4.get("caesar_shift") is not None:
        transformations.append("caesar_shift")

    for decision in s5.get("metadata", {}).get("decisions", []):
        name = decision.get("transformation")
        if name:
            transformations.append(str(name))

    if s6.get("lowered"):
        transformations.append("lowercase")
    if s6.get("whitespace_collapsed"):
        transformations.append("whitespace_collapse")
    if s6.get("punctuation_normalized"):
        transformations.append("punctuation_normalization")

    anomalies: list[str] = []
    entropy = float((stage1_profile or {}).get("shannon_entropy", 0.0))
    if entropy >= 4.6:
        anomalies.append("high_entropy_payload")

    unicode_dist = dict((stage1_profile or {}).get("unicode_block_distribution", {}))
    total_chars = max(1, int((stage1_profile or {}).get("character_count", 0)))
    non_basic = sum(
        count for block, count in unicode_dist.items() if block not in {"Basic Latin", "Other"}
    )
    if (non_basic / total_chars) >= 0.25:
        anomalies.append("high_non_basic_unicode_ratio")

    if len(s2.get("steps", [])) >= 3:
        anomalies.append("multi_layer_encoding")
    if int(s3.get("zero_width_stripped", 0)) > 0:
        anomalies.append("zero_width_obfuscation")
    if int(s3.get("homoglyph_replacements", 0)) > 0:
        anomalies.append("homoglyph_obfuscation")
    if int(s4.get("leet_replacements", 0)) >= 2:
        anomalies.append("leet_substitution_obfuscation")
    if bool(s5.get("reversed_applied", False)):
        anomalies.append("reverse_text_obfuscation")
    if len(transformations) >= 6:
        anomalies.append("heavy_transformation_chain")
    if not canonical_text:
        anomalies.append("empty_after_normalization")

    return {
        "original_text": original_text,
        "canonical_text": canonical_text,
        "obfuscation_depth": len(s2.get("steps", [])),
        "transformations": transformations,
        "anomalies": anomalies,
        "metadata": {
            "transformations_applied": len(transformations),
            "anomalies_detected": len(anomalies),
            "decode_layers": len(s2.get("steps", [])),
            "diff": make_unified_diff(original_text, canonical_text),
        },
    }
