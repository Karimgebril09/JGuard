from __future__ import annotations

from typing import Any
from collections.abc import Callable

from defenses.obfuscation.stage1_profiling import profile_input
from defenses.obfuscation.stage2_decoding import decode_stage2
from defenses.obfuscation.stage3_unicoding import normalize_stage3
from defenses.obfuscation.stage4_leet import resolve_stage4
from defenses.obfuscation.stage5_defragmenting import defragment_stage5
from defenses.obfuscation.stage6_canonicalizing import canonicalize_stage6
from defenses.obfuscation.stage7_metadata import package_stage7
from defenses.obfuscation.stage8_harm_classifier import classify_stage8

import time


Stage8Classifier = Callable[[str], dict[str, Any]]


def _resolve_harm_label(stage8_result: dict[str, Any]) -> str | None:
    mapped = stage8_result.get("mapped_categories")
    if isinstance(mapped, list) and mapped:
        first = mapped[0]
        return str(first) if first is not None else None

    raw = stage8_result.get("raw_categories")
    if isinstance(raw, list) and raw:
        first = raw[0]
        return str(first) if first is not None else None

    return None


def run_obfuscation_pipeline(
    raw_input: str | bytes,
    *,
    stage8_classifier: Stage8Classifier | None = None,
    stage8_model_id: str | None = None,
) -> dict[str, Any]:
    start_time = time.time()
    stage_outputs: dict[str, Any] = {}

    # Stage 1: Profile raw input
    s1_profile = profile_input(raw_input)
    stage_outputs["stage1"] = s1_profile

    # Stage 2: Decode ciphers
    s2_decoded = decode_stage2(raw_input)
    stage_outputs["stage2"] = s2_decoded

    # Stage 3: Unicode normalization and homoglyph collapse
    s3_normalized = normalize_stage3(str(s2_decoded["decoded_text"]))
    stage_outputs["stage3"] = s3_normalized

    # Stage 4: Leetspeak and substitution resolution
    s4_resolved = resolve_stage4(str(s3_normalized["normalized_text"]))
    stage_outputs["stage4"] = s4_resolved

    # Stage 5: Structural defragmentation
    s5_defragmented = defragment_stage5(str(s4_resolved["resolved_text"]))
    stage_outputs["stage5"] = s5_defragmented

    # Stage 6: Canonicalization
    s6_canonical = canonicalize_stage6(str(s5_defragmented["defragmented_text"]))
    stage_outputs["stage6"] = s6_canonical

    # Stage 7: metadata 
    s7_metadata_envelope = package_stage7(
        canonical_input=s6_canonical,
        stage1_profile=s1_profile,
        stage2_decoded=s2_decoded,
        stage3_normalized=s3_normalized,
        stage4_resolved=s4_resolved,
        stage5_defragmented=s5_defragmented,
    )
    stage_outputs["stage7"] = s7_metadata_envelope

    s8_final_envelope = classify_stage8(
        canonical_text=str(s7_metadata_envelope["canonical_text"]),
        metadata_envelope=s7_metadata_envelope,
        classifier=stage8_classifier,
        model_id=stage8_model_id,
    )
    stage8_result = s8_final_envelope["stage8"]
    stage_outputs["stage8"] = stage8_result

    clean_text = str(s6_canonical["canonical_text"])
    is_safe = bool(stage8_result.get("is_safe", False))
    decision = "safe" if is_safe else "unsafe"
    harm_label = _resolve_harm_label(stage8_result)

    elapsed_ms = (time.time() - start_time) * 1000

    return {
        "original_input": raw_input,
        "clean_text": clean_text,
        "decision": decision,
        "is_safe": is_safe,
        "harm_label": harm_label,
        "action": stage8_result.get("action"),
        "metadata_envelope": s8_final_envelope,
        "stage_outputs": stage_outputs,
        "execution_time_ms": round(elapsed_ms, 2),
    }


def run_obfuscation_guard(
    raw_input: str | bytes,
    *,
    stage8_classifier: Stage8Classifier | None = None,
    stage8_model_id: str | None = None,
) -> dict[str, Any]:
    """Run the 8-stage pipeline and return a minimal integration-friendly result."""
    pipeline_result = run_obfuscation_pipeline(
        raw_input,
        stage8_classifier=stage8_classifier,
        stage8_model_id=stage8_model_id,
    )
    return {
        "clean_text": pipeline_result["clean_text"],
        "decision": pipeline_result["decision"],
        "is_safe": pipeline_result["is_safe"],
        "harm_label": pipeline_result["harm_label"],
        "action": pipeline_result["action"],
        "execution_time_ms": pipeline_result["execution_time_ms"],
    }
