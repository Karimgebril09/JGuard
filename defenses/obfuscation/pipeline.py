from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from defenses.obfuscation.stage1_profiling import profile_input
from defenses.obfuscation.stage2_decoding import decode_stage2
from defenses.obfuscation.stage3_unicoding import normalize_stage3
from defenses.obfuscation.stage4_leet import resolve_stage4
from defenses.obfuscation.stage5_defragmenting import defragment_stage5
from defenses.obfuscation.stage6_canonicalizing import canonicalize_stage6
from defenses.obfuscation.stage7_metadata import package_stage7, Stage7ProvenanceEnvelope


@dataclass(frozen=True)
class PipelineResult:
    """Complete pipeline execution with all intermediate stage outputs."""

    original_input: str | bytes
    provenance_envelope: Stage7ProvenanceEnvelope
    stage_outputs: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0


def run_obfuscation_pipeline(raw_input: str | bytes) -> PipelineResult:
    """Run the full obfuscation pipeline as a function-only flow."""
    import time

    start_time = time.time()
    stage_outputs: dict[str, Any] = {}

    # Stage 1: Profile raw input.
    s1_profile = profile_input(raw_input)
    stage_outputs["stage1"] = s1_profile

    # Stage 2: Decode ciphers.
    s2_decoded = decode_stage2(raw_input)
    stage_outputs["stage2"] = s2_decoded

    # Stage 3: Unicode normalization and homoglyph collapse.
    s3_normalized = normalize_stage3(s2_decoded.decoded_text)
    stage_outputs["stage3"] = s3_normalized

    # Stage 4: Leetspeak and substitution resolution.
    s4_resolved = resolve_stage4(s3_normalized.normalized_text)
    stage_outputs["stage4"] = s4_resolved

    # Stage 5: Structural defragmentation.
    s5_defragmented = defragment_stage5(s4_resolved.resolved_text)
    stage_outputs["stage5"] = s5_defragmented

    # Stage 6: Canonicalization.
    s6_canonical = canonicalize_stage6(s5_defragmented.defragmented_text)
    stage_outputs["stage6"] = s6_canonical

    # Stage 7: metadata packaging.
    s7_envelope = package_stage7(
        canonical_input=s6_canonical,
        stage1_profile=s1_profile,
        stage2_decoded=s2_decoded,
        stage3_normalized=s3_normalized,
        stage4_resolved=s4_resolved,
        stage5_defragmented=s5_defragmented,
    )
    stage_outputs["stage7"] = s7_envelope

    elapsed_ms = (time.time() - start_time) * 1000

    return PipelineResult(
        original_input=raw_input,
        provenance_envelope=s7_envelope,
        stage_outputs=stage_outputs,
        execution_time_ms=round(elapsed_ms, 2),
    )
