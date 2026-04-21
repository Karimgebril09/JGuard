from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import difflib
from typing import Any


def map_as(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    return {}


def _make_unified_diff(original_text: str, normalized_text: str) -> str:
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


@dataclass(frozen=True)
class TransformationRecord:
    """A single transformation event across stages 2-6."""

    stage: str
    transformation: str
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnomalyRecord:
    """An anomaly detected from stage outputs and transformation trace."""

    anomaly: str
    severity: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Stage7ProvenanceEnvelope:
    """Final packaging output for downstream risk-aware policy evaluation."""

    original_text: str
    canonical_text: str
    transformations: list[TransformationRecord] = field(default_factory=list)
    anomalies: list[AnomalyRecord] = field(default_factory=list)
    risk_score: float = 0.0
    normalization_diff: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _collect_transformations(
    s2: dict[str, Any],
    s3: dict[str, Any],
    s4: dict[str, Any],
    s5: dict[str, Any],
    s6: dict[str, Any],
) -> list[TransformationRecord]:
    records: list[TransformationRecord] = []

    for step in s2.get("steps", []):
        records.append(
            TransformationRecord(
                stage="stage2",
                transformation=str(step.get("transformation", "decode")),
                confidence=float(step.get("confidence", 0.8)),
                details=dict(step.get("details", {})),
            )
        )

    if s3.get("zero_width_stripped", 0) > 0:
        records.append(
            TransformationRecord(
                stage="stage3",
                transformation="strip_zero_width",
                confidence=0.98,
                details={"count": int(s3.get("zero_width_stripped", 0))},
            )
        )
    if s3.get("homoglyph_replacements", 0) > 0:
        records.append(
            TransformationRecord(
                stage="stage3",
                transformation="homoglyph_collapse",
                confidence=0.94,
                details={"count": int(s3.get("homoglyph_replacements", 0))},
            )
        )

    if s4.get("leet_replacements", 0) > 0:
        records.append(
            TransformationRecord(
                stage="stage4",
                transformation="leet_resolution",
                confidence=float(s4.get("confidence", 0.75)),
                details={"count": int(s4.get("leet_replacements", 0))},
            )
        )
    if s4.get("caesar_shift") is not None:
        records.append(
            TransformationRecord(
                stage="stage4",
                transformation="caesar_shift",
                confidence=float(s4.get("confidence", 0.75)),
                details={"shift": int(s4.get("caesar_shift"))},
            )
        )

    for decision in s5.get("metadata", {}).get("decisions", []):
        records.append(
            TransformationRecord(
                stage="stage5",
                transformation=str(decision.get("transformation", "defragment")),
                confidence=float(decision.get("confidence", 0.85)),
                details={
                    "before": decision.get("before", ""),
                    "after": decision.get("after", ""),
                },
            )
        )

    if bool(s6.get("lowered")):
        records.append(
            TransformationRecord(
                stage="stage6",
                transformation="lowercase",
                confidence=1.0,
                details={},
            )
        )
    if bool(s6.get("whitespace_collapsed")):
        records.append(
            TransformationRecord(
                stage="stage6",
                transformation="whitespace_collapse",
                confidence=1.0,
                details={},
            )
        )
    if bool(s6.get("punctuation_normalized")):
        records.append(
            TransformationRecord(
                stage="stage6",
                transformation="punctuation_normalization",
                confidence=1.0,
                details={"changes": int(s6.get("metadata", {}).get("punctuation_changes", 0))},
            )
        )

    return records


def _detect_anomalies(
    *,
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    stage3: dict[str, Any],
    stage4: dict[str, Any],
    stage5: dict[str, Any],
    stage6: dict[str, Any],
    transformation_count: int,
) -> list[AnomalyRecord]:
    anomalies: list[AnomalyRecord] = []

    entropy = float(stage1.get("shannon_entropy", 0.0))
    if entropy >= 4.6:
        anomalies.append(
            AnomalyRecord(
                anomaly="high_entropy_payload",
                severity="high",
                score=min(1.0, entropy / 8.0),
                details={"entropy": round(entropy, 4)},
            )
        )

    unicode_dist = dict(stage1.get("unicode_block_distribution", {}))
    total_chars = max(1, int(stage1.get("character_count", 0)))
    non_basic = sum(
        count for block, count in unicode_dist.items() if block not in {"Basic Latin", "Other"}
    )
    non_basic_ratio = non_basic / total_chars
    if non_basic_ratio >= 0.25:
        anomalies.append(
            AnomalyRecord(
                anomaly="high_non_basic_unicode_ratio",
                severity="medium",
                score=min(1.0, non_basic_ratio),
                details={"ratio": round(non_basic_ratio, 4)},
            )
        )

    decode_layers = len(stage2.get("steps", []))
    if decode_layers >= 3:
        anomalies.append(
            AnomalyRecord(
                anomaly="multi_layer_encoding",
                severity="high",
                score=min(1.0, decode_layers / 6.0),
                details={"layers": decode_layers},
            )
        )

    if int(stage3.get("zero_width_stripped", 0)) > 0:
        anomalies.append(
            AnomalyRecord(
                anomaly="zero_width_obfuscation",
                severity="medium",
                score=0.65,
                details={"count": int(stage3.get("zero_width_stripped", 0))},
            )
        )

    if int(stage3.get("homoglyph_replacements", 0)) > 0:
        anomalies.append(
            AnomalyRecord(
                anomaly="homoglyph_obfuscation",
                severity="medium",
                score=0.62,
                details={"count": int(stage3.get("homoglyph_replacements", 0))},
            )
        )

    if int(stage4.get("leet_replacements", 0)) >= 2:
        anomalies.append(
            AnomalyRecord(
                anomaly="leet_substitution_obfuscation",
                severity="medium",
                score=0.58,
                details={"count": int(stage4.get("leet_replacements", 0))},
            )
        )

    if bool(stage5.get("reversed_applied", False)):
        anomalies.append(
            AnomalyRecord(
                anomaly="reverse_text_obfuscation",
                severity="medium",
                score=0.6,
                details={},
            )
        )

    if transformation_count >= 6:
        anomalies.append(
            AnomalyRecord(
                anomaly="heavy_transformation_chain",
                severity="high",
                score=min(1.0, transformation_count / 10.0),
                details={"count": transformation_count},
            )
        )

    if not stage6.get("canonical_text"):
        anomalies.append(
            AnomalyRecord(
                anomaly="empty_after_normalization",
                severity="low",
                score=0.35,
                details={},
            )
        )

    return anomalies


def _compute_risk_score(
    transformations: list[TransformationRecord],
    anomalies: list[AnomalyRecord],
) -> float:
    if not transformations and not anomalies:
        return 0.05

    transformation_factor = min(1.0, len(transformations) / 10.0)
    confidence_factor = (
        sum(record.confidence for record in transformations) / len(transformations)
        if transformations
        else 0.0
    )
    anomaly_factor = (
        sum(record.score for record in anomalies) / len(anomalies)
        if anomalies
        else 0.0
    )

    score = (0.35 * transformation_factor) + (0.25 * confidence_factor) + (0.4 * anomaly_factor)
    return round(min(1.0, max(0.0, score)), 4)


def package_stage7(
    *,
    canonical_input: Any,
    stage1_profile: Any | None = None,
    stage2_decoded: Any | None = None,
    stage3_normalized: Any | None = None,
    stage4_resolved: Any | None = None,
    stage5_defragmented: Any | None = None,
) -> Stage7ProvenanceEnvelope:
    s1 = map_as(stage1_profile)
    s2 = map_as(stage2_decoded)
    s3 = map_as(stage3_normalized)
    s4 = map_as(stage4_resolved)
    s5 = map_as(stage5_defragmented)
    s6 = map_as(canonical_input)

    original_text = str(s6.get("original_text", ""))
    canonical_text = str(s6.get("canonical_text", original_text))

    transformations = _collect_transformations(s2, s3, s4, s5, s6)
    anomalies = _detect_anomalies(
        stage1=s1,
        stage2=s2,
        stage3=s3,
        stage4=s4,
        stage5=s5,
        stage6=s6,
        transformation_count=len(transformations),
    )
    risk_score = _compute_risk_score(transformations, anomalies)

    return Stage7ProvenanceEnvelope(
        original_text=original_text,
        canonical_text=canonical_text,
        transformations=transformations,
        anomalies=anomalies,
        risk_score=risk_score,
        normalization_diff=_make_unified_diff(original_text, canonical_text),
        metadata={
            "transformations_applied": len(transformations),
            "anomalies_detected": len(anomalies),
            "decode_layers": len(s2.get("steps", [])),
        },
    )
