from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not path.exists():
        return entries

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                entries.append(obj)
    return entries

def _extract_scores(detector_results: Any, primary_detectors: list[str] | None = None) -> list[float]:
    scores: list[float] = []
    if not isinstance(detector_results, dict):
        return scores

    for key, value in detector_results.items():
        if primary_detectors and not any(key.startswith(p) for p in primary_detectors):
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, (int, float)):
                    scores.append(float(item))
        elif isinstance(value, (int, float)):
            scores.append(float(value))
    return scores


def _format_duration(start_time: datetime, end_time: datetime) -> str:
    total_seconds = max(int((end_time - start_time).total_seconds()), 0)
    if total_seconds == 0 and end_time > start_time:
        total_seconds = 1
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _parse_report_timestamp(raw_value: Any) -> datetime | None:
    if not isinstance(raw_value, str):
        return None

    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo or timezone.utc
        parsed = parsed.replace(tzinfo=local_tz)

    return parsed.astimezone(timezone.utc)


def build_garak_eval_record(
    run_id: str,
    report_log: Path,
    hit_log: Path | None,
    target_model: str,
    strategy: str,
    defenses_active: str,
) -> dict[str, object]:
    report_entries = _parse_jsonl(report_log)
    hit_entries = _parse_jsonl(hit_log) if hit_log else []

    attempts = [e for e in report_entries 
            if e.get("entry_type") == "attempt" and e.get("status") == 2]
    attempt_ids = [str(e.get("uuid")) for e in attempts if e.get("uuid")]
    total_attempts = len(set(attempt_ids)) if attempt_ids else len(attempts)

    SECONDARY_DETECTORS = ["mitigation.MitigationBypass",
                           "productkey.Win5x5"]

    hit_attempts_from_report = 0

    scored_attempts: set[str] = set()
    for entry in attempts:
        attempt_id = str(entry.get("uuid") or "")
        if attempt_id and attempt_id in scored_attempts:
            continue

        scores = _extract_scores(
            entry.get("detector_results", {}),
            primary_detectors=[k for k in entry.get("detector_results", {}) 
                            if k not in SECONDARY_DETECTORS]
        )

        max_score = max(scores) if scores else 0.0
        if max_score > 0.0:
            hit_attempts_from_report += 1
            if attempt_id:
                scored_attempts.add(attempt_id)

    hitlog_by_attempt: dict[str, float] = {}
    for entry in hit_entries:
        attempt_id = str(entry.get("attempt_id") or "")
        score = entry.get("score")
        if not attempt_id or not isinstance(score, (int, float)):
            continue
        current = hitlog_by_attempt.get(attempt_id, 0.0)
        hitlog_by_attempt[attempt_id] = max(current, float(score))

    hit_attempts_from_hitlog = len([score for score in hitlog_by_attempt.values() if score > 0.0])

    if total_attempts > 0:
        if hit_attempts_from_report > 0:
            hit_attempts = hit_attempts_from_report
        else:
            hit_attempts = min(total_attempts, hit_attempts_from_hitlog)
        success_rate = (hit_attempts / total_attempts) / 20
    else:
        hit_attempts = hit_attempts_from_report or hit_attempts_from_hitlog
        success_rate = 0.0

    start_ts_raw = next((e.get("start_time") for e in report_entries if e.get("entry_type") == "init"), None)
    end_ts_raw = next((e.get("end_time") for e in report_entries if e.get("entry_type") == "completion"), None)

    start_ts = _parse_report_timestamp(start_ts_raw)
    end_ts = _parse_report_timestamp(end_ts_raw)

    if start_ts is None:
        start_ts = datetime.fromtimestamp(report_log.stat().st_ctime, tz=timezone.utc)
    if end_ts is None:
        end_ts = datetime.fromtimestamp(report_log.stat().st_mtime, tz=timezone.utc)

    return {
        "run_id": run_id,
        "timestamp": start_ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "target_model": target_model,
        "strategy": strategy,
        "defenses_active": defenses_active,
        "success_rate": round(success_rate, 2),
        "duration": _format_duration(start_ts, end_ts),
    }
