from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_results(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    results = data.get("results", {})
    if isinstance(results, dict):
        items = results.get("results", [])
        return [item for item in items if isinstance(item, dict)]
    return []


def _is_hit(result: dict[str, Any]) -> bool:
    grading = result.get("gradingResult")
    if isinstance(grading, dict) and "pass" in grading:
        return not bool(grading["pass"])
    return False


def _format_duration(start_time: datetime, end_time: datetime) -> str:
    total_seconds = max(int((end_time - start_time).total_seconds()), 0)
    if total_seconds == 0 and end_time > start_time:
        total_seconds = 1
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_promptfoo_eval_record(
    run_id: str,
    results_paths: list[Path],
    target_model: str,
    strategy: str,
    defenses_active: str,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, object]:
    total = 0
    hits = 0
    for path in results_paths:
        for result in _parse_results(path):
            grading = result.get("gradingResult")
            if not isinstance(grading, dict) or "pass" not in grading:
                continue
            total += 1
            if not bool(grading["pass"]):
                hits += 1

    success_rate = round(hits / total, 4) if total > 0 else 0.0

    return {
        "run_id": run_id,
        "timestamp": start_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "target_model": target_model,
        "strategy": strategy,
        "defenses_active": defenses_active,
        "success_rate": success_rate,
        "duration": _format_duration(start_time, end_time),
    }
