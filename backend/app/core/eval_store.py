import csv
import io
import json
from pathlib import Path
from typing import Any, cast


_RUNS_STORE_PATH = Path("outputs") / "eval_runs.json"


def _ensure_store_exists() -> None:
    _RUNS_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _RUNS_STORE_PATH.exists():
        _RUNS_STORE_PATH.write_text("[]", encoding="utf-8")


def _load_runs() -> list[dict[str, object]]:
    _ensure_store_exists()
    try:
        data = json.loads(_RUNS_STORE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass
    return []


def _save_runs(runs: list[dict[str, object]]) -> None:
    _ensure_store_exists()
    _RUNS_STORE_PATH.write_text(json.dumps(runs, indent=2), encoding="utf-8")


def _run_success_rate(run: dict[str, object]) -> float:
    return float(cast(Any, run["success_rate"]))


def add_run(run: dict[str, object]) -> None:
    runs = _load_runs()
    runs.append(run)
    _save_runs(runs)


def get_summary() -> dict[str, object]:
    runs = _load_runs()
    if not runs:
        return {
            "total_campaigns": 0,
            "avg_jailbreak_success_rate": 0.0,
            "defense_blocked_sweeps_pct": 0.0,
        }

    total_campaigns = len(runs)
    avg_success = sum(_run_success_rate(run) for run in runs) / total_campaigns
    blocked_pct = round(1 - avg_success, 4)
    return {
        "total_campaigns": total_campaigns,
        "avg_jailbreak_success_rate": round(avg_success, 4),
        "defense_blocked_sweeps_pct": blocked_pct,
    }


def get_attack_trends() -> list[dict[str, object]]:
    runs = _load_runs()
    return [
        {
            "run_id": str(run["run_id"]),
            "success_rate": _run_success_rate(run),
        }
        for run in runs
    ]


def get_runs() -> list[dict[str, object]]:
    runs = _load_runs()
    fields = [
        "run_id",
        "timestamp",
        "target_model",
        "strategy",
        "defenses_active",
        "success_rate",
        "duration",
    ]
    return [{field: run[field] for field in fields} for run in runs]


def compare_runs(baseline_run_id: str, compare_run_id: str) -> dict[str, object]:
    runs = _load_runs()
    baseline = next((run for run in runs if run["run_id"] == baseline_run_id), None)
    compare = next((run for run in runs if run["run_id"] == compare_run_id), None)
    if baseline is None or compare is None:
        raise KeyError("One or both run IDs were not found.")

    base_success = _run_success_rate(baseline)
    compare_success = _run_success_rate(compare)

    return {
        "jailbreak_success_rate": {
            "base": base_success,
            "compare": compare_success,
            "delta": round(compare_success - base_success, 4),
        },
        "assessment_duration": {
            "base": str(baseline["duration"]),
            "compare": str(compare["duration"]),
        },
    }


def export_runs_as_json() -> bytes:
    return json.dumps(get_runs(), indent=2).encode("utf-8")


def export_runs_as_csv() -> bytes:
    runs = get_runs()
    if not runs:
        return b""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(runs[0].keys()))
    writer.writeheader()
    writer.writerows(runs)
    return output.getvalue().encode("utf-8")
