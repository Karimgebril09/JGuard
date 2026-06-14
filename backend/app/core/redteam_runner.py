from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Literal, TypedDict, cast
from uuid import uuid4

from data_generation.custom.cutom_data_generator import CustomGenerator
from evaluation.redteaming_tools.garak.config import DEFAULT_TARGET_NAME, DEFAULT_TARGET_TYPE
from evaluation.redteaming_tools.garak.garak_pipeline import run_with_config_defaults

from backend.app.core import eval_store
from backend.app.core.garak_eval_metrics import build_garak_eval_record
from backend.app.models.redteam_models import LaunchCampaignRequest


class CampaignState(TypedDict):
    status: str
    progress_pct: int
    last_poll: int
    mode: Literal["garak", "custom", "mock"]
    started_at: str
    stop_event: Event
    worker: Thread | None
    log_lines: list[str]


_CAMPAIGNS: dict[str, CampaignState] = {}
_CAMPAIGNS_LOCK = Lock()
LOGGER = logging.getLogger(__name__)
_CUSTOM_METRICS_PATH = Path("data_generation") / "custom" / "outputs" / "metrics.json"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _format_duration(start_time: datetime, end_time: datetime) -> str:
    total_seconds = max(int((end_time - start_time).total_seconds()), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _build_custom_defenses(payload: LaunchCampaignRequest) -> str:
    defenses: list[str] = []
    if payload.obfuscation_protection:
        defenses.append("obfuscation")
    if payload.multi_turn_protection:
        defenses.append("multi_turn")
    if payload.roleplay_protection:
        defenses.append("roleplay")
    if payload.pii_protection:
        defenses.append(f"pii:{payload.pii_strategy or 'enabled'}")
    return ",".join(defenses) if defenses else "none"


def _build_custom_target_model(payload: LaunchCampaignRequest) -> str:
    if payload.target_type and payload.target_model:
        return f"{payload.target_type}/{payload.target_model}"
    return payload.target_model or "unknown"


def _build_custom_eval_record(
    run_id: str,
    payload: LaunchCampaignRequest,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, object]:
    if not _CUSTOM_METRICS_PATH.exists():
        raise FileNotFoundError(f"Custom metrics file not found: {_CUSTOM_METRICS_PATH}")

    data = json.loads(_CUSTOM_METRICS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Custom metrics file is empty or not a list.")

    latest = data[-1]
    if not isinstance(latest, dict):
        raise ValueError("Latest custom metrics entry is not an object.")

    success_rate = float(latest.get("ASR", 0.0))
    successful_attacks = int(latest.get("successful_attacks", 0))
    attack_type = payload.custom_attack_type or "custom"

    return {
        "run_id": run_id,
        "timestamp": start_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "target_model": _build_custom_target_model(payload),
        "strategy": f"custom_{attack_type}",
        "defenses_active": _build_custom_defenses(payload),
        "success_rate": round(success_rate, 4),
        "vulnerabilities": successful_attacks,
        "duration": _format_duration(start_time, end_time),
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": successful_attacks,
    }


def _append_log(state: CampaignState, message: str) -> None:
    log_line = f"[{_utcnow_iso()}] {message}"
    cast(list[str], state["log_lines"]).append(log_line)
    LOGGER.info(log_line)


def _run_garak_campaign(campaign_id: str, payload: LaunchCampaignRequest) -> None:
    with _CAMPAIGNS_LOCK:
        campaign = _CAMPAIGNS.get(campaign_id)
        if campaign is None:
            return
        stop_event = campaign["stop_event"]
        started_at = _parse_iso_datetime(campaign["started_at"])

    with _CAMPAIGNS_LOCK:
        campaign = _CAMPAIGNS.get(campaign_id)
        if campaign is not None:
            campaign["progress_pct"] = 20
            _append_log(campaign, "Starting Garak run with config defaults")
            _append_log(campaign, f"Target: {DEFAULT_TARGET_TYPE}/{DEFAULT_TARGET_NAME}")

    try:
        result = run_with_config_defaults(stop_event=stop_event)
    except Exception as exc:  # noqa: BLE001
        is_cancelled = stop_event.is_set() or "cancel" in str(exc).lower()
        new_status = "stopped" if is_cancelled else "failed"
        with _CAMPAIGNS_LOCK:
            campaign = _CAMPAIGNS.get(campaign_id)
            if campaign is None:
                return
            campaign["status"] = new_status
            campaign["progress_pct"] = 100
            if is_cancelled:
                _append_log(campaign, "Garak campaign stopped by user.")
            else:
                _append_log(campaign, f"Garak campaign failed: {exc}")
        return

    with _CAMPAIGNS_LOCK:
        campaign = _CAMPAIGNS.get(campaign_id)
        if campaign is None:
            return
        campaign["status"] = "completed"
        campaign["progress_pct"] = 100
        _append_log(campaign, f"Garak report log: {result.report_log}")
        if result.hit_log:
            _append_log(campaign, f"Garak hit log: {result.hit_log}")
        _append_log(campaign, f"Saved {result.prompt_count} prompts to {result.output_path}")

    try:
        eval_record = build_garak_eval_record(
            run_id=campaign_id,
            report_log=result.report_log,
            hit_log=result.hit_log,
            target_model=f"{DEFAULT_TARGET_TYPE}/{DEFAULT_TARGET_NAME}",
            strategy="tool_based",
            defenses_active="garak_default",
        )
        eval_store.add_run(eval_record)
        with _CAMPAIGNS_LOCK:
            campaign = _CAMPAIGNS.get(campaign_id)
            if campaign is not None:
                _append_log(campaign, f"Saved evaluation metrics for run_id={campaign_id}")
    except Exception as exc:  # noqa: BLE001
        with _CAMPAIGNS_LOCK:
            campaign = _CAMPAIGNS.get(campaign_id)
            if campaign is not None:
                _append_log(campaign, f"Metrics generation failed: {exc}")


def _start_garak_worker(campaign_id: str, payload: LaunchCampaignRequest) -> None:
    worker = Thread(
        target=_run_garak_campaign,
        args=(campaign_id, payload),
        daemon=True,
        name=f"garak-campaign-{campaign_id}",
    )
    with _CAMPAIGNS_LOCK:
        campaign = _CAMPAIGNS.get(campaign_id)
        if campaign is None:
            return
        campaign["worker"] = worker
    worker.start()


def _run_custom_campaign(campaign_id: str, payload: LaunchCampaignRequest) -> None:
    with _CAMPAIGNS_LOCK:
        campaign = _CAMPAIGNS.get(campaign_id)
        if campaign is None:
            return
        stop_event = campaign["stop_event"]
        started_at = _parse_iso_datetime(campaign["started_at"])

    if stop_event.is_set():
        with _CAMPAIGNS_LOCK:
            campaign = _CAMPAIGNS.get(campaign_id)
            if campaign is not None:
                campaign["status"] = "stopped"
                campaign["progress_pct"] = 100
                _append_log(campaign, "Custom campaign was stopped before start")
        return

    with _CAMPAIGNS_LOCK:
        campaign = _CAMPAIGNS.get(campaign_id)
        if campaign is not None:
            campaign["progress_pct"] = 20
            _append_log(campaign, "Starting custom redteaming campaign")

    try:
        if payload.num_samples is None or payload.num_samples <= 0:
            raise ValueError("num_samples must be a positive integer for strategy='custom'.")

        required_custom_fields = {
            "custom_attack_type": payload.custom_attack_type,
            "custom_harm_type": payload.custom_harm_type,
            "judge_model": payload.judge_model,
            "attacker_model": payload.attacker_model,
            "target_model": payload.target_model,
            "judge_type": payload.judge_type,
            "attacker_type": payload.attacker_type,
            "target_type": payload.target_type,
        }
        missing_fields = [name for name, value in required_custom_fields.items() if not value]
        if missing_fields:
            raise ValueError(
                "Missing required fields for strategy='custom': " + ", ".join(missing_fields)
            )

        generator = CustomGenerator(
            attack_type=payload.custom_attack_type,
            harm_type=payload.custom_harm_type,
            judge=payload.judge_model,
            attacker=payload.attacker_model,
            target=payload.target_model,
            type_judge=payload.judge_type,
            type_attacker=payload.attacker_type,
            type_target=payload.target_type,
            api_key_judge=payload.judge_api_key,
            api_key_attacker=payload.attacker_api_key,
            api_key_target=payload.target_api_key,
            base_url_judge=payload.judge_base_url,
            base_url_attacker=payload.attacker_base_url,
            base_url_target=payload.target_base_url,
            activate_role_playing_defense=bool(payload.roleplay_protection),
            activate_obfuscation_defense=bool(payload.obfuscation_protection),
            activate_pii=bool(payload.pii_protection),
            activate_multi_turn=bool(payload.multi_turn_protection),
            pii_masking_strategy=payload.pii_strategy,
        )
        with _CAMPAIGNS_LOCK:
            campaign = _CAMPAIGNS.get(campaign_id)
            if campaign is not None:
                campaign["progress_pct"] = 60
                _append_log(campaign, f"Generating dataset with num_samples={payload.num_samples}")

        generator.generate_dataset(num_trials=payload.num_samples)

        eval_record = _build_custom_eval_record(
            run_id=campaign_id,
            payload=payload,
            start_time=started_at,
            end_time=datetime.now(timezone.utc),
        )
        eval_store.add_run(eval_record)
    except Exception as exc:  # noqa: BLE001
        is_cancelled = stop_event.is_set() or "cancel" in str(exc).lower()
        new_status = "stopped" if is_cancelled else "failed"
        with _CAMPAIGNS_LOCK:
            campaign = _CAMPAIGNS.get(campaign_id)
            if campaign is None:
                return
            campaign["status"] = new_status
            campaign["progress_pct"] = 100
            if is_cancelled:
                _append_log(campaign, "Custom campaign stopped by user.")
            else:
                _append_log(campaign, f"Custom campaign failed: {exc}")
        return

    with _CAMPAIGNS_LOCK:
        campaign = _CAMPAIGNS.get(campaign_id)
        if campaign is None:
            return
        campaign["status"] = "completed"
        campaign["progress_pct"] = 100
        _append_log(campaign, "Custom campaign completed")
        _append_log(campaign, f"Saved evaluation metrics for run_id={campaign_id}")


def _start_custom_worker(campaign_id: str, payload: LaunchCampaignRequest) -> None:
    worker = Thread(
        target=_run_custom_campaign,
        args=(campaign_id, payload),
        daemon=True,
        name=f"custom-campaign-{campaign_id}",
    )
    with _CAMPAIGNS_LOCK:
        campaign = _CAMPAIGNS.get(campaign_id)
        if campaign is None:
            return
        campaign["worker"] = worker
    worker.start()


def launch_campaign(payload: LaunchCampaignRequest) -> dict[str, object]:
    campaign_id = str(uuid4())
    now = _utcnow_iso()

    strategy = payload.strategy
    tool_framework = payload.tool_framework
    is_garak_campaign = strategy == "tool_based" and tool_framework == "garak"
    is_custom_campaign = strategy == "custom"

    with _CAMPAIGNS_LOCK:
        _CAMPAIGNS[campaign_id] = CampaignState(
            status="running",
            progress_pct=5,
            last_poll=0,
            mode="garak" if is_garak_campaign else ("custom" if is_custom_campaign else "mock"),
            started_at=now,
            stop_event=Event(),
            worker=None,
            log_lines=[
                f"[{now}] Campaign initialized",
                f"Strategy: {strategy}",
                f"Tool framework: {tool_framework or 'none'}",
            ],
        )

        campaign = _CAMPAIGNS[campaign_id]
        if is_garak_campaign:
            _append_log(campaign, "Dispatching Garak runner")
        elif is_custom_campaign:
            _append_log(campaign, "Dispatching custom redteaming runner")
        else:
            _append_log(campaign, "Using mock runner (real execution only enabled for tool_based + garak)")

    if is_garak_campaign:
        _start_garak_worker(campaign_id, payload)
    elif is_custom_campaign:
        _start_custom_worker(campaign_id, payload)

    return {
        "campaign_id": campaign_id,
        "status": "started",
        "timestamp": now,
    }


def get_campaign_status(campaign_id: str) -> dict[str, object]:
    with _CAMPAIGNS_LOCK:
        campaign = _CAMPAIGNS.get(campaign_id)
        if campaign is None:
            raise KeyError(f"Unknown campaign_id '{campaign_id}'.")

        if campaign["status"] == "running" and campaign["mode"] == "mock":
            campaign["progress_pct"] = min(campaign["progress_pct"] + 15, 100)
            _append_log(campaign, f"Progress {campaign['progress_pct']}%")
            if campaign["progress_pct"] >= 100:
                campaign["status"] = "completed"
                _append_log(campaign, "Campaign completed")

        log_lines = campaign["log_lines"]
        last_poll = campaign["last_poll"]
        new_lines = log_lines[last_poll:]
        campaign["last_poll"] = len(log_lines)

        return {
            "campaign_id": campaign_id,
            "status": campaign["status"],
            "log_lines": new_lines,
            "progress_pct": campaign["progress_pct"],
        }


def stop_campaign(campaign_id: str) -> dict[str, object]:
    with _CAMPAIGNS_LOCK:
        campaign = _CAMPAIGNS.get(campaign_id)
        if campaign is None:
            raise KeyError(f"Unknown campaign_id '{campaign_id}'.")

        stop_event = campaign["stop_event"]
        stop_event.set()
        campaign["status"] = "stopped"
        campaign["progress_pct"] = max(campaign["progress_pct"], 100)
        _append_log(campaign, "Campaign stopped by user")

    return {
        "success": True,
        "campaign_id": campaign_id,
        "status": "stopped",
    }
